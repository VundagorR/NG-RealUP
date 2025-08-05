import cv2
import math
import numpy as np
import os
import queue
import threading
import torch
from basicsr.utils.download_util import load_file_from_url
from torch.nn import functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class RealESRGANer():
    """A helper class for upsampling images with RealESRGAN.

    Args:
        scale (int): Upsampling scale factor used in the networks. It is usually 2 or 4.
        model_path (str or list): The path(s) to the pretrained model(s). Can be URL(s).
        model (nn.Module): The defined network.
        tile (int): Tile size for processing large images. 0 for no tiling.
        tile_pad (int): Padding for tiles to reduce border artifacts.
        pre_pad (int): Padding for input images to reduce border artifacts.
        half (bool): Use half precision (fp16) during inference.
        device (torch.device or str): Device to run inference on.
        gpu_id (int or None): GPU id to use if device not specified.
    """

    def __init__(self,
                 scale,
                 model_path,
                 dni_weight=None,
                 model=None,
                 tile=0,
                 tile_pad=10,
                 pre_pad=10,
                 half=False,
                 device=None,
                 gpu_id=None):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.half = half

        # Device setup with explicit CUDA check
        if device is not None:
            self.device = torch.device(device)
        elif gpu_id is not None and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu_id}')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(f'[INFO] Using device: {self.device}')

        # Load model weights
        if isinstance(model_path, list):
            # Deep network interpolation (DNI)
            assert len(model_path) == len(dni_weight), 'model_path and dni_weight must have the same length.'
            loadnet = self.dni(model_path[0], model_path[1], dni_weight)
        else:
            if model_path.startswith('https://'):
                try:
                    model_path = load_file_from_url(
                        url=model_path, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
                except Exception as e:
                    print(f'[ERROR] Failed to download model:\nНе удалось загрузить модель.\nError: {e}')
                    raise e
            try:
                loadnet = torch.load(model_path, map_location='cpu')
            except Exception as e:
                print(f'[ERROR] Failed to load model weights from {model_path}.\nОшибка загрузки весов модели.\n{e}')
                raise e

        # Prefer params_ema if exists
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'

        try:
            model.load_state_dict(loadnet[keyname], strict=True)
        except Exception as e:
            print(f'[ERROR] Failed to load state dict into model.\nОшибка загрузки state dict в модель.\n{e}')
            raise e

        model.eval()
        self.model = model.to(self.device)
        if self.half:
            self.model = self.model.half()

    def dni(self, net_a_path, net_b_path, dni_weight, key='params', loc='cpu'):
        """Deep network interpolation.

        Paper: Deep Network Interpolation for Continuous Imagery Effect Transition
        """
        try:
            net_a = torch.load(net_a_path, map_location=loc)
            net_b = torch.load(net_b_path, map_location=loc)
        except Exception as e:
            print(f'[ERROR] Failed to load DNI models.\nОшибка загрузки моделей для DNI.\n{e}')
            raise e

        for k, v_a in net_a[key].items():
            net_a[key][k] = dni_weight[0] * v_a + dni_weight[1] * net_b[key][k]
        return net_a

    def pre_process(self, img):
        """Pre-process image: convert to tensor, pad, etc."""
        # Convert numpy HWC to CHW tensor
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = img.unsqueeze(0).to(self.device)
        if self.half:
            self.img = self.img.half()

        # pre_pad padding
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), mode='reflect')

        # mod pad for scale divisibility
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        else:
            self.mod_scale = None

        if self.mod_scale is not None:
            _, _, h, w = self.img.size()
            self.mod_pad_h = (self.mod_scale - h % self.mod_scale) if h % self.mod_scale != 0 else 0
            self.mod_pad_w = (self.mod_scale - w % self.mod_scale) if w % self.mod_scale != 0 else 0
            self.img = F.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), mode='reflect')

    def process(self):
        """Run model inference on self.img"""
        self.output = self.model(self.img)

    def tile_process(self):
        """Tile-based inference to avoid OOM on large images."""
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        for y in range(tiles_y):
            for x in range(tiles_x):
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size

                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y

                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                try:
                    with torch.inference_mode():
                        output_tile = self.model(input_tile)
                except RuntimeError as error:
                    print(f'[ERROR] Runtime error during tile inference: {error}')
                    print('Try reducing tile size (--tile) to avoid CUDA out of memory.')
                    raise error

                print(f'\tTile {y * tiles_x + x + 1}/{tiles_x * tiles_y}')

                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                      output_start_x_tile:output_end_x_tile]

    def post_process(self):
        """Remove padding from output image."""
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]
        return self.output

    def enhance(self, img, outscale=None, alpha_upsampler='realesrgan'):
        """Enhance image with RealESRGAN.

        Args:
            img (np.ndarray): Input image (HWC, BGR or BGRA).
            outscale (float): Optional output scale to resize final output.
            alpha_upsampler (str): Upsampler for alpha channel ('realesrgan' or 'bicubic').

        Returns:
            output (np.ndarray): Enhanced image.
            img_mode (str): Image mode ('RGB', 'RGBA', or 'L').
        """
        h_input, w_input = img.shape[0:2]
        img = img.astype(np.float32)

        if np.max(img) > 256:  # 16-bit image
            max_range = 65535
            print('\tInput is a 16-bit image')
        else:
            max_range = 255

        img = img / max_range

        if len(img.shape) == 2:
            img_mode = 'L'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img_mode = 'RGBA'
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if alpha_upsampler == 'realesrgan':
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = 'RGB'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process main image
        self.pre_process(img)
        if self.tile_size > 0:
            self.tile_process()
        else:
            self.process()

        output_img = self.post_process()
        output_img = output_img.squeeze().float().cpu().clamp(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))  # RGB to BGR

        if img_mode == 'L':
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        # Process alpha channel if RGBA
        if img_mode == 'RGBA':
            if alpha_upsampler == 'realesrgan':
                self.pre_process(alpha)
                if self.tile_size > 0:
                    self.tile_process()
                else:
                    self.process()
                output_alpha = self.post_process()
                output_alpha = output_alpha.squeeze().float().cpu().clamp(0, 1).numpy()
                output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
                output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
            else:
                h, w = alpha.shape[0:2]
                output_alpha = cv2.resize(alpha, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LINEAR)

            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
            output_img[:, :, 3] = output_alpha

        # Convert back to original bit-depth
        if max_range == 65535:
            output = (output_img * 65535.0).round().astype(np.uint16)
        else:
            output = (output_img * 255.0).round().astype(np.uint8)

        # Resize output if outscale differs
        if outscale is not None and outscale != float(self.scale):
            output = cv2.resize(
                output,
                (int(w_input * outscale), int(h_input * outscale)),
                interpolation=cv2.INTER_LANCZOS4)

        return output, img_mode


class PrefetchReader(threading.Thread):
    """Prefetch images.

    Args:
        img_list (list[str]): A image list of image paths to be read.
        num_prefetch_queue (int): Number of prefetch queue.
    """

    def __init__(self, img_list, num_prefetch_queue):
        super().__init__()
        self.que = queue.Queue(num_prefetch_queue)
        self.img_list = img_list

    def run(self):
        for img_path in self.img_list:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            self.que.put(img)
        self.que.put(None)

    def __next__(self):
        next_item = self.que.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class IOConsumer(threading.Thread):
    def __init__(self, opt, que, qid):
        super().__init__()
        self._queue = que
        self.qid = qid
        self.opt = opt

    def run(self):
        while True:
            msg = self._queue.get()
            if isinstance(msg, str) and msg == 'quit':
                break

            output = msg['output']
            save_path = msg['save_path']
            try:
                cv2.imwrite(save_path, output)
            except Exception as e:
                print(f'[ERROR] Failed to save image {save_path}.\nОшибка сохранения изображения.\n{e}')
        print(f'IO worker {self.qid} is done.')
