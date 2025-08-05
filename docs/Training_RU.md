
# :computer: Как обучать/дообучать Real-ESRGAN

- [Обучение Real-ESRGAN](#train-real-esrgan)
  - [Обзор](#overview)
  - [Подготовка датасета](#dataset-preparation)
  - [Обучение Real-ESRNet](#Train-Real-ESRNet)
  - [Обучение Real-ESRGAN](#Train-Real-ESRGAN)
- [Дообучение Real-ESRGAN на вашем собственном датасете](#Finetune-Real-ESRGAN-on-your-own-dataset)
  - [Генерация ухудшенных изображений на лету](#Generate-degraded-images-on-the-fly)
  - [Использование собственных парных данных](#use-your-own-paired-data)

[English](Training.md) **|** [Русский](Training_RU.md)

## Обучение Real-ESRGAN

### Обзор

Обучение разделено на два этапа. Эти два этапа используют одинаковый процесс синтеза данных и pipeline обучения, за исключением функций потерь. А именно,

1. Сначала мы обучаем Real-ESRNet с L1 loss, начиная с предобученной модели ESRGAN.
1. Затем используем обученную модель Real-ESRNet в качестве инициализации генератора и обучаем Real-ESRGAN с комбинацией L1 loss, перцептуальной потери и GAN loss.

### Подготовка датасета

Для обучения мы используем датасеты DF2K (DIV2K и Flickr2K) + OST. Требуются только HR изображения. <br>
Вы можете скачать их по ссылкам:

1. DIV2K: http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
2. Flickr2K: https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar
3. OST: https://openmmlab.oss-cn-hangzhou.aliyuncs.com/datasets/OST_dataset.zip

Вот шаги по подготовке данных.

#### Шаг 1: [Опционально] Генерация мульти-масштабных изображений

Для датасета DF2K мы используем стратегию мульти-масштаба, *т.е.* понижаем разрешение HR изображений, чтобы получить несколько Ground-Truth изображений с разными масштабами. <br>
Вы можете использовать скрипт [scripts/generate_multiscale_DF2K.py](scripts/generate_multiscale_DF2K.py) для генерации мульти-масштабных изображений. <br>
Обратите внимание, что этот шаг можно пропустить, если хотите быстро попробовать.

```bash
python scripts/generate_multiscale_DF2K.py --input datasets/DF2K/DF2K_HR --output datasets/DF2K/DF2K_multiscale
```

#### Шаг 2: [Опционально] Вырезка подизображений

Затем мы нарезаем изображения DF2K на подизображения для ускорения ввода-вывода и обработки.<br>
Этот шаг опционален, если у вас достаточно быстродействия или ограничено место на диске.

Вы можете использовать скрипт [scripts/extract_subimages.py](scripts/extract_subimages.py). Пример:

```bash
 python scripts/extract_subimages.py --input datasets/DF2K/DF2K_multiscale --output datasets/DF2K/DF2K_multiscale_sub --crop_size 400 --step 200
```

#### Шаг 3: Подготовка txt файла с метаинформацией

Нужно подготовить txt файл с путями к изображениям. Ниже пример из `meta_info_DF2Kmultiscale+OST_sub.txt` (так как у разных пользователей могут быть разные разбиения подизображений, этот файл может не подойти, и вам нужно подготовить свой собственный):

```txt
DF2K_HR_sub/000001_s001.png
DF2K_HR_sub/000001_s002.png
DF2K_HR_sub/000001_s003.png
...
```

Вы можете использовать скрипт [scripts/generate_meta_info.py](scripts/generate_meta_info.py) для генерации txt файла. <br>
Можно объединять несколько папок в один meta_info txt. Пример:

```bash
 python scripts/generate_meta_info.py --input datasets/DF2K/DF2K_HR datasets/DF2K/DF2K_multiscale --root datasets/DF2K datasets/DF2K --meta_info datasets/DF2K/meta_info/meta_info_DF2Kmultiscale.txt
```

### Обучение Real-ESRNet

1. Скачайте предобученную модель [ESRGAN](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth) в `experiments/pretrained_models`.
    ```bash
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth -P experiments/pretrained_models
    ```
1. Отредактируйте файл опций `options/train_realesrnet_x4plus.yml` следующим образом:
    ```yml
    train:
        name: DF2K+OST
        type: RealESRGANDataset
        dataroot_gt: datasets/DF2K  # измените на корневой путь вашей папки
        meta_info: realesrgan/meta_info/meta_info_DF2Kmultiscale+OST_sub.txt  # измените на ваш сгенерированный meta info txt
        io_backend:
            type: disk
    ```
1. Если хотите выполнять валидацию во время обучения, раскомментируйте соответствующие строки и измените пути:
    ```yml
      # Раскомментируйте для валидации
      # val:
      #   name: validation
      #   type: PairedImageDataset
      #   dataroot_gt: path_to_gt
      #   dataroot_lq: path_to_lq
      #   io_backend:
      #     type: disk

    ...

      # Раскомментируйте для валидации
      # настройки валидации
      # val:
      #   val_freq: !!float 5e3
      #   save_img: True

      #   metrics:
      #     psnr: # название метрики, может быть любое
      #       type: calculate_psnr
      #       crop_border: 4
      #       test_y_channel: false
    ```
1. Перед основным обучением можно запустить в режиме `--debug`, чтобы проверить, всё ли в порядке. Мы используем 4 GPU:
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py -opt options/train_realesrnet_x4plus.yml --launcher pytorch --debug
    ```

    Обучение на **одном GPU** в режиме *debug*:
    ```bash
    python realesrgan/train.py -opt options/train_realesrnet_x4plus.yml --debug
    ```
1. Основное обучение. Используем 4 GPU. Аргумент `--auto_resume` позволяет автоматически возобновить обучение при необходимости.
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py -opt options/train_realesrnet_x4plus.yml --launcher pytorch --auto_resume
    ```

    Обучение на **одном GPU**:
    ```bash
    python realesrgan/train.py -opt options/train_realesrnet_x4plus.yml --auto_resume
    ```

### Обучение Real-ESRGAN

1. После обучения Real-ESRNet у вас появится файл `experiments/train_RealESRNetx4plus_1000k_B12G4_fromESRGAN/model/net_g_1000000.pth`. Если нужно указать путь к другой предобученной модели, измените значение `pretrain_network_g` в файле опций `train_realesrgan_x4plus.yml`.
1. Отредактируйте файл опций `train_realesrgan_x4plus.yml` аналогично предыдущему.
1. Перед основным обучением можно запустить в режиме `--debug`, чтобы проверить, всё ли в порядке. Используем 4 GPU:
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py -opt options/train_realesrgan_x4plus.yml --launcher pytorch --debug
    ```

    Обучение на **одном GPU** в режиме *debug*:
    ```bash
    python realesrgan/train.py -opt options/train_realesrgan_x4plus.yml --debug
    ```
1. Основное обучение. Используем 4 GPU. Аргумент `--auto_resume` для автоматического возобновления.
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py -opt options/train_realesrgan_x4plus.yml --launcher pytorch --auto_resume
    ```

    Обучение на **одном GPU**:
    ```bash
    python realesrgan/train.py -opt options/train_realesrgan_x4plus.yml --auto_resume
    ```

## Дообучение Real-ESRGAN на вашем собственном датасете

Вы можете дообучить Real-ESRGAN на своем датасете. Обычно процесс дообучения делится на два варианта:

1. [Генерация ухудшенных изображений на лету](#Generate-degraded-images-on-the-fly)
1. [Использование собственных **парных** данных](#Use-paired-training-data)

### Генерация ухудшенных изображений на лету

Требуются только изображения высокого разрешения. Изображения низкого качества генерируются во время обучения с помощью процесса деградации, описанного в Real-ESRGAN.

**1. Подготовьте датасет**

Смотрите [раздел выше](#dataset-preparation) для подробностей.

**2. Скачайте предобученные модели**

Скачайте модели в `experiments/pretrained_models`.

- *RealESRGAN_x4plus.pth*:
    ```bash
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models
    ```

- *RealESRGAN_x4plus_netD.pth*:
    ```bash
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth -P experiments/pretrained_models
    ```

**3. Дообучение**

Отредактируйте [options/finetune_realesrgan_x4plus.yml](options/finetune_realesrgan_x4plus.yml) особенно секцию `datasets`:

```yml
train:
    name: DF2K+OST
    type: RealESRGANDataset
    dataroot_gt: datasets/DF2K  # измените на корневой путь вашей папки
    meta_info: realesrgan/meta_info/meta_info_DF2Kmultiscale+OST_sub.txt  # измените на ваш сгенерированный meta info txt
    io_backend:
        type: disk
```

Используем 4 GPU. Аргумент `--auto_resume` для автоматического возобновления.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py -opt options/finetune_realesrgan_x4plus.yml --launcher pytorch --auto_resume
```

Дообучение на **одном GPU**:
```bash
python realesrgan/train.py -opt options/finetune_realesrgan_x4plus.yml --auto_resume
```

### Использование собственных парных данных

Вы также можете дообучить RealESRGAN на своих парных данных. Это похоже на дообучение ESRGAN.

**1. Подготовьте датасет**

Предположим, у вас есть две папки:

- **gt папка** (Ground-truth, изображения высокого разрешения): *datasets/DF2K/DIV2K_train_HR_sub*
- **lq папка** (низкого качества, изображения низкого разрешения): *datasets/DF2K/DIV2K_train_LR_bicubic_X4_sub*

Затем можно подготовить meta_info txt с помощью скрипта [scripts/generate_meta_info_pairdata.py](scripts/generate_meta_info_pairdata.py):

```bash
python scripts/generate_meta_info_pairdata.py --input datasets/DF2K/DIV2K_train_HR_sub datasets/DF2K/DIV2K_train_LR_bicubic_X4_sub --meta_info datasets/DF2K/meta_info/meta_info_DIV2K_sub_pair.txt
```

**2. Скачайте предобученные модели**

Скачайте модели в `experiments/pretrained_models`.

- *RealESRGAN_x4plus.pth*
    ```bash
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models
    ```

- *RealESRGAN_x4plus_netD.pth*
    ```bash
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth -P experiments/pretrained_models
    ```

**3. Дообучение**

Отредактируйте [options/finetune_realesrgan_x4plus_pairdata.yml](options/finetune_realesrgan_x4plus_pairdata.yml), особенно секцию `datasets`:

```yml
train:
    name: DIV2K
    type: RealESRGANPairedDataset
    dataroot_gt: datasets/DF2K  # измените на корневой путь вашей папки
    dataroot_lq: datasets/DF2K  # измените на корневой путь вашей папки
    meta_info: datasets/DF2K/meta_info/meta_info_DIV2K_sub_pair.txt  # измените на ваш сгенерированный meta info txt
    io_backend:
        type: disk
```

Используем 4 GPU. Аргумент `--auto_resume` для автоматического возобновления.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py -opt options/finetune_realesrgan_x4plus_pairdata.yml --launcher pytorch --auto_resume
```

Дообучение на **одном GPU**:
```bash
python realesrgan/train.py -opt options/finetune_realesrgan_x4plus_pairdata.yml --auto_resume
```
