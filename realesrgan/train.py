# flake8: noqa
import os.path as osp
from basicsr.train import train_pipeline

import realesrgan.archs
import realesrgan.data
import realesrgan.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    try:
        train_pipeline(root_path)
    except Exception as e:
        print(f'[ERROR] Training pipeline failed.\nОшибка в процессе обучения.\n{e}')
        raise e
