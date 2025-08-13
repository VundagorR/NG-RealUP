#!/usr/bin/env python

import os
import subprocess
import time
from setuptools import setup, find_packages

version_file = 'realesrgan/version.py'

def get_git_hash():
    def _minimal_ext_cmd(cmd):
        env = {k: os.environ.get(k) for k in ['SYSTEMROOT', 'PATH', 'HOME'] if os.environ.get(k) is not None}
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
    except Exception:
        sha = 'unknown'

    return sha

def get_hash():
    if os.path.exists('.git'):
        return get_git_hash()[:7]
    else:
        return 'unknown'

def write_version_py():
    content = """# GENERATED VERSION FILE
# TIME: {}
__version__ = '{}'
__gitsha__ = '{}'
version_info = ({})
"""
    if not os.path.exists('VERSION'):
        short_version = '0.0.0'
    else:
        with open('VERSION', 'r') as f:
            short_version = f.read().strip()
    version_info = ', '.join([x if x.isdigit() else f'"{x}"' for x in short_version.split('.')])
    sha = get_hash()

    with open(version_file, 'w') as f:
        f.write(content.format(time.asctime(), short_version, sha, version_info))

def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']

def get_requirements(filename='requirements.txt'):
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, filename), 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

if __name__ == '__main__':
    write_version_py()
    setup(
        name='realesrgan',
        version=get_version(),
        description='Real-ESRGAN aims at developing Practical Algorithms for General Image Restoration',
        long_description=open('README.md', encoding='utf-8').read(),
        long_description_content_type='text/markdown',
        author='Xintao Wang',
        author_email='xintao.wang@outlook.com',
        keywords='computer vision, pytorch, image restoration, super-resolution, esrgan, real-esrgan',
        url='https://github.com/xinntao/Real-ESRGAN',
        include_package_data=True,
        packages=find_packages(exclude=('options', 'datasets', 'experiments', 'results', 'tb_logger', 'wandb')),
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        license='BSD-3-Clause License',
        setup_requires=['cython>=0.29.21', 'numpy>=1.20.0'],
        install_requires=get_requirements(),
        zip_safe=False
    )
