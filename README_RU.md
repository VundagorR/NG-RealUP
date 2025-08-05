<p align="center">
  <img src="assets/realesrgan_logo.png" height=120>
</p>

## <div align="center"><b><a href="README.md">English</a> | <a href="README_RU.md">Русский</a></b></div>

<div align="center">

👀[**Демо**](#-demos-videos) **|** 🚩[**Обновления**](#-updates) **|** ⚡[**Использование**](#-quick-inference) **|** 🏰[**Модельный парк**](docs/model_zoo.md) **|** 🔧[Установка](#-dependencies-and-installation)  **|** 💻[Обучение](docs/Training.md) **|** ❓[FAQ](docs/FAQ.md) **|** 🎨[Вклад в проект](docs/CONTRIBUTING.md)

[![download](https://img.shields.io/github/downloads/xinntao/Real-ESRGAN/total.svg)](https://github.com/xinntao/Real-ESRGAN/releases)
[![PyPI](https://img.shields.io/pypi/v/realesrgan)](https://pypi.org/project/realesrgan/)
[![Open issue](https://img.shields.io/github/issues/xinntao/Real-ESRGAN)](https://github.com/xinntao/Real-ESRGAN/issues)
[![Closed issue](https://img.shields.io/github/issues-closed/xinntao/Real-ESRGAN)](https://github.com/xinntao/Real-ESRGAN/issues)
[![LICENSE](https://img.shields.io/github/license/xinntao/Real-ESRGAN.svg)](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE)
[![python lint](https://github.com/xinntao/Real-ESRGAN/actions/workflows/pylint.yml/badge.svg)](https://github.com/xinntao/Real-ESRGAN/blob/master/.github/workflows/pylint.yml)
[![Publish-pip](https://github.com/xinntao/Real-ESRGAN/actions/workflows/publish-pip.yml/badge.svg)](https://github.com/xinntao/Real-ESRGAN/blob/master/.github/workflows/publish-pip.yml)

</div>

🔥 **Модель AnimeVideo-v3 (маленькая модель для аниме-видео)**. Подробнее см. [[*anime video models*](docs/anime_video_model.md)] и [[*сравнения*](docs/anime_comparisons.md)]<br>
🔥 **RealESRGAN_x4plus_anime_6B** для аниме-изображений **(аниме-модель)**. Подробнее см. [[*anime_model*](docs/anime_model.md)]

<!-- 1. Вы можете попробовать на нашем сайте: [ARC Demo](https://arc.tencent.com/en/ai-demos/imgRestore) (сейчас поддерживается только RealESRGAN_x4plus_anime_6B) -->
1. :boom: **Обновлено** онлайн-демо на Replicate: [![Replicate](https://img.shields.io/static/v1?label=Demo&message=Replicate&color=blue)](https://replicate.com/xinntao/realesrgan)
1. Онлайн-демо Colab для Real-ESRGAN: [![Colab](https://img.shields.io/static/v1?label=Demo&message=Colab&color=orange)](https://colab.research.google.com/drive/1k2Zod6kSHEvraybHl50Lys0LerhyTMCo?usp=sharing) **|** Онлайн-демо Colab для Real-ESRGAN (**аниме-видео**): [![Colab](https://img.shields.io/static/v1?label=Demo&message=Colab&color=orange)](https://colab.research.google.com/drive/1yNl9ORUxxlL4N0keJa2SEPB61imPQd1B?usp=sharing)
1. Портативные [Windows](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip) / [Linux](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip) / [MacOS](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-macos.zip) **исполняемые файлы для GPU Intel/AMD/Nvidia**. Подробнее см. [здесь](#portable-executable-files-ncnn). Реализация на ncnn доступна в [Real-ESRGAN-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan)
<!-- 1. Вы можете посмотреть улучшенные анимации на [Tencent Video](https://v.qq.com/s/topic/v_child/render/fC4iyCAM.html). Добро пожаловать на [Tencent Video Anime Restoration](https://v.qq.com/s/topic/v_child/render/fC4iyCAM.html) -->

Real-ESRGAN направлен на разработку **практичных алгоритмов для общего восстановления изображений и видео**.<br>
Мы разработали мощный ESRGAN до практического решения вопроса восстановления изображений (Real-ESRGAN), обученного на полностью синтетических данных (под синтетическими данными обычно понимаются изображения, созданные искусственно, а не полученные с помощью фотографии реальных объектов). То есть, для обучения модели Real-ESRGAN используют сгенерированные или искусственно искажённые изображения — например, высококачественные изображения, к которым применяют различные преобразования (шум, размытие, уменьшение разрешения и т.д.) с целью получить "плохие" версии. Модель учится восстанавливать эти "плохие" изображения обратно в высококачественные.
Real-ESRGAN — это расширение ESRGAN, адаптированное для практического применения, обученное именно на таких искусственно сгенерированных данных, что позволяет модели лучше работать с реальными фотографиями, восстанавливая их качество.

🌌 Спасибо за ваши ценные отзывы и предложения. Все отзывы обновляются в [feedback.md](docs/feedback.md).

---
Если Real-ESRGAN вам помог, пожалуйста, поставьте ⭐ этому репозиторию или порекомендуйте его друзьям 😊 <br>
Другие рекомендуемые проекты:<br>
▶️ [GFPGAN](https://github.com/TencentARC/GFPGAN): Практичный алгоритм восстановления лиц в реальных условиях <br>
▶️ [BasicSR](https://github.com/xinntao/BasicSR): Открытый набор инструментов для восстановления изображений и видео<br>
▶️ [facexlib](https://github.com/xinntao/facexlib): Коллекция полезных функций для работы с лицами.<br>
▶️ [HandyView](https://github.com/xinntao/HandyView): Просмотрщик изображений на базе PyQt5, удобный для просмотра и сравнения <br>
▶️ [HandyFigure](https://github.com/xinntao/HandyFigure): Открытый исходный код для работы с рисунками из научных статей <br>

---

### 📖 Real-ESRGAN: Обучение реальному слепому супер-разрешению с использованием полностью синтетических данных

> [[Статья](https://arxiv.org/abs/2107.10833)] &emsp; [[Видео на YouTube](https://www.youtube.com/watch?v=fxHWoDSSvSc)] &emsp; [[Объяснение на Bilibili](https://www.bilibili.com/video/BV1H34y1m7sS/)] &emsp; [[Постер](https://xinntao.github.io/projects/RealESRGAN_src/RealESRGAN_poster.pdf)] &emsp; [[Презентация (PPT)](https://docs.google.com/presentation/d/1QtW6Iy8rm8rGLsJ0Ldti6kP-7Qyzy6XL/edit?usp=sharing&ouid=109799856763657548160&rtpof=true&sd=true)]<br>
> [Синтао Ванг](https://xinntao.github.io/), Лянбин Се, [Чао Донг](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ), [Йинг Шань](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en) <br>
> Лаборатория Tencent ARC; Шэньчжэньский институт передовых технологий, Китайская академия наук

<p align="center">
  <img src="assets/teaser.jpg">
</p>

---

## 🚩 Обновления

- ✅ Добавлена модель **realesr-general-x4v3** — маленькая модель для общих сцен. Также поддерживается опция **-dn** для балансировки шума (чтобы избежать чрезмерного сглаживания). **-dn** — сокращение от "denoising strength" (сила шумоподавления).
- ✅ Обновлена модель **RealESRGAN AnimeVideo-v3**. Подробнее см. [anime video models](docs/anime_video_model.md) и [сравнения](docs/anime_comparisons.md).
- ✅ Добавлены маленькие модели для аниме-видео. Подробнее в [anime video models](docs/anime_video_model.md).
- ✅ Добавлена реализация на ncnn: [Real-ESRGAN-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan).
- ✅ Добавлена модель [*RealESRGAN_x4plus_anime_6B.pth*](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth), оптимизированная для **аниме**-изображений с гораздо меньшим размером модели. Подробнее и сравнения с [waifu2x](https://github.com/nihui/waifu2x-ncnn-vulkan) в [**anime_model.md**](docs/anime_model.md).
- ✅ Поддержка дообучения на собственных данных или парных данных (*т.е.* дообучение ESRGAN). Подробнее см. [здесь](docs/Training.md#Finetune-Real-ESRGAN-on-your-own-dataset).
- ✅ Интеграция с [GFPGAN](https://github.com/TencentARC/GFPGAN) для поддержки **улучшения лиц**.
- ✅ Интеграция с платформой [Huggingface Spaces](https://huggingface.co/spaces) с помощью [Gradio](https://github.com/gradio-app/gradio). Онлайн-демо: [Gradio Web Demo](https://huggingface.co/spaces/akhaliq/Real-ESRGAN). Спасибо [@AK391](https://github.com/AK391).
- ✅ Поддержка произвольного масштаба с помощью `--outscale` (фактически дополнительно изменяет размер выходных данных с помощью `LANCZOS4`). Добавлена модель *RealESRGAN_x2plus.pth*.
- ✅ [Код инференса](inference_realesrgan.py) поддерживает: 1) опции **tile** (разбиение на блоки); 2) изображения с **альфа-каналом**; 3) **оттенки серого**; 4) **16-битные** изображения.
- ✅ Код обучения опубликован. Подробное руководство доступно в [Training.md](docs/Training.md).

---

## 👀 Демо видео

#### Bilibili

- [Фрагмент из "Путешествие на Запад"](https://www.bilibili.com/video/BV1ja41117zb)
- [Anime dance cut — аниме-танец](https://www.bilibili.com/video/BV1wY4y1L7hT/)
- [Фрагмент из "One Piece"](https://www.bilibili.com/video/BV1i3411L7Gy/)

#### YouTube

## 🔧 Зависимости и установка

- Python >= 3.7 (рекомендуется использовать [Anaconda](https://www.anaconda.com/download/#linux) или [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)

### Установка

1. Клонируйте репозиторий

    ```bash
    git clone https://github.com/xinntao/Real-ESRGAN.git
    cd Real-ESRGAN
    ```

2. Установите зависимости

    ```bash
    # Установка basicsr - https://github.com/xinntao/BasicSR
    # Используется для обучения и инференса
    pip install basicsr
    # facexlib и gfpgan для улучшения лиц
    pip install facexlib
    pip install gfpgan
    pip install -r requirements.txt
    python setup.py develop
    ```

---

## ⚡ Быстрый запуск (Inference)

Обычно существует три способа использовать Real-ESRGAN для восстановления:

1. [Онлайн-инференс](#online-inference)  
2. [Портативные исполняемые файлы (NCNN)](#portable-executable-files-ncnn)  
3. [Python-скрипт](#python-script)

### Онлайн-инференс

1. Вы можете попробовать на нашем сайте: [ARC Demo](https://arc.tencent.com/en/ai-demos/imgRestore) (сейчас поддерживается только RealESRGAN_x4plus_anime_6B)  
2. [Colab Demo](https://colab.research.google.com/drive/1k2Zod6kSHEvraybHl50Lys0LerhyTMCo?usp=sharing) для Real-ESRGAN **|** [Colab Demo](https://colab.research.google.com/drive/1yNl9ORUxxlL4N0keJa2SEPB61imPQd1B?usp=sharing) для Real-ESRGAN (**аниме-видео**).

### Портативные исполняемые файлы (NCNN)

Вы можете скачать [Windows](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip) / [Linux](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip) / [MacOS](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-macos.zip) **исполняемые файлы для GPU Intel/AMD/Nvidia**.

Этот исполняемый файл **портативен** и содержит все необходимые бинарники и модели. Не требуется установка CUDA или среды PyTorch.<br>

Вы можете просто запустить команду (пример для Windows, подробнее в README каждого исполняемого файла):

```bash
./realesrgan-ncnn-vulkan.exe -i input.jpg -o output.png -n model_name
```
Доступно пять моделей:

realesrgan-x4plus (по умолчанию)
realesrnet-x4plus
realesrgan-x4plus-anime (оптимизирована для аниме, небольшой размер модели)
realesr-animevideov3 (анимационное видео)

Для выбора другой модели используйте аргумент -n, например: `./realesrgan-ncnn-vulkan.exe -i input.jpg -o output.png -n realesrnet-x4plus`

#### Использование портативных исполняемых файлов
1. Подробнее смотрите в Real-ESRGAN-ncnn-vulkan.
2. Обратите внимание, что не все функции (например, outscale) поддерживаются, как в python-скрипте inference_realesrgan.py.

```console
Usage: realesrgan-ncnn-vulkan.exe -i infile -o outfile [options]...

  -h                   show this help
  -i input-path        input image path (jpg/png/webp) or directory
  -o output-path       output image path (jpg/png/webp) or directory
  -s scale             upscale ratio (can be 2, 3, 4. default=4)
  -t tile-size         tile size (>=32/0=auto, default=0) can be 0,0,0 for multi-gpu
  -m model-path        folder path to the pre-trained models. default=models
  -n model-name        model name (default=realesr-animevideov3, can be realesr-animevideov3 | realesrgan-x4plus | realesrgan-x4plus-anime | realesrnet-x4plus)
  -g gpu-id            gpu device to use (default=auto) can be 0,1,2 for multi-gpu
  -j load:proc:save    thread count for load/proc/save (default=1:2:2) can be 1:2,2,2:2 for multi-gpu
  -x                   enable tta mode"
  -f format            output image format (jpg/png/webp, default=ext/png)
  -v                   verbose output
```
Обратите внимание, что возможны артефакты на стыках блоков (а также результаты могут немного отличаться от реализации на PyTorch), так как этот исполняемый файл сначала разбивает входное изображение на несколько тайлов, обрабатывает их отдельно, а затем склеивает обратно.

### Python-скрипт

### Использование python-скрипта

1. Вы можете использовать X4 модель для **произвольного размера вывода** с помощью аргумента outscale. Программа дополнительно выполнит недорогую операцию ресайза после вывода Real-ESRGAN.


```console
Usage: python inference_realesrgan.py -n RealESRGAN_x4plus -i infile -o outfile [options]...

A common command: python inference_realesrgan.py -n RealESRGAN_x4plus -i infile --outscale 3.5 --face_enhance

  -h                   show this help
  -i --input           Input image or folder. Default: inputs
  -o --output          Output folder. Default: results
  -n --model_name      Model name. Default: RealESRGAN_x4plus
  -s, --outscale       The final upsampling scale of the image. Default: 4
  --suffix             Suffix of the restored image. Default: out
  -t, --tile           Tile size, 0 for no tile during testing. Default: 0
  --face_enhance       Whether to use GFPGAN to enhance face. Default: False
  --fp32               Use fp32 precision during inference. Default: fp16 (half precision).
  --ext                Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto
```

#### Восстановление обычных изображений

Скачайте предобученную модель: RealESRGAN_x4plus.pth

```bash
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights
```

Запуск инференса:

```bash
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs --face_enhance
```

Результаты будут в папке results

#### Восстановление аниме-изображений
<p align="center"> <img src="https://raw.githubusercontent.com/xinntao/public-figures/master/Real-ESRGAN/cmp_realesrgan_anime_1.png"> </p>
Предобученная модель: RealESRGAN_x4plus_anime_6B<br>
Подробнее и сравнения с waifu2x в anime_model.md

```bash
# скачать модель
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P weights
# запуск инференса
python inference_realesrgan.py -n RealESRGAN_x4plus_anime_6B -i inputs
```

Результаты будут в папке results

---

## BibTeX

    @InProceedings{wang2021realesrgan,
        author    = {Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
        title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
        booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
        date      = {2021}
    }

## 📧 Контакты

Если у вас есть вопросы, пожалуйста, пишите на xintao.wang@outlook.com или xintaowang@tencent.com.

<!---------------------------------- Проекты, использующие Real-ESRGAN --------------------------->
## 🧩 Проекты, использующие Real-ESRGAN

Если вы разрабатываете или используете Real-ESRGAN в своих проектах, будем рады узнать об этом.

- NCNN-Android: [RealSR-NCNN-Android](https://github.com/tumuyan/RealSR-NCNN-Android) by [tumuyan](https://github.com/tumuyan)
- VapourSynth: [vs-realesrgan](https://github.com/HolyWu/vs-realesrgan) by [HolyWu](https://github.com/HolyWu)
- NCNN: [Real-ESRGAN-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan)

&nbsp;&nbsp;&nbsp;&nbsp;**Графические интерфейсы (GUI)**

- [Waifu2x-Extension-GUI](https://github.com/AaronFeng753/Waifu2x-Extension-GUI) by [AaronFeng753](https://github.com/AaronFeng753)
- [Squirrel-RIFE](https://github.com/Justin62628/Squirrel-RIFE) by [Justin62628](https://github.com/Justin62628)
- [Real-GUI](https://github.com/scifx/Real-GUI) by [scifx](https://github.com/scifx)
- [Real-ESRGAN_GUI](https://github.com/net2cn/Real-ESRGAN_GUI) by [net2cn](https://github.com/net2cn)
- [Real-ESRGAN-EGUI](https://github.com/WGzeyu/Real-ESRGAN-EGUI) by [WGzeyu](https://github.com/WGzeyu)
- [anime_upscaler](https://github.com/shangar21/anime_upscaler) by [shangar21](https://github.com/shangar21)
- [Upscayl](https://github.com/upscayl/upscayl) by [Nayam Amarshe](https://github.com/NayamAmarshe) and [TGS963](https://github.com/TGS963)

## 🤗 Благодарности

Спасибо всем участникам и контрибьюторам.

- [AK391](https://github.com/AK391): Интеграция RealESRGAN в [Huggingface Spaces](https://huggingface.co/spaces) с помощью [Gradio](https://github.com/gradio-app/gradio). Смотрите [Gradio Web Demo](https://huggingface.co/spaces/akhaliq/Real-ESRGAN).
- [2ji3150](https://github.com/2ji3150): Спасибо за детальные и ценные отзывы/предложения.
- [VundagorR](https://github.com/VundagorR): Перевод README.md на Русский язык.


