# License-Plate-Recognition
## Introdution
The main content of this project is to identify the license plates of Chinese vehicles. We use cascaded classifiers to extract license plates and We use deep learning for character extraction.
![车牌识别](https://pic.imgdb.cn/item/60d7ec535132923bf8653066.jpg)

## Requirement
- Tensorflow2.0
- A machine with NVIDIA GPU
- keras
- opencv
- PIL
- numpy
- importlib

## Quick Start
- Download the project
- Run the ```demo.py```

## Content
```
|-- README.md
|-- __pycache__
|-- demo.py        # 运行测试程序（只能识别蓝色车牌）
|-- detect.py      # 车牌提取喝模型等函数
|-- model          # 模型文件
|   |-- cascade.xml
|   |-- model12.h5
|   |-- ocr_plate_all_gru.h5
|   |-- platech.ttf
|-- test_images
|-- results        # 放置结果图
```

## Results
![results](https://pic.imgdb.cn/item/60d7ecb15132923bf8673fe6.jpg)
