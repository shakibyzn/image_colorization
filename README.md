# image_colorization
image Colorization is the practice of estimating color channels for grayscale images or video frames. In this project, we make use of CIELAB color space to estimated the a*, b* components given the luminance component. For this project, we implemented three models, which are as follow:  
1- Deep Koalarization: Image Colorization using CNNs and Inception-V3   
2- Attention U-Net  
3- Fusion of Attention U-Net model with Inception-V3 as a global feature extractor  

## Setup

```console
$ python3 -m venv ~/.virtualenvs/env
$ source ~/.virtualenvs/env/bin/activate
```

### Updating project dependencies

```console
# And to install the packages
$ pip install -r requirements.txt
```

## Getting the dataset
*  Places 205 dataset [Data](http://places2.csail.mit.edu/download.html)


## Running the code

### Default arguments

```shell
EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=1e-3
```

### Training from scratch

```shell
1- Deep Koalarization model
python main.py --epochs=100 --lr=0.0005 --seed=5 --batch_size=32 --model_name=koalarization
2- Attention U-Net
main.py --epochs=100 --lr=0.0005 --seed=5 --batch_size=32 --model_name=attention_unet
3- Fusion of Attention U-Net model with Inception-V3
Attention U-Net
main.py --epochs=100 --lr=0.0005 --seed=5 --batch_size=32 --model_name=attention_unet_fusion
```

## License

MIT License

Copyright (c) [2022]    
[Back To The Top](#image_colorization)