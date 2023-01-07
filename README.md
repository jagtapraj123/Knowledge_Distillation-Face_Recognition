# Face Recognition Models Comparison

## Objective

Repository created for CS-57300 Data Mining Course at Purdue University.

In this project we plan to compare how deep learning models perform in dearth of labelled data.

We use dataset [Pins Face Recognition](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition). This dataset has images of 105 celebrity faces. To simulate problem of low data we subsample the dataset keeping only 32 images per class for training.

We plan to compare performance of following experiments with different models on the subsampled dataset:

1. Training from Scratch
   1. Shallow CNN - ConvNet
   2. ResNet-50
   3. ResNet-18
2. Transfer Learning
   1. ResNet-50
3. Knowledge Distillation - Feature
   1. Train ResNet-18 using pre-trained ResNet-50 as teacher using MSE
   2. Train ConvNet using pre-trained ResNet-50 as teacher using MSE
   3. Train ResNet-18 using pre-trained ResNet-50 as teacher using BCE
   4. Train ConvNet using pre-trained ResNet-50 as teacher using BCE
4. One-shot-Learning

The purpose of knowledge distillation is to train shallow CNN to identify facial features learned by deep learning models trained on large datasets.

## Installation

1. Creating Virtual Environment:
    - Using Conda:

        ```bash
        conda create --name dm_project python=3.9
        conda activate dm_project
        ```

    - Using Virtualenv:

        ```bash
        virtualenv venv_dm_project
        source venv_dm_project/bin/activate
        ```

2. Installing Python Dependancies:

    ```bash
    pip install -r requirements.txt
    ```

## Dataset Setup

- The code required to get and setup the dataset is provided in the `dataset/` directory.
- Run `dataset.sh`
  - The script downloads the dataset from kaggle, unzip it and subsample it as required by the experiments.
  - Users can change the subsampling number in the shell script.
  - The results in the report were based on experiments with 32 training images per class.

## Downloading Pre-trained Model Weights

|Model|Download link|
| :--- | :---: |
|`resnet50_ft`|[link](https://drive.google.com/open?id=1A94PAAnwk6L7hXdBXLFosB_s0SzEhAFU)|
|`resnet50_scratch`|[link](https://drive.google.com/open?id=1gy9OJlVfBulWkIEnZhGpOLu084RgHw39)|

Download the models in `src/saved_models` as

1. `resnet50_ft_weight.pkl`
2. `resnet50_scratch_weights.pkl`

## Generating Faces for Knowledge Distillation

- The code for generating faces is present in `src/utils/generate_dataset/`.
- Run the `generate.sh` using NVIDIA GPU with CUDA to generate 102400 images of faces.
- The shell script will clone the [StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch.git) repository provided by NVIDIA and run the generate python file to generate faces.
- Users can generate more or less number of faces as per requirement.
- The results in the report were based on experiments with 102400 faces.

## Code Format (Before committing)

Run [`Black`](https://black.readthedocs.io/en/stable/) code formatter before committing code to git repository.

Run following command in root folder of repository:

```sh
black .
```

Using same code formatter will help prevent error such as tab-space conversions and will code look uniform and readable throughout.

## Team

1. Raj Jagtap
2. Pranav Patil
3. Mansi Shinde
4. Rucha Deshpande

## Acknowledgement

- This project was completed under the guidance of Dr. Rajiv Khanna (Purdue University).
- The ResNet-50 model code in file `src/models/resnet.py` is used from [VGGFace2-PyTorch](https://github.com/cydonia999/VGGFace2-pytorch/blob/master/models/resnet.py)
- The pre-trained model ResNet-50 is used from the repository [VGGFace2-PyTorch](https://github.com/cydonia999/VGGFace2-pytorch)
