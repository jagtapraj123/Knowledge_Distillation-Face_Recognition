# Face Recognition Models Comparison

## Objective

Repository created for CS-57300 Data Mining Course at Purdue University.

In this project we plan to compare how deep learning models perform in dearth of labelled data.

We plan to compare performance of following models on face recognition dataset with ~20 images of each subject.

1. Shallow CNN - trianed end-to-end
2. Inception-v3 - trained end-to-end
3. Pretrained Inception-v3 with transfer learning
4. Knowledge Distillation - pretrained Inception-v3 as teacher and Shallow CNN as student - trained end-to-end
5. Knowledge Distillation - pretrained Inception-v3 as teacher and Shallow CNN as student - trained for identifying facial features

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

## Code Format

Run [`Black`](https://black.readthedocs.io/en/stable/) code formatter before committing code to git repository.

Run following command in root folder of repository:

```sh
black .
```

Using same code formatter will help prevent error such as tab-space conversions and will code look uniform and readable throughout.

## Team

1. Mansi Shinde
2. Pranav Patil
3. Raj Jagtap
4. Rucha Deshpande
