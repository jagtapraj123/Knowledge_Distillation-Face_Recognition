# To be written

import pandas as pd
import torch
import models.resnet_50 as ResNet
import models.conv_net as ConvNet
from pipeline_two_step import TwoStepPipeline
from pipeline import Pipeline
from utils.preprocessing import Preprocessor
from utils.model_utils import load_state_dict
from utils.teacher_function import teacher_function
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from functools import partial
import pickle

if __name__ == "__main__":

    data_file = "../dataset/train_image_data.csv"

    data = pd.read_csv(data_file)

    preprocessor = Preprocessor()

    # print(preprocessor.make_random_combinations(10, p_transformations= {'rotate': 0.5, 'scale': 0.5, 'flip': 0.5, 'gaussian_blur': 0.5, 'color_jitter': 0.5, 'random_erasing': 0}))
    # img = preprocessor.get('g', "../dataset/105_classes_pins_dataset/pins_Anne Hathaway/Anne Hathaway0_293.jpg", color_jitter=None, is_url=False, gaussian_blur=1)
    # exit()

    # load model from pth file
    teacher_model_name = "resnet_scratch"
    teacher_model = ResNet.resnet50(
        num_classes=len(data["label"].unique()), include_top=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.load_state_dict(torch.load("saved_models/resnet50_ft_weight.pth", map_location=device)['model_state_dict'])
    teacher_model.to(device)
    # print(teacher_model)

    teacher_model.eval()




    model_name = "convnet_resnet_kd"
    model = ConvNet.ConvNet(num_classes=len(data["label"].unique()), include_top=True)

    trainer = Pipeline(
        name=model_name,
        model=model,
        batch_size=16,
        workers=16,
        url="https://thispersondoesnotexist.com/image",
        root_dir="../dataset",
        train_image_data_file="../dataset/train_image_data.csv",
        test_image_data_file="../dataset/test_image_data.csv",
        preprocessor=preprocessor,
        teacher_model=teacher_model,
        pretrain_size=1024,
        num_classes=len(data["label"].unique()),
        include_top=True,
    )


    num_epochs = 100
    lr = 0.0001
    step_size_func = lambda e: 1 / math.sqrt(1 + e)

    loss_func_with_grad = torch.nn.CrossEntropyLoss()
    loss_func = torch.nn.functional.cross_entropy

    
    score_functions = [
        {"name": "f1_score_micro", "func": partial(f1_score, average="micro")},
        {"name": "f1_score_macro", "func": partial(f1_score, average="macro")},
        {"name": "precision_micro", "func": partial(precision_score, average="micro")},
        {"name": "precision_macro", "func": partial(precision_score, average="macro")},
        {"name": "recall_micro", "func": partial(recall_score, average="micro")},
        {"name": "recall_macro", "func": partial(recall_score, average="macro")},
        {"name": "accuracy", "func": accuracy_score},
    ]
    

    training_log, validation_log = trainer.train(
        num_epochs=num_epochs,
        lr=lr,
        step_size_func=step_size_func,
        loss_func_with_grad=loss_func_with_grad,
        loss_func=loss_func,
        score_functions=score_functions,
        knlg_distill=True,
        knlg_distill_weight=0.5,
        teacher_model = teacher_model,
        save_checkpoints={
            "epoch": 5,
            # "path": lambda e: "saved_models/{}/checkpoints/{}/".format(model_name, e),
            "path": "saved_models/",
        },
    )

    with open(trainer.name + "training_log.pkl", "wb") as f:
        pickle.dump(training_log, f)

    with open(trainer.name + "validation_log.pkl", "wb") as f:
        pickle.dump(validation_log, f)
