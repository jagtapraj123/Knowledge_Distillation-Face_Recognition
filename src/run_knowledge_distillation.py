# To be written

import pandas as pd
import torch
import models.resnet_50 as ResNet
import models.conv_net as ConvNet
from pipeline_two_step import TwoStepPipeline
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

    teacher_model_name = "resnet_scratch"
    teacher_model = ResNet.resnet50(
        num_classes=len(data["label"].unique()), include_top=False
    )
    # model = ResNet.resnet50

    load_state_dict(teacher_model, "saved_models/resnet50_ft_weight.pkl")

    for params in teacher_model.parameters():
        params.requires_grad = False

    teacher_model.eval()

    teacher_func = partial(teacher_function, model=teacher_model)

    model_name = "convnet_resnet_kd"
    model = ConvNet.ConvNet(num_classes=len(data["label"].unique()), include_top=False)

    trainer = TwoStepPipeline(
        name=model_name,
        model=model,
        batch_size=16,
        workers=16,
        url="https://thispersondoesnotexist.com/image",
        root_dir="../dataset",
        train_image_data_file="../dataset/train_image_data.csv",
        test_image_data_file="../dataset/test_image_data.csv",
        preprocessor=preprocessor,
        teacher_func=teacher_func,
        pretrain_size=1024,
        num_classes=len(data["label"].unique()),
        include_top=True,
    )

    num_epochs = 100

    lr_s1 = 0.0001
    lr_s2 = 0.001

    step_size_func_s1 = lambda e: 1 / math.sqrt(1 + e)
    step_size_func_s2 = lambda e: 1 / math.sqrt(1 + e)

    loss_func_s1_with_grad = torch.nn.MSELoss()
    loss_func_s1 = torch.nn.functional.mse_loss

    loss_func_s2_with_grad = torch.nn.CrossEntropyLoss()
    loss_func_s2 = torch.nn.functional.cross_entropy

    score_functions_s1 = [
        {"name": "mae_s1", "func": mean_absolute_error},
        {"name": "mse_s1", "func": mean_squared_error},
        {"name": "r2_score_s1", "func": r2_score},
    ]

    score_functions_s2 = [
        {"name": "f1_score_micro_s2", "func": partial(f1_score, average="micro")},
        {"name": "f1_score_macro_s2", "func": partial(f1_score, average="macro")},
        {
            "name": "precision_micro_s2",
            "func": partial(precision_score, average="micro"),
        },
        {
            "name": "precision_macro_s2",
            "func": partial(precision_score, average="macro"),
        },
        {"name": "recall_micro_s2", "func": partial(recall_score, average="micro")},
        {"name": "recall_macro_s2", "func": partial(recall_score, average="macro")},
        {"name": "accuracy_s2", "func": accuracy_score},
    ]

    training_log, validation_log = trainer.train(
        num_epochs=num_epochs,
        lr_s1=lr_s1,
        lr_s2=lr_s2,
        step_size_func_s1=step_size_func_s1,
        step_size_func_s2=step_size_func_s2,
        loss_func_s1_with_grad=loss_func_s1_with_grad,
        loss_func_s1=loss_func_s1,
        loss_func_s2_with_grad=loss_func_s2_with_grad,
        loss_func_s2=loss_func_s2,
        score_functions_s1=score_functions_s1,
        score_functions_s2=score_functions_s2,
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
