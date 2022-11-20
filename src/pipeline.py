import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd

from utils.data_mappers import DatasetMapper
from utils.preprocessing import Preprocessor

import os
from datetime import datetime
from tqdm import tqdm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter


class Pipeline:
    def __init__(
        self,
        name,
        model,
        batch_size,
        workers,
        root_dir,
        train_image_data_file,
        test_image_data_file,
        preprocessor,
        **kwargs
    ):
        self.name = name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")

        self.model = model
        self.batch_size = batch_size

        self.root_dir = root_dir
        self.train_image_data_file = train_image_data_file
        self.test_image_data_file = test_image_data_file

        self.preprocessor = preprocessor

        self.args = kwargs

        # Set training device (CUDA-GPU / CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Training Device: {}".format(self.device))
        self.model.to(self.device)

        # Creating dataset mapper instances
        train_set = DatasetMapper(
            root_dir, train_image_data_file, self.preprocessor, augment=True, **kwargs
        )
        test_set = DatasetMapper(
            root_dir, test_image_data_file, self.preprocessor, augment=False, **kwargs
        )

        # Creating dataset loader to load data parallelly
        self.train_loader = DataLoader(
            train_set, batch_size=self.batch_size, num_workers=workers, shuffle=True
        )
        self.test_loader = DataLoader(
            test_set, batch_size=self.batch_size, num_workers=workers
        )

        # Create summary writers for tensorboard logs
        self.train_writer = SummaryWriter("logs/fit/" + self.name + "/train")
        self.valid_writer = SummaryWriter("logs/fit/" + self.name + "/validation")

    def train(self, **kwargs):
        assert "num_epochs" in kwargs.keys(), "Provide num_epochs in **kwargs"
        self.epochs = kwargs["num_epochs"]

        # Setting learning rate
        if "lr" in kwargs.keys():
            lr = kwargs["lr"]
        else:
            lr = 0.001

        # Setting optimzer
        if "optim" in kwargs.keys():
            optimizer = kwargs["optim"](self.model.parameters(), lr=lr)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Learning rate scheduler for changing learning rate during training
        if "lr_scheduler" in kwargs.keys():
            lr_scheduler = kwargs["lr_scheduler"](optimizer)
        elif "step_size_func" in kwargs.keys():
            step_size_func = kwargs["step_size_func"]
            lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, step_size_func)
        else:
            lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda e: 1)

        # Loss function for minimizing loss
        assert (
            "loss_func" in kwargs.keys() and "loss_func_with_grad" in kwargs.keys()
        ), "loss_func and loss_func_with_grad must be present in **kwargs"
        loss_func_with_grad = kwargs["loss_func_with_grad"]
        loss_func = kwargs["loss_func"]

        knlg_distill = False

        if "knlg_distill" in kwargs.keys():
            print('Pure knowledge Distillation')
            knlg_distill = kwargs["knlg_distill"]
            if knlg_distill:
                
                assert "teacher_model" in kwargs.keys(), "Provide teacher_model in **kwargs"
                # load saved teacher model
                teacher_model = kwargs["teacher_model"]
                

                if "knlg_distill_weight" in kwargs.keys():
                    knlg_distill_weight = kwargs["knlg_distill_weight"]
                else:
                    knlg_distill_weight = 0.5
                
                print('Knowledge Distillation Weight: {}'.format(knlg_distill_weight))

        else:
            print('No kwargs Pure knowledge Distillation')
            knlg_distill = False
            knlg_distill_weight = 0




        # if 'loss_func' in kwargs.keys() and 'loss_func_with_grad' in kwargs.keys():
        # loss_func_with_grad = kwargs['loss_func_with_grad']
        # loss_func = kwargs['loss_func']
        # else:
        #     # cross entropy loss for learning features
        #     loss_func_with_grad = nn.CrossEntropyLoss()
        #     loss_func = F.cross_entropy
        #     # will require different loss for learning features
        #     # loss_func = nn.MSELoss()

        training_log = {"errors": [], "scores": []}
        validation_log = {"errors": [], "scores": []}

        # Training
        # pbar = tqdm(range(self.epochs), desc="Training epoch")
        for epoch in range(1, self.epochs + 1):
            print("lr: {}".format(lr_scheduler.get_last_lr()))

            # Putting model in training mode to calculate back gradients
            self.model.train()

            ys = []
            y_preds = []

            # Batch-wise optimization
            pbar = tqdm(self.train_loader, desc="Training epoch {}".format(epoch))
            for x_train, y_train in pbar:
                x = x_train.type(torch.FloatTensor).to(self.device)
                # print(type(x_train), type(x), x.shape)
                y = y_train.type(torch.LongTensor).to(self.device)
                # print(type(y_train), type(y), y.shape)

                # Forward pass
                y_pred = self.model(x)

                # Clearing previous epoch gradients
                optimizer.zero_grad()

                # Calculating loss
                if knlg_distill:
                        y_pred = self.model(x)
                        
                        y_pred_teacher = torch.argmax(teacher_model(x), dim=1)
                        
                        # loss against the ground truth
                        loss1 = loss_func_with_grad(y_pred, y)
                        # loss against the teacher model
                        # print(y.shape)
                        # print(y_pred_teacher.shape)
                        # print(y_pred.shape)
                        loss2 = loss_func(y_pred, y_pred_teacher)
                        
                        # weighted sum of the two losses
                        loss = knlg_distill_weight * loss1 + (1 - knlg_distill_weight) * loss2

                else:
                    y_pred = self.model(x)
                    loss = loss_func_with_grad(y_pred, y)
                    


                # Backward pass to calculate gradients
                loss.backward()

                # Update gradients
                optimizer.step()

                if knlg_distill:
                    # Save/show loss per step of training batches
                    pbar.set_postfix({"training error": loss.item()})
                    training_log["errors"].append({"epoch": epoch, "loss": loss.item(), "loss1": loss1.item(), "loss2": loss2.item()})
                else:
                    # Save/show loss per step of training batches
                    pbar.set_postfix({"training error": loss.item()})
                    training_log["errors"].append({"epoch": epoch, "loss": loss.item()})

                self.train_writer.add_scalar("loss", loss.item(), epoch)
                self.train_writer.flush()

                # Save y_true and y_pred in lists for calculating epoch-wise scores
                ys += list(y.cpu().detach().numpy())
                y_preds += list(torch.argmax(y_pred, dim=1).cpu().detach().numpy())

            # Update learning rate as defined above
            lr_scheduler.step()

            # Save/show training scores per epoch
            training_scores = []
            if "score_functions" in kwargs:
                for score_func in kwargs["score_functions"]:
                    score = score_func["func"](ys, y_preds)
                    training_scores.append({score_func["name"]: score})
                    self.train_writer.add_scalar(score_func["name"], score, epoch)

                self.train_writer.flush()
                print(
                    "epoch:{}, Training Scores:{}".format(epoch, training_scores),
                    flush=True,
                )
                training_log["scores"].append(
                    {"epoch": epoch, "scores": training_scores}
                )

            ys = []
            y_preds = []

            # Putting model in evaluation mode to stop calculating back gradients
            self.model.eval()
            with torch.no_grad():
                for x_test, y_test in tqdm(
                    self.test_loader, desc="Validation epoch {}".format(epoch)
                ):
                    x = x_test.type(torch.FloatTensor).to(self.device)
                    y = y_test.type(torch.LongTensor).to(self.device)

                    if knlg_distill:
                        y_pred = self.model(x)
                        y_pred_teacher = teacher_model(x)
                        y_pred_teacher = torch.argmax(y_pred_teacher, dim=1)
                        # loss against the ground truth
                        loss1 = loss_func_with_grad(y_pred, y)
                        # loss against the teacher model
                        loss2 = loss_func(y_pred, y_pred_teacher)
                        # weighted sum of the two losses
                        loss = knlg_distill_weight * loss1 + (1 - knlg_distill_weight) * loss2
                        validation_log["errors"].append(
                        {"epoch": epoch, "loss": loss.item(), "loss1": loss1.item(), "loss2": loss2.item()}
                    )

                    else:
                        y_pred = self.model(x)
                        loss = loss_func_with_grad(y_pred, y)
                    
                    
                        # Save/show loss per batch of validation data
                        # pbar.set_postfix({"test error": loss})
                        validation_log["errors"].append(
                            {"epoch": epoch, "loss": loss.item()}
                        )

                    self.valid_writer.add_scalar("loss", loss.item(), epoch)

                    # Save y_true and y_pred in lists for calculating epoch-wise scores
                    ys += list(y.cpu().detach().numpy())
                    y_preds += list(torch.argmax(y_pred, dim=1).cpu().detach().numpy())

            # Save/show validation scores per epoch
            validation_scores = []
            if "score_functions" in kwargs:
                for score_func in kwargs["score_functions"]:
                    score = score_func["func"](ys, y_preds)
                    validation_scores.append({score_func["name"]: score})
                    self.valid_writer.add_scalar(score_func["name"], score, epoch)

                self.valid_writer.flush()
                print(
                    "epoch:{}, Validation Scores:{}".format(epoch, validation_scores),
                    flush=True,
                )
                validation_log["scores"].append(
                    {"epoch": epoch, "scores": validation_scores}
                )

            # Saving model at specified checkpoints
            if "save_checkpoints" in kwargs.keys():
                if epoch % kwargs["save_checkpoints"]["epoch"] == 0:
                    # os.makedirs(
                    #     kwargs["save_checkpoints"]["path"]
                    #     if type(kwargs["save_checkpoints"]["path"]) == str
                    #     else kwargs["save_checkpoints"]["path"](epoch)
                    # )
                    chkp_path = os.path.join(
                        kwargs["save_checkpoints"]["path"],
                        self.name,
                        "checkpoints",
                        "{}".format(epoch),
                    )
                    os.makedirs(chkp_path)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss,
                        },
                        chkp_path + "/model.pth",
                    )

        return training_log, validation_log
