import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd

from utils.data_mappers import DatasetMapper, DatasetGen
from utils.preprocessing import Preprocessor

import os
from datetime import datetime
from tqdm import tqdm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter


class TwoStepPipeline:
    def __init__(
        self,
        name,
        model,
        batch_size,
        workers,
        url,
        root_dir,
        train_image_data_file,
        test_image_data_file,
        preprocessor,
        teacher_func,
        pretrain_size,
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

        # Creating dataset generator instances
        train_data_gen = DatasetGen(
            url, self.preprocessor, teacher_func, pretrain_size, augment=True, **kwargs
        )

        # Creating dataset mapper instances
        train_set = DatasetMapper(
            root_dir, train_image_data_file, self.preprocessor, augment=True, **kwargs
        )
        test_set = DatasetMapper(
            root_dir, test_image_data_file, self.preprocessor, augment=False, **kwargs
        )

        # Creating dataset loader to load data parallelly
        self.train_data_gen_loader = DataLoader(
            train_data_gen,
            batch_size=self.batch_size,
            num_workers=workers,
            prefetch_factor=4,
        )
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

        # Setting learning rate for step 1
        if "lr_s1" in kwargs.keys():
            lr_s1 = kwargs["lr_s1"]
        else:
            lr_s1 = 0.001

        # Setting learning rate for step 2
        if "lr_s2" in kwargs.keys():
            lr_s2 = kwargs["lr_s2"]
        else:
            lr_s2 = 0.001

        # Setting optimzer for step 1
        if "optim_s1" in kwargs.keys():
            for params in self.model.fc.parameters():
                params.requires_grad = False

            optimizer_s1 = kwargs["optim_s1"](
                filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr_s1
            )

            for params in self.model.fc.parameters():
                params.requires_grad = True
        else:
            for params in self.model.fc.parameters():
                params.requires_grad = False

            optimizer_s1 = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr_s1
            )

            for params in self.model.fc.parameters():
                params.requires_grad = True

        # Setting optimzer for step 2
        if "optim_s2" in kwargs.keys():
            optimizer_s2 = kwargs["optim_s2"](self.model.fc.parameters(), lr=lr_s2)
        else:
            optimizer_s2 = optim.Adam(self.model.fc.parameters(), lr=lr_s2)

        # Learning rate scheduler for step 1 for changing learning rate during training
        if "lr_scheduler_s1" in kwargs.keys():
            lr_scheduler_s1 = kwargs["lr_scheduler_s1"](optimizer_s1)
        elif "step_size_func_s1" in kwargs.keys():
            step_size_func_s1 = kwargs["step_size_func_s1"]
            lr_scheduler_s1 = optim.lr_scheduler.LambdaLR(
                optimizer_s1, step_size_func_s1
            )
        else:
            lr_scheduler_s1 = optim.lr_scheduler.LambdaLR(optimizer_s1, lambda e: 1)

        # Learning rate scheduler for step 2 changing learning rate during training
        if "lr_scheduler_s2" in kwargs.keys():
            lr_scheduler_s2 = kwargs["lr_scheduler_s2"](optimizer_s2)
        elif "step_size_func_s2" in kwargs.keys():
            step_size_func_s2 = kwargs["step_size_func_s2"]
            lr_scheduler_s2 = optim.lr_scheduler.LambdaLR(
                optimizer_s2, step_size_func_s2
            )
        else:
            lr_scheduler_s2 = optim.lr_scheduler.LambdaLR(optimizer_s2, lambda e: 1)

        # Loss function for minimizing loss
        assert (
            "loss_func_s1" in kwargs.keys()
            and "loss_func_s1_with_grad" in kwargs.keys()
        ), "loss_func_s1 and loss_func_s1_with_grad must be present in **kwargs"
        loss_func_s1_with_grad = kwargs["loss_func_s1_with_grad"]
        loss_func_s1 = kwargs["loss_func_s1"]

        assert (
            "loss_func_s2" in kwargs.keys()
            and "loss_func_s2_with_grad" in kwargs.keys()
        ), "loss_func_s2 and loss_func_s2_with_grad must be present in **kwargs"
        loss_func_s2_with_grad = kwargs["loss_func_s2_with_grad"]
        loss_func_s2 = kwargs["loss_func_s2"]

        # if 'loss_func' in kwargs.keys() and 'loss_func_with_grad' in kwargs.keys():
        # loss_func_with_grad = kwargs['loss_func_with_grad']
        # loss_func = kwargs['loss_func']
        # else:
        #     # cross entropy loss for learning features
        #     loss_func_with_grad = nn.CrossEntropyLoss()
        #     loss_func = F.cross_entropy
        #     # will require different loss for learning features
        #     # loss_func = nn.MSELoss()

        training_log = {
            "errors s1": [],
            "errors s2": [],
            "scores s1": [],
            "scores s2": [],
        }
        validation_log = {"errors": [], "scores": []}

        # Training
        # pbar = tqdm(range(self.epochs), desc="Training epoch")
        for epoch in range(1, self.epochs + 1):
            print("lr_s1: {}, lr_s2: {}".format(lr_scheduler_s1.get_last_lr(), lr_scheduler_s2.get_last_lr()))

            # Putting model in training mode to calculate back gradients
            self.model.train()

            # Epoch Step 1:
            # Pre-training
            # Training 1st half using teacher model
            self.model.include_top = False
            for params in self.model.parameters():
                params.requires_grad = True
            for params in self.model.fc.parameters():
                params.requires_grad = False

            ys = []
            y_preds = []

            # Batch-wise optimization
            pbar = tqdm(
                self.train_data_gen_loader, desc="Training S1 epoch {}".format(epoch)
            )
            for x_train, y_train in pbar:
                x = x_train.type(torch.FloatTensor).to(self.device)
                # print(type(x_train), type(x), x.shape)
                y = y_train.type(torch.FloatTensor).to(self.device)
                # print(type(y_train), type(y), y.shape)

                # Forward pass
                y_pred = self.model(x)

                # Clearing previous epoch gradients
                optimizer_s1.zero_grad()

                # Calculating loss
                loss = loss_func_s1_with_grad(y_pred, y)

                # Backward pass to calculate gradients
                loss.backward()

                # Update gradients
                optimizer_s1.step()

                # Save/show loss per step of training batches
                pbar.set_postfix({"training error s1": loss.item()})
                training_log["errors s1"].append(
                    {"epoch": epoch, "loss s1": loss.item()}
                )

                self.train_writer.add_scalar("loss s1", loss.item(), epoch)
                self.train_writer.flush()

                # Save y_true and y_pred in lists for calculating epoch-wise scores
                ys += list(torch.squeeze(y).cpu().detach().numpy())
                y_preds += list(torch.squeeze(y_pred).cpu().detach().numpy())

            # Update learning rate as defined above
            lr_scheduler_s1.step()

            # Save/show training scores per epoch
            training_scores = []
            if "score_functions_s1" in kwargs:
                for score_func in kwargs["score_functions_s1"]:
                    score = score_func["func"](ys, y_preds)
                    training_scores.append({score_func["name"]: score})
                    self.train_writer.add_scalar(score_func["name"], score, epoch)

                self.train_writer.flush()
                print(
                    "epoch:{}, Training S1 Scores:{}".format(epoch, training_scores),
                    flush=True,
                )
                training_log["scores s1"].append(
                    {"epoch": epoch, "scores": training_scores}
                )

            # Epoch Step 2:
            # Fine-tuning
            # Training 2nd half by training on dataset
            self.model.include_top = True
            for params in self.model.parameters():
                params.requires_grad = False
            for params in self.model.fc.parameters():
                params.requires_grad = True

            ys = []
            y_preds = []

            # Batch-wise optimization
            pbar = tqdm(self.train_loader, desc="Training S2 epoch {}".format(epoch))
            for x_train, y_train in pbar:
                x = x_train.type(torch.FloatTensor).to(self.device)
                # print(type(x_train), type(x), x.shape)
                y = y_train.type(torch.LongTensor).to(self.device)
                # print(type(y_train), type(y), y.shape)

                # Forward pass
                y_pred = self.model(x)

                # Clearing previous epoch gradients
                optimizer_s2.zero_grad()

                # Calculating loss
                loss = loss_func_s2_with_grad(y_pred, y)

                # Backward pass to calculate gradients
                loss.backward()

                # Update gradients
                optimizer_s2.step()

                # Save/show loss per step of training batches
                pbar.set_postfix({"training error s2": loss.item()})
                training_log["errors s2"].append(
                    {"epoch": epoch, "loss s2": loss.item()}
                )

                self.train_writer.add_scalar("loss s2", loss.item(), epoch)
                self.train_writer.flush()

                # Save y_true and y_pred in lists for calculating epoch-wise scores
                ys += list(y.cpu().detach().numpy())
                y_preds += list(torch.argmax(y_pred, dim=1).cpu().detach().numpy())

            # Update learning rate as defined above
            lr_scheduler_s2.step()

            # Save/show training scores per epoch
            training_scores = []
            if "score_functions_s2" in kwargs:
                for score_func in kwargs["score_functions_s2"]:
                    score = score_func["func"](ys, y_preds)
                    training_scores.append({score_func["name"]: score})
                    self.train_writer.add_scalar(score_func["name"], score, epoch)

                self.train_writer.flush()
                print(
                    "epoch:{}, Training S2 Scores:{}".format(epoch, training_scores),
                    flush=True,
                )
                training_log["scores s2"].append(
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

                    # Predicting
                    y_pred = self.model(x)

                    # Calculating loss
                    loss = loss_func_s2(y_pred, y)

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
            if "score_functions_s2" in kwargs:
                for score_func in kwargs["score_functions_s2"]:
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
                # if epoch % kwargs["save_checkpoints"]["epoch"] == 0:
                #     for params in self.model.parameters():
                #         params.requires_grad = True
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
                        "optimizer_s1_state_dict": optimizer_s1.state_dict(),
                        "optimizer_s2_state_dict": optimizer_s2.state_dict(),
                        "loss": loss,
                    },
                    chkp_path + "/model.pth",
                )

        return training_log, validation_log
