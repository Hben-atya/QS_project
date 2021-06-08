# add LDDMM shooting code into path
import sys

sys.path.append('../vectormomentum/Code/Python')
sys.path.append('../library')
from subprocess import call
import argparse
import os.path
import gc

# Add deep learning related libraries
from collections import Counter
import torch
from torch.utils.serialization import load_lua
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.tensorboard import SummaryWriter

from QS_network import net
import util
import numpy as np

# Add LDDMM registration related libraries
# library for importing LDDMM formulation configs
import yaml
# others
import logging
import copy
import math

torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)


def loader_and_generator(training_list, validation_list):
    class ImageDataset(TorchDataset):
        def __init__(self, data, transform=(), patch_size=(15, 15, 15)):
            super().__init__()
            self.data = data
            self.transform = transform
            self.patch_size = patch_size

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index: int):
            data = self.data[index]
            # Todo: add loader - nifit??
            #     NiftiLoader = LoadNiftid(["image", "deformation"])
            #     data = NiftiLoader(data)
            generate_patches = _split_patches(self.patch_size)
            data = generate_patches(data)

    class _split_patches(object):
        def __init__(self, patch_size):
            super().__init__()
            self.patch_size = patch_size
        def __call__(self, data, **kwargs):
            image, deformation = data['image'], data['deformation']
            split_images


def train_model(dl_train, dl_valid, max_epochs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = net(feature_num=64, use_dropout=True).to(device)

    # use parallel process if possible:
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        with open(training_info, 'a') as f:
            print("Let's use", torch.cuda.device_count(), "GPUs!", file=f)
        model = nn.DataParallel(model)

    # check if we are working with multi-GPU:
    if isinstance(model, nn.DataParallel):
        parallel_flag = True
    else:
        parallel_flag = False

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-8, amsgrad=True)
    # loss_func = nn.L1Loss(False)  Todo: Ask why sum instead of mean?? (why False)
    loss_func = nn.L1Loss()

    if os.path.exists(os.path.join(training_path, 'checkpoint.pt')):
        checkpoint = torch.load(os.path.join(training_path, 'checkpoint.pt'))
        # check if we are working with multi-GPU:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        epoch_loss_values.append(checkpoint['epoch_loss'])
        best_metric = checkpoint['best_metric']
        metric_values = checkpoint['metric_values']

    else:
        last_epoch = 0
        best_metric = 0
        metric_values = list()

    for epoch in range(last_epoch, max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in dl_train:
            step += 1
            inputs, target = (
                batch_data["image"].to(device),
                batch_data["deformation"].to(device),
            )
            # dist_map = batch_data["dist_map"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, target)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                metric_sum = 0.0
                metric_count = 0
                for val_data in dl_valid:
                    val_inputs, val_deform = (
                        val_data["image"].to(device),
                        val_data["deformation"].to(device),
                    )
                    val_outputs = model(val_inputs)[0]
                    # compute overall mean dice
                    l1_metric = nn.L1Loss()
                    value = l1_metric(val_outputs, val_deform)  # average over the batch size
                    metric_count += 1
                    metric_sum += value.item()

                metric = metric_sum / metric_count
                metric_values.append(metric)
                writer.add_scalar('validation_L1_norm', metric, epoch + 1)

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    if parallel_flag:
                        torch.save(model.module.state_dict(),
                                   os.path.join(training_path, all_params['best_model_name']))

                    else:
                        torch.save(model.state_dict(),
                                   os.path.join(training_path, all_params['best_model_name']))

                    print("saved new best metric model")
                    with open(training_info, 'a') as file:
                        print(
                            f"current epoch: {epoch + 1} current mean abs error : {metric:.4f}",
                            file=file
                        )

                print(
                    f"current epoch: {epoch + 1} current mean abs error : {metric:.4f}",
                    file=file
                )

        if (epoch + 1) % save_freq == 0:
            if parallel_flag:
                torch.save(
                    {'epoch': epoch,
                     'model_state_dict': model.module.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'epoch_loss': epoch_loss,
                     'best_metric': best_metric,
                     'metric_values': metric_values},
                    os.path.join(training_path, 'checkpoint.pt'))

            else:
                torch.save(
                    {'epoch': epoch,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'epoch_loss': epoch_loss,
                     'best_metric': best_metric,
                     'metric_values': metric_values},
                    os.path.join(training_path, 'checkpoint.pt'))
