import matplotlib.pyplot as plt
from tsne_torch import TorchTSNE as TSNE
import lightgbm as lgbm
import os
import pandas as pd
import pywt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import itertools as itt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from ful-archi.py import *
from loss.py import *
from lightgbm.py import *
from dataset.py import *


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


# The model

# The following code is the Driver2Vec model setup.


INPUT_LENGTH = 60
CHANNELS_SIZES = [25, 14]
OUTPUT_SIZE = 10
KERNEL_SIZE = 5
DROPOUT = 0.1

LR = 0.0001
BATCH_SIZE = 10
EPOCHS = 10

def get_model(input_channels):
    model = Driver2Vec(input_channels, INPUT_LENGTH, CHANNELS_SIZES, OUTPUT_SIZE,
                       kernel_size=KERNEL_SIZE, dropout=DROPOUT, do_wavelet=True)
    model.to(device)
    return model

def get_accuracy(predictions, labels):
    return (torch.argmax(predictions, -1) == labels).sum().to(device).numpy()/predictions.shape[0]
  
  
if __name__ == "__main__":

    raw_data, raw_labels = loaded_dataset()
    x_train, y_train, x_test, y_text = split_train_test(raw_data, raw_labels)
    training_set = Dataset(x_train, y_train, INPUT_LENGTH)
    test_set = Dataset(x_test, y_text, INPUT_LENGTH)
    training_generator = DataLoader(training_set, 20, 4)


    model = get_model(14)
    loss = HardTripletLoss(device, margin=1)
    optimizer = torch.optim.Adam(
    model.parameters(), lr=LR)
    pbar = tqdm(range(EPOCHS))
    for epoch in pbar:
        loss_list = []
        for anchor, label in training_generator:
            anchor = anchor.to(device)
            optimizer.zero_grad()
            y_anchor = model(anchor)
            loss_value = loss(y_anchor, label)
            loss_value.backward()
            optimizer.step()
            loss_list.append(loss_value.cpu().detach().numpy())
            #print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, np.mean(loss_list)))
            pbar.set_description("Loss: %0.5g, Epochs" % (np.mean(loss_list)))
    #show_TSNE(model, training_set)
    acc = get_n_way_accuracy(2, test_set, training_set, model)
    print(acc)
