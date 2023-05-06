import sys
import os
import numpy as np
import torch

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch.nn as nn
from main import load_config,load_data,model_eval
from data_loader import load_data as data_load
import TMSC_m7G
from model_operation import load_model


config = load_config()  #"../result/m7G_model/config.pkl"
data_iter = data_load('your file path', config)
# data_iter = load_data(config)[0]
print('=' * 20, 'load data over', '=' * 20)
#model

model = TMSC_m7G.TMSC_m7G(config)
path_pretrain_model = "trained model.pt"
# path_pretrain_model = "result/m7G_model/ACC[0.9375], m7G_model.pt"
model = load_model(model, path_pretrain_model)
criterion = nn.CrossEntropyLoss()

last_test_metric, last_test_loss, last_test_repres_list, last_test_label_list, last_test_roc_data, last_test_prc_data = model_eval(data_iter, model, criterion, config)
print('[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
print(last_test_metric.numpy())


