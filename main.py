import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
from src.preprocess import preprocess_config, preprocess
from src.train import train_config, train
import yaml


prep_params=yaml.safe_load(open("params.yaml"))["preprocess"]
train_params=yaml.safe_load(open("params.yaml"))["train"]

config=preprocess_config(prep_params)
prep=preprocess(config)
categories,train_data,test_data=prep.preprocess_data()

train_cfg=train_config(train_params)
train_data=train(train_cfg,categories,train_data,test_data)
model,validation_accuracy=train_data.train_model()
train_data.save_model(model,train_params["save_dir"],validation_accuracy)