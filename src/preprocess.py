import tensorflow as tf
from tensorflow import keras
import yaml
import matplotlib.pyplot as plt
import pandas
import numpy
import os
import json




class preprocess_config:
  def __init__(self,params):
    self.base_dir=params["data"]
    self.test=os.path.join(self.base_dir,"valid")
    self.train=os.path.join(self.base_dir,"train")
    self.image_size=params["image_size"]
    self.class_indices_path=params["classindices"]
    self.batch_size=params["batch_size"]
    self.class_mode=params["class_mode"]
    self.rescale=params["rescale"]

  
class preprocess:
  def __init__(self,preprocess_config):
    self.train=preprocess_config.train
    self.test=preprocess_config.test
    self.image_size=preprocess_config.image_size
    self.class_indices_path=preprocess_config.class_indices_path
    self.batch_size=preprocess_config.batch_size
    self.class_mode=preprocess_config.class_mode
    self.rescale=preprocess_config.rescale


  def preprocess_data(self):
    train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.0,
                                                             shear_range = 0.2,
                                                             zoom_range = 0.2,
                                                             width_shift_range = 0.2,
                                                             height_shift_range = 0.2,
                                                             fill_mode="nearest")

    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

    train_data = train_datagen.flow_from_directory(self.train,
                                               target_size = (self.image_size, self.image_size),
                                               batch_size = self.batch_size,
                                               class_mode = self.class_mode)

    test_data = test_datagen.flow_from_directory(self.test,
                                              target_size = (self.image_size, self.image_size),
                                              batch_size = self.batch_size,
                                              class_mode = self.class_mode)
    
    categories = list(train_data.class_indices.keys())
    print(train_data.class_indices)

    class_indices_path=self.class_indices_path

    with open(class_indices_path,'w') as f:
      json.dump(train_data.class_indices, f)

    print(f"The data is preprocessed and the class indices is saved at {class_indices_path}")
    return categories, train_data, test_data


