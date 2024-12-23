import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

class train_config:
  def __init__(self,params):
    self.batch_size=params["batch_size"]
    self.epochs=params["epochs"]
    self.weights=params["weights"]
    self.include_top=params["include_top"]
    self.name=params["name"]
    self.steps_per_epoch=params["steps_per_epoch"]
    self.validation_steps=params["validation_steps"]
    self.image_size=params["image_size"]


class train:
  def __init__(self,train_config,categories,train_data,test_data):
    self.batch_size=train_config.batch_size
    self.epochs=int(train_config.epochs)
    self.weights=train_config.weights
    self.include_top=train_config.include_top
    self.name=train_config.name
    self.steps_per_epoch=train_config.steps_per_epoch
    self.validation_steps=train_config.validation_steps
    self.image_size=train_config.image_size
    self.input_shape=(self.image_size,self.image_size,3)
    self.categories=categories
    self.train_data=train_data
    self.test_data=test_data

  def train_model(self):
    base_model = tf.keras.applications.MobileNet(weights = "imagenet",
                                             include_top = False,
                                             input_shape = self.input_shape)

    base_model.trainable = False
    inputs = keras.Input(shape = self.input_shape)
    x = base_model(inputs, training = False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(len(self.categories),
                              activation="softmax")(x)

    model = keras.Model(inputs = inputs,
                        outputs = x,
                        name="LeafDisease_MobileNet")
    optimizer = tf.keras.optimizers.Adam()

    model.compile(optimizer = optimizer,
                  loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True),
                  metrics=[keras.metrics.CategoricalAccuracy(),
                          'accuracy'])
    history = model.fit(self.train_data,
                    validation_data=self.test_data,
                    epochs=self.epochs,
                    steps_per_epoch=150,
                    validation_steps=100)
    training_accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']
    print(f"Training Accuracy = {training_accuracy}, Validation Accuracy = {validation_accuracy}")
    return model, validation_accuracy

  def save_model(self,model,save_directory,validation_accuracy):
    self.model=model
    model.save(os.path.join(save_directory, f'mobile_net_50{validation_accuracy}.h5'))
    print(f"Model saved as 'mobile_net.h5' in {save_directory} directory.")



