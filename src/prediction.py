import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
import json
from IPython.display import FileLink
from PIL import Image
import io
import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st
import json

class prediction_config:
  def __init__(self,params):
    self.model=params["model"]
    self.class_indices=params["class_indices"]

class predict:
  def __init__(self,prediction_config):
    self.class_indices=prediction_config.class_indices
    self.model=prediction_config.model

  def predict_image(self, image):
        # Check if the image is a Streamlit file-like object, then open it
        if isinstance(image, bytes):  # If image is a byte stream (for example, when uploaded via Streamlit)
            image = Image.open(image)

        # Load the image
        image = image.resize((224, 224))
        image = np.array(image)
        image = image / 255.0  # Normalize image to [0, 1]
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        model_path=self.model
       
        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # Get input details and output details from the model
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Set the input tensor (model expects a specific format)
        input_index = input_details[0]['index']
        interpreter.set_tensor(input_index, image.astype(np.float32))

        # Run inference
        interpreter.invoke()

        # Get the output
        output_index = output_details[0]['index']
        prediction = interpreter.get_tensor(output_index)

        # Get the predicted class
        predicted_class = np.argmax(prediction)

        # Load the class indices from JSON
        with open(self.class_indices, 'r') as f:
            class_indices = json.load(f)

        predicted_disease = ""
        # Map the predicted class to the disease name
        for key, value in class_indices.items():
            if value == predicted_class:
                predicted_disease = key
                break

        # Print and return the predicted disease
        print(f"The disease is {predicted_disease}")
        return predicted_disease
