import numpy as np
from tensorflow.keras.applications import MobileNetV2

class FeatureExtractor:
    def __init__(self, input_shape=(224, 224, 3)):
        self.base_model = self.setup_model(input_shape)

    def setup_model(self, input_shape):
        base_model = MobileNetV2(input_shape=input_shape, weights='imagenet', include_top=False, pooling='avg')
        base_model.trainable = False
        return base_model

    def get_features(self, image_array):
        # Ensure the input image is a 224x224x3 NumPy array
        if image_array.shape != (224, 224, 3):
            raise ValueError(f"Input image shape {image_array.shape} does not match expected shape (224, 224, 3)")
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        features = self.base_model.predict(image_array).flatten()  # Extract features
        return features