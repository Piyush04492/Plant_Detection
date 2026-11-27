"""
Convolutional Neural Network (CNN) architecture for Plant Disease Classification.
Demonstrates:
- Feature extraction with 2D Convolution layers
- Spatial downsampling with Max Pooling
- Regularization with Dropout to combat overfitting
- Multiclass classification with Softmax activation
"""

import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers, models


def build_plant_cnn(input_shape=(128, 128, 3), num_classes=6):
    """
    Build a custom deep Convolutional Neural Network for plant disease classification.
    
    Architecture:
    -------------
    Input Layer: (128, 128, 3)
    Block 1:
      - Conv2D (32 filters, 3x3 kernel, ReLU, same padding) -> Extracts edge & texture features
      - Conv2D (32 filters, 3x3 kernel, ReLU)
      - MaxPooling2D (2x2 pool) -> Downsamples spatial dimension to 63x63
      - Dropout (0.25) -> Reduces co-adaptation of features to prevent overfitting
    Block 2:
      - Conv2D (64 filters, 3x3 kernel, ReLU, same padding) -> Extracts vein and lesion shapes
      - Conv2D (64 filters, 3x3 kernel, ReLU)
      - MaxPooling2D (2x2 pool) -> Downsamples to 30x30
      - Dropout (0.25)
    Block 3:
      - Conv2D (128 filters, 3x3 kernel, ReLU, same padding) -> Extracts disease spots & complex patterns
      - MaxPooling2D (2x2 pool) -> Downsamples to 15x15
      - Dropout (0.30)
    Classifier Head:
      - Flatten -> Flattens 2D feature maps to 1D feature vector
      - Dense (128 units, ReLU) -> High-level representation
      - Dropout (0.50) -> Strong regularization
      - Dense (num_classes, Softmax) -> Probability distribution over disease classes
    """
    model = models.Sequential([
        layers.Input(shape=input_shape, name="leaf_image_input"),
        
        # --- Block 1 ---
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_1"),
        layers.Conv2D(32, (3, 3), activation="relu", name="conv1_2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),
        layers.Dropout(0.25, name="dropout1"),
        
        # --- Block 2 ---
        layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_1"),
        layers.Conv2D(64, (3, 3), activation="relu", name="conv2_2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),
        layers.Dropout(0.25, name="dropout2"),
        
        # --- Block 3 ---
        layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool3"),
        layers.Dropout(0.30, name="dropout3"),
        
        # --- Classification Head ---
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense_features"),
        layers.Dropout(0.50, name="dropout_dense"),
        layers.Dense(num_classes, activation="softmax", name="disease_probabilities")
    ], name="PlantVillage_CNN")
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile model with Adam optimizer and Sparse Categorical Crossentropy.
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
