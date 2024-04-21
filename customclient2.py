import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import flwr as fl
from sklearn.model_selection import train_test_split

# Define U-Net model
def unet_model(input_shape):
    # Define the model architecture here
    # Example:
    inputs = tf.keras.layers.Input(shape=input_shape)
    # Encoder, decoder, output layers...
    # Define your model architecture here
    return model

# Load your data
X = np.load('X2.npy')
Y = np.load('Y2.npy')

# Check the shape of Y
print("Shape of Y before resizing:", Y.shape)

# Ensure X and Y have 4 dimensions
if len(X.shape) == 3:
    X = np.expand_dims(X, axis=-1)
if len(Y.shape) == 3:
    Y = np.expand_dims(Y, axis=-1)

# Check the shape of Y after adjustment
print("Shape of Y after adjustment:", Y.shape)

# Resize images to (224, 224, 3)
X_resized = tf.image.resize(X, (224, 224))
Y_resized = tf.image.resize(Y, (224, 224))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resized, Y_resized, test_size=0.2, random_state=42)

# Further split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Data augmentation for the training set
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data augmentation for the validation and testing sets (only rescaling)
val_test_datagen = ImageDataGenerator()

# Train data generator
train_generator = train_datagen.flow(
    X_train,
    y_train,
    batch_size=32
)

# Validation data generator
val_generator = val_test_datagen.flow(
    X_val,
    y_val,
    batch_size=32
)

# Test data generator
test_generator = val_test_datagen.flow(
    X_test,
    y_test,
    batch_size=32,
    shuffle=False
)

# Define U-Net model with input shape (224, 224, 3)
unet_model = unet_model(input_shape=(224, 224, 3))

# Compile the model
unet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return unet_model.get_weights()

    def fit(self, parameters, config):
        unet_model.set_weights(parameters)
        for i in range(5):
            unet_model.fit(
                train_generator,
                steps_per_epoch=len(X_train) // 32,
                validation_data=val_generator,
                validation_steps=len(X_val) // 32
            )
        return unet_model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        unet_model.set_weights(parameters)
        test_loss, test_accuracy = unet_model.evaluate(test_generator, steps=len(X_test) // 32)
        return test_loss, len(X_test), {"accuracy": test_accuracy}

fl.client.start_client(server_address="127.0.0.1:5000", client=FlowerClient().to_client())
