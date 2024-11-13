import os
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig



class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            # Boolean flag to include or exclude the top (fully connected) layers.
            include_top=self.config.params_include_top
        )
        self.save_model(path=self.config.base_model_path, model=self.model)
    # static method means it's class dependent not instance dependent
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        # If True, sets all layers in the base model to be non-trainable (they won’t update during training).
        if freeze_all:
            for _ in model.layers:
                model.trainable = False
        # Specifies how many layers at the end of the model should remain trainable (useful for fine-tuning).
        elif (freeze_till is not None) and (freeze_till > 0):
            for _ in model.layers[:-freeze_till]:
                model.trainable = False
        # The Flatten layer converts the model’s output to a 1D array, making it suitable for a fully connected layer.
        flatten_in = tf.keras.layers.Flatten()(model.output)
        # Dense Layer (Softmax): A new Dense layer with softmax activation is added, with units equal to the number of classes (from params_classes), enabling multi-class classification.
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)
        # Model Creation: A new Model is created with the same input as the base model but a new output layer (softmax for classification).
        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            # For multi-class classification
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        full_model.summary()
        return full_model
    # This method calls _prepare_full_model with the base model and the configuration parameters, then saves the updated model.
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)