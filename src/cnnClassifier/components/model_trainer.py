import os
import tensorflow as tf
from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint  # Import ModelCheckpoint


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            # normalize pixel values to [0,1]
            rescale=1./255,
            # validation data percentage
            validation_split=0.20
        )
        # sets params related to the data shape
        dataflow_kwargs = dict(
            # dimensions of each image, excluding the number of channels.
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )
        # Creates a generator for validation data.
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        # Reads images directly from a directory and organizes them based on subdirectories.
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )
        # Configures the training data generator with optional data augmentation.
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                # Rotates the images randomly within 40 degrees.
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
        
        # Define the checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            'artifacts/training/model.h5',  # Path where the model is saved
            save_best_only=True,  # Save only the best model based on validation accuracy
            save_weights_only=True,  # Save weights only (model architecture is assumed to be the same)
            verbose=1
        )

        # Train the model with the checkpoint callback
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=[checkpoint_callback]  # Pass the checkpoint callback here
        )

        # Save the final model after training (in case you want to keep the final model too)
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )