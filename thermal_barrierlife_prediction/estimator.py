import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Union
import warnings


class Estimator:
    """
    Estimator class. Contains all necessary methods for data loading,
    model initialization, training, evaluation and prediction.
    """

    def __init__(self):
        self.model = None  # TODO what is this

    def get_data(
            self,
            test_split=0.1,
            validation_split=0.1,
            train_split=None,
            seed: int = 1,
    ):
        """
        Prepares the necessary data input for the model.
        """
        pass

    def train(
            self,
    ):
        """
        Train model.
        """
        train_dataset = ...
        eval_dataset = ...

        self.history = self.model.training_model.fit(
            x=train_dataset,
            validation_data=eval_dataset,
            verbose=2
        ).history

    def _compile_model(
            self,
            optimizer,
    ):
        """
        Prepares the losses and metrics and compiles the model.
        """
        self.model.training_model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            loss_weights=loss_weights
        )

    def evaluate(
            self,
            keys=None,
    ):
       pass

    def predict(
            self,
            keys=None,
    ):
      pass
