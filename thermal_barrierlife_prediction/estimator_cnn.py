import tensorflow as tf
import tensorflow.keras.backend as K

from model_cnn import ModelCNN
from estimator import Estimator

    
class EstimatorCNN(Estimator):
        
    def init_model(self):
        """
        Initializes and compiles the model.
        """

        self.model = ModelCNN()
        self._compile_model()
