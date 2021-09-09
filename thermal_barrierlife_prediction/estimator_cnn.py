from thermal_barrierlife_prediction.model_cnn import ModelCNN
from thermal_barrierlife_prediction.estimator import Estimator

    
class EstimatorCNN(Estimator):
        
    def init_model(self):
        """
        Initializes and compiles the model.
        """

        input_shape = self.data.greyscale.shape[1:]
        self.model = ModelCNN(
            input_shape=input_shape,
        )
        self._compile_model()
