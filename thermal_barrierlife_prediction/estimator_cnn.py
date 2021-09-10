from thermal_barrierlife_prediction.model_cnn import ModelCNN
from thermal_barrierlife_prediction.estimator import Estimator

    
class EstimatorCNN(Estimator):
        
    def init_model(self, **model_args):
        """
        Initializes and compiles the model.
        """

        input_shape = self.data['greyscale'].shape[1:]
        self.model = ModelCNN(
            input_shape=input_shape,
            **model_args
        )
        self._compile_model()
