import numpy as np

from thermal_barrierlife_prediction import EstimatorCNN
from thermal_barrierlife_prediction.evaluation import performance_report


class EnsembleEstimator:

    def __init__(self):
        self.estimators = {}

    def create_models(
            self,
            estimator_args,
            val_sets,
    ):
        for val_set in val_sets:
            for estimator_name, args in estimator_args:
                if 'CNN' in estimator_name:
                    estim = EstimatorCNN()
                else:
                    raise ValueError('Estimator not recognized!')

                estim.prepare_data(**args['data'])
                estim.init_model(**args['init'])
                estim.train(
                    val_samples=val_set,
                    **args['train']
                )
                if estimator_name not in self.estimators.keys():
                    self.estimators[estimator_name] = []
                self.estimators[estimator_name].append(estim)

    def evaluate_models(
            self,
    ):
        for model_type in self.estimators.keys():
            for estim in self.estimators[model_type]:
                y_pred = estim.predict(val_idx=estim.val_idx)  # Predicts with saved val data
                y_true = estim.data['lifetime'][estim.val_idx]
                y_max = estim.data['magnification'][estim.val_idx]
                estim.performance_report = performance_report
