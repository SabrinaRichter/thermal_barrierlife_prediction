import numpy as np
from thermal_barrierlife_prediction.load_data import read_data
from thermal_barrierlife_prediction import EstimatorCNN
from thermal_barrierlife_prediction.evaluation import performance_report, performance_report_magnification            
import os


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
            runs = 15,
    ):
        for model_type in self.estimators.keys():
            for estim in self.estimators[model_type]:
                y_pred = []
                for i in range(runs):
                    y_pred.append(estim.predict(val_idx=estim.val_idx))  # Predicts with saved val data
                y_pred = np.mean(np.array(y_pred), axis=0)
                y_true = estim.data['lifetime'][estim.val_idx]
                y_max = estim.data['magnification'][estim.val_idx]
                estim.performance_report = performance_report(y_true, y_pred)
                estim.performance_report_magnification = performance_report_magnification(y_true, y_pred, y_max)



    def predict_val(
        self,
        runs = 15,
    ):
        y_model_avg = []
        model_scores = []
        for model_type in self.estimators.keys():
            for estim in self.estimators[model_type]:
                val_data = read_data(csv_file_path=False, tiff_folder_path='../data/valid/')['greyscale']
                y_pred = []
                for i in range(runs):
                    y_pred.append(estim.predict(val_data=val_data))  # Predicts with saved val data
                y_model_avg.append(np.mean(np.array(y_pred), axis=0))
                model_scores.append(1/estim.performance_report['mean_absolute_error'])
        model_weights = np.array(model_scores)/np.sum(model_scores)
        y_pred = y_model_avg*model_weights
        print(y_pred.shape)
        y_pred = np.mean(y_pred, axis=0)
        return y_pred