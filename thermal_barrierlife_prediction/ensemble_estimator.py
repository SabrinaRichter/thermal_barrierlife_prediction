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
                if estimator_name == 'CNN':
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
                val_samples = np.unique(estim.val_id)
                y_pred = estim.predict(val_samples)
                y_true = estim.data.sel(image_id=estim.data.image_id[[el in val_samples for el in estim.data.sample]]).lifetime.values
                y_mag = estim.data.sel(image_id=estim.data.image_id[[el in val_samples for el in estim.data.sample]]).magnification.values
                estim.performance_report = performance_report