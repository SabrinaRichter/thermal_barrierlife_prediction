from thermal_barrierlife_prediction import EstimatorCNN

class EnsembleEstimator:

    def __init__(self):
        self.estimators = {}

    def create_models(
            self,
            estimator_args,
            val_sets,
    ):
        for val_set in val_sets:
            for estimator_name, args in estimator_args.items():
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
