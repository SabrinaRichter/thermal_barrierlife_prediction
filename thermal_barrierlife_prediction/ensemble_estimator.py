import numpy as np

from thermal_barrierlife_prediction import EstimatorCNN

from thermal_barrierlife_prediction.evaluation import performance_report, performance_report_magnification
from thermal_barrierlife_prediction.paralelise_utils import parallelize


class Params:
    def __init__(self, val_set, estimator_name, args):
        self.val_set = val_set
        self.estimator_name = estimator_name
        self.args = args



class EnsembleEstimator:

    def __init__(self):
        self.estimators = {}

    def create_models(
            self,
            estimator_args,
            val_sets,
            n_jobs=-1,
    ):
        """
        :param n_jobs:Use at most n_jobs. If below 1 use as many jobs as param combinations.
        """
        params = []
        # TODO self.estimators
        for val_set in val_sets:
            for estimator_name, args in estimator_args:
                params.append(Params(val_set=val_set, estimator_name=estimator_name, args=args))
        paralelize = True  # Set to False only for testing out if the helper function even works in non-parallel mode
        if not paralelize:
            self.res = []
            self.res.append(self._estimator_setup_train(params=np.array(params), queue=None))
        else:
            self.res = parallelize(
                self._estimator_setup_train,
                collection=params,
                extractor=self._extract,
                use_ixs=False,
                n_jobs=min([len(params),n_jobs]) if n_jobs > 0 else len(params),
                # multiprocessing does not work, loky not ok as tries to pickle
                backend="threading",
                show_progress_bar=False,
            )()


    def _extract(self, res):
        return [i for r in res for i in r]

    def _estimator_setup_train(self, params, queue=None):
        """
        Loop through array or Param instances to create and train Estimators
        :return: List or results for each param combination
        """
        res = []
        for param in params:
            estimator_name = param.estimator_name
            args = param.args
            val_set = param.val_set

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
            res.append({'estim': estim, 'param': param})
        return res

    def evaluate_models(
            self,
    ):
        for res in self.res:
            estim=res['estim']
            y_pred = estim.predict(val_idx=estim.val_idx)  # Predicts with saved val data
            y_true = estim.data['lifetime'][estim.val_idx]
            y_max = estim.data['magnification'][estim.val_idx]
            estim.performance_report = performance_report(y_true, y_pred)
            estim.performance_report_magnification = performance_report_magnification(y_true, y_pred, y_max)
