import numpy as np
import sklearn.metrics

def performance_report(y_true, y_pred):
    performance_dict = {'mean_squared_error':sklearn.metrics.mean_squared_error(y_true, y_pred),
                        'mean_absolute_error':sklearn.metrics.mean_absolute_error(y_true, y_pred),
                        'mean_absolute_percentage_error':sklearn.metrics.mean_absolute_percentage_error(y_true, y_pred),                        
                       }
    return performance_dict

def performance_report_magnitude(y_true, y_pred, magnitude):
    try:
        y_true_500 = y_true[magnitude==500]
        y_pred_500 = y_pred[magnitude==500]
        y_true_2000 = y_true[magnitude==2000]
        y_pred_2000 = y_pred[magnitude==2000]
        performance_dict = {'mse':sklearn.metrics.mean_squared_error(y_true, y_pred),
                            'mse_500':sklearn.metrics.mean_squared_error(y_true_500, y_pred_500),
                            'mse_2000':sklearn.metrics.mean_squared_error(y_true_2000, y_pred_2000),
                            'mae':sklearn.metrics.mean_absolute_error(y_true, y_pred),
                            'mae_500':sklearn.metrics.mean_absolute_error(y_true_500, y_pred_500),
                            'mae_2000':sklearn.metrics.mean_absolute_error(y_true_2000, y_pred_2000),
                            'mape':sklearn.metrics.mean_absolute_percentage_error(y_true, y_pred), 
                            'mape_500':sklearn.metrics.mean_absolute_percentage_error(y_true_500, y_pred_500), 
                            'mape_2000':sklearn.metrics.mean_absolute_percentage_error(y_true_2000, y_pred_2000), 
                           }
        return performance_dict
    except:
        print("Oops! No samples in one of the magnitude classes :( try the regular performance report")