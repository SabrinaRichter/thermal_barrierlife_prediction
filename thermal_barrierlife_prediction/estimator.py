import tensorflow as tf
import numpy as np

from thermal_barrierlife_prediction.load_data import read_data

class Estimator:
    """
    Estimator class. Contains all necessary methods for data loading,
    model initialization, training, evaluation and prediction.
    """

    def prepare_data(
            self,
            csv_file_path='../data/train-orig.csv',
            tiff_folder_path='../data/train/', # for validation: '../data/valid'
            for_training=True, # if True, standardization parameters will be computed, if False then apply parameters computed from training data
    ):
        """
        Prepares the necessary data input for the model.
        """
        self.data = read_data(
            csv_file_path=csv_file_path,
            tiff_folder_path=tiff_folder_path,
        )

        ### augment, do whatever you want (distinguish between train and validation setting!)

    def train(
            self,
            val_samples=[],
            batch_size=8,
            epochs=20,
    ):
        """
        Trains the model.
        """
        self.val_id = val_samples
        train_data = self.data.sel(image_id=self.data.image_id[[el not in val_samples for el in self.data.sample]])

        X_train = train_data.greyscale.values
        y_train = train_data.lifetime.values

        if len(val_samples) > 0:
            val_data = self.data.sel(image_id=self.data.image_id[[el in val_samples for el in self.data.sample]])
            validation_data = (val_data.greyscale.values, val_data.lifetime.values)
        else:
            val_data = None
            validation_data = None

        self.history = self.model.training_model.fit(
            x=X_train,
            y=y_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=2,
        ).history

    def _compile_model(
            self,
    ):
        """
        Prepares the losses and metrics and compiles the model.
        """
        self.model.training_model.compile(
            loss=tf.keras.losses.mean_squared_error,
            optimizer='adam',
            metrics=[
                tf.keras.metrics.mean_squared_error,
                tf.keras.metrics.mean_absolute_error
            ],
        )
        
    def predict(self,
                val_samples,
    ):
        '''
        predicts a set of input samples
        '''
        if len(val_samples) > 0:
            val_data = self.data.sel(image_id=self.data.image_id[[el in val_samples for el in self.data.sample]])
        y_pred = self.model.training_model.predict(val_data.greyscale.values)
        return y_pred
        
    def compute_gradients_input(
            self,
            image_ids,
            plot=True,
    ):
        """
         Computes and plots gradients with respect to input data.
        """
        if not isinstance(image_ids, list):
            image_ids = [image_ids]
        predictions = []
        gradients = []
        for image_id in image_ids:
            input_raw = self.data.sel(image_id=image_id).greyscale.values
            input_X = tf.convert_to_tensor(input_raw.astype('float32'))
            input_X = tf.expand_dims(input_X, 0)
            with tf.GradientTape(persistent=True) as g:
                g.watch(input_X)
                pred = self.model.training_model(input_X)
                grad = g.gradient(pred, input_X).numpy()[0]
            predictions.append(pred.numpy()[0, 0])
            gradients.append(grad)
            if plot:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].matshow(np.abs(grad), cmap='gray', vmax=np.sort(np.abs(grad).flatten())[-10000])
                ax[1].matshow(input_raw, cmap='gray', vmin=0, vmax=255)
                ax[0].axis('off')
                ax[1].axis('off')
                plt.tight_layout()
        return predictions, gradients
