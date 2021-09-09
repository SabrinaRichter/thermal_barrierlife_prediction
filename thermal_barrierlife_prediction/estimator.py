import tensorflow as tf

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
        self.train_data = self.data.sel(image_id=self.data.image_id[[el not in ['M-19-271'] for el in self.data.sample]])

        X_train = self.train_data.greyscale.values
        y_train = self.train_data.lifetime.values

        if len(val_samples) > 0:
            self.val_data = self.data.sel(image_id=self.data.image_id[[el in ['M-19-271'] for el in self.data.sample]])
            validation_data = (self.val_data.greyscale.values, self.val_data.lifetime.values)
        else:
            self.val_data = None
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

    def compute_gradients_input(
            self,
            image_id,
            plot=True,
    ):
        """
         Compute gradients with respect to input data. (A list of gradients per sample if target is multidimensional.)
        """
        input_raw = self.data.sel(image_id=image_id).greyscale.values
        input_X = tf.convert_to_tensor(input_raw.astype('float32'))
        input_X = tf.expand_dims(input_X, 0)
        with tf.GradientTape(persistent=True) as g:
            g.watch(input_X)
            prediction = self.model.training_model(input_X)
            gradients = g.gradient(prediction, input_X).numpy()[0]
        if plot:
            import matplotlib.pyplot as plt
            plt.matshow(gradients, cmap='gray')
            plt.matshow(input_raw, cmap='gray')
        return prediction.numpy()[0, 0], gradients
