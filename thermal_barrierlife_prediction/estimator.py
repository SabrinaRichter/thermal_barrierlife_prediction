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
            tiff_folder_path='../data/train/',  # for validation: '../data/valid'
            mixup: bool = False,
            for_training=True,
            data_ratio_mixup=2,
            alpha_mixup=0.2,
            # if True, standardization parameters will be computed, if False then apply parameters computed from training data
    ):
        """
        Prepares the necessary data input for the model.
        """
        self.data = read_data(
            csv_file_path=csv_file_path,
            tiff_folder_path=tiff_folder_path,
        )
        # Mixup
        if mixup:
            self.mixup_data(data_ratio_produce=data_ratio_mixup, alpha=alpha_mixup)

        ### augment, do whatever you want (distinguish between train and validation setting!)

    def train(
            self,
            val_samples=[],
            batch_size=8,
            epochs=20
    ):
        """
        Trains the model.
        """
        self.train_idx = np.argwhere([sample not in val_samples for sample in self.data['sample']]).ravel()
        self.val_idx = np.argwhere([sample in val_samples for sample in self.data['sample']]).ravel() \
            if len(val_samples) > 0 else None

        X_train = self.data['greyscale'][self.train_idx]
        y_train = self.data['lifetime'][self.train_idx]

        if len(val_samples) > 0:
            validation_data = (self.data['greyscale'][self.val_idx], self.data['lifetime'][self.val_idx])
        else:
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

    def mixup_data(self, data_ratio_produce=2, alpha=0.2):
        """
        Make mixup data, add to original data, and save in estimator's data object
        :param data_ratio_produce: Prudouce int(X_train.shape[0]*data_ratio_produce) samples
        :param alpha: Beta distn param for sampling mixing weights
        """
        real_samples_idx = np.argwhere(self.data['real']).ravel()
        n_training_samples = real_samples_idx.shape[0]
        # Make random mixup samples
        n_samples = int(n_training_samples * data_ratio_produce)
        data_new = dict()
        for key in self.data:
            data_new[key] = []
        for i in range(n_samples):
            # Mixup ratio
            lam = np.random.beta(alpha, alpha)
            # Should not happen, but just in case to detect bugs
            if lam < 0 or lam > 1:
                raise ValueError('Lam not between 0 and 1')
            # Images to choose for mixup, choose only from real samples
            idxs = np.random.choice(real_samples_idx, 2, replace=False)
            idx0 = idxs[0]
            idx1 = idxs[1]

            # Make mixup data
            data_new['greyscale'].append(
                self.data['greyscale'][idx0] * lam + self.data['greyscale'][idx1] * (1 - lam))
            data_new['sample'].append(
                '_'.join([str(self.data['sample'][idx0]), str(lam), str(str(self.data['sample'][idx1])), str(1 - lam)]))
            data_new['lifetime'].append(
                self.data['lifetime'][idx0] * lam + self.data['lifetime'][idx1] * (1 - lam))
            data_new['magnification'].append(
                self.data['magnification'][idx0] * lam + self.data['magnification'][idx1] * (1 - lam))
            data_new['uncertainty'].append(
                self.data['uncertainty'][idx0] * lam + self.data['uncertainty'][idx1] * (1 - lam))
            data_new['image_id'].append(
                '_'.join(
                    [str(self.data['image_id'][idx0]), str(lam), str(self.data['image_id'][idx1]), str(1 - lam)]))
            data_new['real'].append(0)

        # Add mixup to data
        for key in self.data.keys():
            if len(data_new[key]) != n_samples:
                raise ValueError('Mixup data for %s not of corect length' % key)
            # Do not use np concat as it is slow - filling an array is quicker
            # data_temp = np.empty((self.data[key].shape[0] + len(data_new[key]), *self.data[key].shape[1:]),
            #                      dtype=self.data[key].dtype)
            # for i in range(self.data[key].shape[0]):
            #     data_temp[i] = self.data[key][i]
            # # Add new data after old one (array positions starting after positions of original data)
            # for i in range(len(data_new[key])):
            #     data_temp[i+self.data[key].shape[0]] = data_new[key][i]
            # self.data[key] = data_temp
            self.data[key] = np.concatenate([self.data[key], data_new[key]])

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
            input_raw = self.data['greyscale'][np.argwhere(self.data['image_id'] == image_id)]
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
