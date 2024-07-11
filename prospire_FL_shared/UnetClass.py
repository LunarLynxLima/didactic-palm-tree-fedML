import timeit

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv1D, GlobalAveragePooling1D, MaxPooling1D, \
    GlobalMaxPooling1D, LeakyReLU, BatchNormalization, Conv2D, MaxPool2D, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.python.platform import gfile
# import tensorflow.contrib.tensorrt as trt
# from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.platform import gfile
from tensorflow.keras import backend as K
# import KerasModelToFrozenGraph as freeze
import numpy as np
import json, itertools, random
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import tensorflow as tf
import math, datetime


class UnetClass:

    def __init__(self, params, img_height, img_width, img_depth, var_loss=False, ):
        self.img_height, self.img_width, self.img_depth = img_height, img_width, img_depth
        self.para = params
        self.min_rss = params.rss_min; self.max_rss = params.rss_max
        self.image_preprocessing = None     # 'normalize' will cause problem with custom loss
        if self.image_preprocessing == 'normalize':
            self.datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
        self.normalize_rss = True   # If not using this need to check that predicted values is within range
        self.rescale = 1.0 / (self.max_rss - self.min_rss)
        self.var_loss = var_loss

        self.use_custom_loss = True
        self.model = self.unet_architecture()
        self.history_file = "unet_history.json"; self.checkpoint_model = 'unet_best_model.h5'
        self.model_file = 'unet_model.h5'
        self.best_batch_size, self.best_epochs = 32, 1000    # model parameters for the nn
        self.worker_epochs = 1000, 10
        # self.best_batch_size, self.best_epochs = 32, 1000    # model parameters for the nn
        self.es = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=1)    # simple early stopping
        self.mc = ModelCheckpoint(self.checkpoint_model, monitor='val_loss', mode='min', verbose=0,
                                      save_best_only=True) # saves the best model
        self.log_dir = "logs/" + "fit/" + 'Unet' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)

    def unet_architecture(self):
        alpha = 0.0  # for leaky relu

        def conv_block(input, num_filters, kernel_size):
            x = Conv2D(num_filters, kernel_size, padding="same", kernel_initializer='he_normal',
                       kernel_regularizer='l2')(input)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=alpha)(x)

            x = Conv2D(num_filters, kernel_size, padding="same", kernel_initializer='he_normal',
                       kernel_regularizer='l2')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=alpha)(x)
            return x

        def encoder_block(input, num_filters, kernel_size):
            # kernel_size = (3, 3)
            x = conv_block(input, num_filters, kernel_size)
            p = MaxPool2D((2, 2))(x)
            # x = BatchNormalization()(x)
            # p = Dropout(rate=0.25)(p)
            return x, p

        def decoder_block(input, skip_features, num_filters, kernel_size):
            x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same", kernel_initializer='he_normal',
                                kernel_regularizer='l2')(input)
            # x = BatchNormalization()(x)
            # x = LeakyReLU(alpha=alpha)(x)
            x = Concatenate()([skip_features, x])
            # kernel_size = (3, 3)
            x = conv_block(x, num_filters, kernel_size)
            return x

        inputs = Input(shape=(self.img_height - 2, self.img_width - 2, self.img_depth)) # for adjustment from 50 to 48

        # Encoder
        s1, p1 = encoder_block(inputs, 32, (3, 3))
        s2, p2 = encoder_block(p1, 64, (3, 3))
        s3, p3 = encoder_block(p2, 128, (3, 3))
        s4, p4 = encoder_block(p3, 256, (3, 3))

        b1 = conv_block(p4, 256, (3, 3))

        d1 = decoder_block(b1, s4, 256, (3, 3))
        d2 = decoder_block(d1, s3, 128, (3, 3))
        d3 = decoder_block(d2, s2, 64, (3, 3))
        d4 = decoder_block(d3, s1, 32, (3, 3))

        outputs = Conv2D(1, (1, 1), padding="same", activation="relu")(d4)
        model = Model(inputs, outputs, name="U-Net")
        adam = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
        if self.use_custom_loss:
            model.compile(loss=self.custom_loss, optimizer='adam')
        else:
            model.compile(optimizer='adam', loss="mean_absolute_error")
            # model.compile(optimizer='adam', loss="mean_squared_error")
        # model.summary()
        return model

    def custom_loss(self, y_true, y_pred):
        img_height = y_true.shape[1]; img_width = y_true.shape[2]

        if self.var_loss:
            lambda_overestimate = 1.0
            lambda_underestimate = 10.0
        else:
            lambda_overestimate = 1.0
            lambda_underestimate = 1.0

        y_true_mask = tf.where(y_true < 0.0, 0.0, 1.0)
        y_true_mask_flattened = K.reshape(y_true_mask, (-1, img_height * img_width))
        y_true_count_nonzero = K.sum(y_true_mask_flattened, axis=-1)

        diff_loss_masked = (y_true - y_pred) * y_true_mask

        underestimate_loss_mask = tf.where(diff_loss_masked > 0.0, 1.0 * lambda_underestimate, 0.0)
        underestimate_loss_mask_flattened = K.reshape(underestimate_loss_mask, (-1, img_height * img_width))
        underestimate_count_nonzero = K.sum(underestimate_loss_mask_flattened, axis=-1)

        overestimate_loss_mask = tf.where(diff_loss_masked < 0.0, 1.0 * lambda_overestimate, 0.0)
        overestimate_loss_mask_flattened = K.reshape(overestimate_loss_mask, (-1, img_height * img_width))
        overestimate_count_nonzero = K.sum(overestimate_loss_mask_flattened, axis=-1)

        diff_loss_masked = tf.where(diff_loss_masked > 0.0, diff_loss_masked * lambda_underestimate, diff_loss_masked)
        diff_loss_masked = tf.where(diff_loss_masked < 0.0, diff_loss_masked * lambda_overestimate, diff_loss_masked)
        abs_loss_masked = tf.abs(diff_loss_masked)
        loss_masked = abs_loss_masked

        loss_masked = K.reshape(loss_masked, (-1, img_height * img_width))
        sum_loss_masked = K.sum(loss_masked, axis=-1)
        mean_loss_masked = sum_loss_masked / (underestimate_count_nonzero + overestimate_count_nonzero)
        # mean_loss_masked = sum_loss_masked / y_true_count_nonzero

        return mean_loss_masked

    def training(self, train_x, train_y, start_training, worker_epochs = -1):
        if not start_training:
            self.model = load_model(self.model_file, custom_objects={'custom_loss': self.custom_loss})
            return

        if self.normalize_rss:
            if self.use_custom_loss:
                train_y = np.where(train_y == 0.0, self.min_rss - 10.0, train_y)
            else:
                train_y = np.where(train_y == 0.0, self.min_rss, train_y)
            train_y = train_y - self.min_rss
            train_y = train_y * self.rescale
        # print(train_x.shape, train_y.shape)
        train_x = np.transpose(train_x, [0, 2, 3, 1])   # for channel last orientation
        train_y = train_y.reshape(tuple(list(train_y.shape) + [1]))
        # print(train_x.shape, train_y.shape)
        # remove the outermost rows and columns for easier integration with UNET structure
        train_x = train_x[:, 1:-1, 1:-1, :]; train_y = train_y[:, 1:-1, 1:-1, :]
        print('Train and test shape for UNET', train_x.shape, train_y.shape)
        print('Train and test set data size', train_x.nbytes, train_y.nbytes)

        if worker_epochs != -1:
            self.best_epochs = worker_epochs

        if self.image_preprocessing == 'normalize':
            train_idxs = np.random.choice(np.arange(train_x.shape[0]), size=int(train_x.shape[0] * 0.8), replace=False)
            val_idxs = list(set(np.arange(train_x.shape[0])) - set(train_idxs))

            x_val, y_val = train_x[val_idxs, ...], train_y[val_idxs, ...]
            train_x, train_y = train_x[train_idxs, ...], train_y[train_idxs, ...]
            print('Train_x and train_y shape for UNET', train_x.shape, train_y.shape)
            print('val_x and val_y shape for UNET', x_val.shape, y_val.shape)

            self.datagen.fit(train_x)

            history = self.model.fit(self.datagen.flow(train_x, train_y, batch_size=self.best_batch_size),
                                     steps_per_epoch=math.ceil(len(train_x) / float(self.best_batch_size)),
                                     epochs=self.best_epochs,
                                     callbacks=[self.es, self.mc],
                                     validation_data=self.datagen.flow(x_val, y_val, batch_size=self.best_batch_size),
                                     validation_steps=math.ceil(len(x_val) / float(self.best_batch_size)),
                                     verbose=0)

        else:
            history = self.model.fit(train_x, train_y,
                                     validation_split=0.2,
                                     batch_size=self.best_batch_size,
                                     epochs=self.best_epochs,
                                     callbacks=[self.es, self.mc, self.tensorboard_callback],
                                     verbose=1,
                                     shuffle=True)

        if self.use_custom_loss:
            self.model = load_model(self.checkpoint_model, custom_objects={'custom_loss': self.custom_loss})
        else:
            self.model.load_weights(self.checkpoint_model)  # load the saved best model
        self.model.save(self.model_file)        # Save the model for future use
        with open(self.history_file, "w") as json_file:     # save the history in a file
            json.dump(history.history, json_file)
        print('best_val_loss, best_train_loss', self.get_train_stats())

    # Returns the validation loss and train loss for the chosen model
    def get_train_stats(self):
        with open(self.history_file, "r") as json_file:
            history = json.load(json_file)
        best_val_loss = min(history['val_loss'])
        i, = np.where(np.array(history['val_loss']) == best_val_loss)
        best_idx = i[0]
        best_train_loss = history['loss'][best_idx]
        return best_val_loss, best_train_loss

    def predict_rss(self, test_x, batch=False):
        if len(test_x.shape) == 3:
            test_x = test_x.reshape(tuple([1] + list(test_x.shape)))
        test_x = np.transpose(test_x, [0, 2, 3, 1])     # for making channels last
        test_x = test_x[:, 1:-1, 1:-1, :]       # for making images 48x48

        if self.image_preprocessing == 'normalize':
            test_iterator = self.datagen.flow(test_x, None, batch_size=1)
            test_x = test_iterator.next()

        tx = timeit.default_timer()
        if batch:
            predicted_values = self.model.predict(test_x, batch_size=1024, verbose=0)
        else:
            # predicted_values = self.model.predict(test_x, verbose=0)
            predicted_values = self.model(test_x, training=False)
        # print(timeit.default_timer() - tx)

        if self.normalize_rss:
            rss_values = predicted_values / self.rescale
            rss_values = rss_values + self.min_rss
        else:
            rss_values = predicted_values
        rss_values = np.where(test_x[..., 1] == 0.0, 0.0, rss_values[..., 0])  # use predictions only for active RX locs
        # make the images back to 50x50
        temp_res = np.zeros((rss_values.shape[0], self.img_height, self.img_width))
        temp_res[:, 1:-1, 1:-1] = rss_values
        rss_values = temp_res
        if not batch:
            return rss_values[0, ...]
        else:
            return rss_values