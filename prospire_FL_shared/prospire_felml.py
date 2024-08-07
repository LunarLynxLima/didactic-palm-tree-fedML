#! /home/shamik/.local/lib/python3.8/site-packages

# nvidia-smi
# source fedml/bin/activate ## to get into environment
import tensorflow as tf
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
from tensorflow.keras import backend as K
import json, itertools, random
import math, datetime
import gc
from tensorflow.keras.backend import clear_session
import numpy as np
import timeit, time, os
import h5py
from matplotlib import pyplot as plt
from Parameters import Parameters
from UnetClass import UnetClass
import shutil

import warnings
warnings.filterwarnings("ignore")

class FederatedWorker:
    def __init__(self, worker_id, nunet_obj, train_x_images, train_y_images, params, worker_epochs:int=100, total_workers:int=4):
        self.worker_id = worker_id
        self.nunet_obj = nunet_obj
        self.raw_train_x_images = train_x_images
        self.raw_train_y_images = train_y_images
        x_mask, y_mask = FederatedWorker.gen_mask(train_x_images, prob = 1/total_workers)
        self.train_x_images = train_x_images * x_mask
        self.train_y_images = train_y_images * y_mask
        self.worker_epochs = worker_epochs
        self.params = params

    def gen_mask(tensor, prob=None, seed=42, dtype=tf.float32,):
        """
        Generates a batched masks.

        Mask is all ones for idx 0 and 2, as per 0 is TX and 2 is location map.
        For idx 1 and 3, it is 1 for sampled points with probability of 1/n or prob as prompted.

        Parameters:
        - tensor: The input tensor, expected to be 4D.
        - prob: The probability for the mask sampling.
        - dtype: The desired output tensor dtype.
        - seed: Seed for random number generation. {For reproducibility :: np.random.seed(seed)} can't get seed different for different workers
        """

        if prob is None: return -1
        mask = np.ones(tensor.shape)

        y_mask = np.ones((tensor.shape[0], tensor.shape[2], tensor.shape[3]))
        if len(tensor.shape) == 4:
            for i in range(tensor.shape[0]):
                mask[i, 1] = np.random.choice([0, 1], size=tensor.shape[2:], p=[1-prob, prob])
                mask[i, 3] = mask[i, 1]
                y_mask[i] = mask[i, 1]
                # mask[i, 3] = np.random.choice([0, 1], size=tensor.shape[2:], p=[1-prob, prob])

        else: raise ValueError("Tensor must be a 4D batch of 3D tensors.")
        return tf.convert_to_tensor(mask, dtype=dtype), tf.convert_to_tensor(y_mask, dtype=dtype)

    def train_local_model(self, accelerate = True):
        """
        Train local model,
        Returns val_loss, and train_loss, training time
        """
        start_time = time.time()
        self.nunet_obj.training(self.train_x_images, self.train_y_images, start_training=True, worker_epochs=self.worker_epochs, accelerate = accelerate)
        training_time = (time.time() - start_time)
        val_loss, train_loss = self.nunet_obj.get_train_stats() 
        return val_loss, train_loss, training_time

    def federated_averaging(global_obj, worker_objs):
        weights = [1/len(worker_objs)]*len(worker_objs)
        avg_weights = [np.zeros_like(w) for w in global_obj.model.get_weights()]

        for i, worker_obj in enumerate(worker_objs):
            local_weights = worker_obj.nunet_obj.model.get_weights()
            for j in range(len(avg_weights)): avg_weights[j] += local_weights[j] * weights[i]

        for i, worker_obj in enumerate(worker_objs): worker_objs[i].nunet_obj.model.set_weights(avg_weights)

        global_obj.model.set_weights(avg_weights)

        # for j in range(len(avg_weights)):
        #     if np.array_equal(global_obj.model.get_weights()[j], avg_weights[j]):
        #         continue
        #     else:
        #         print('weight mismatch')

        return global_obj.model.get_weights(), avg_weights

    def visualize_divided_data(self, datapoints=1, workers=[], colorbar = False, max_workers_visualized = 5, full_plot = True):
        features = ['X1 = primary transmitter', 'X2 = Receiver (RSS measurements for X1)', 'X3 = Area Map', 'X4 = Shadow fading on links b/w X1']
        features = ['X1', 'X2','X3', 'X4']
        a = (1,4,2)
        if full_plot: a = (0,4,1)
        for i in range(datapoints):
            for j in range(a[0], a[1], a[2]):
                tensor1 = self.raw_train_x_images[i][j]
                tensor0 = self.raw_train_y_images[i]
                max_workers_visualized = min(len(workers), max_workers_visualized)
                tensors = [worker.train_x_images[i][j] for worker in workers][:max_workers_visualized]
                ytensors = [worker.train_y_images[i] for worker in workers][:max_workers_visualized]

                fig, axs = plt.subplots(1, 2*len(tensors) +1, figsize=(20, 5))

                p = 1
                im0 = axs[0].imshow(tensor0, cmap='viridis')
                axs[0].set_title(f'Y image of Datapoint_{i+1} :\n {features[j]}')
                p+=1

                im1 = axs[1].imshow(tensor1, cmap='viridis')
                axs[2].set_title(f'Full Datapoint_{i+1} :\n {features[j]}')
                p+=1

                if colorbar and j == 3: fig.colorbar(im1, ax=axs[0])
                for k, tensor2 in enumerate(tensors):
                    # if k%2 == 0 : continue
                    im2 = axs[2*k + 1].imshow(tensor2, cmap='viridis')
                    axs[2*k +1].set_title(f'Worker_{k} Datapoint_{i+1} :\n {features[j]}')

                    im1 = axs[2*k + 2].imshow(ytensors[k] , cmap='viridis')
                    axs[2*k + 2].set_title(f'Sampled Y image of Datapoint_{i+1} :\n {features[j]}')
                    # if colorbar and j == 3: fig.colorbar(im2, ax=axs[k + 1])
                plt.tight_layout()
            plt.show()

def main():
    train_errors, test_errors, val_losses, train_losses = [], [], [], []

    dataset = 'wi_outdoor_15_buildings_8_reflections'  # dataset to use. For other available datasets, see Parameters.py
    para = Parameters(dataset)
    num_pixels_x = int((para.x_max - para.x_min) / para.cell_width)  # image width
    num_pixels_y = int((para.y_max - para.y_min) / para.cell_width)  # image height
    num_samples = 40        # for sampling RXs and creating images
    num_images = 200        # for sampling RXs and creating images

    visualize_fed_data = 0  # v=0 means no visualization needed; v>0 means number of datapoints to visualize
    gen_plots = False
    use_aug = False

    start_training = True
    n_workers = 2           # number of workers to use for federated learning :: 1 means traditional learning without FL
    w_epochs = 0          # w_epochs = 0 (have trd. training); w_epochs > 0 (have fed. training)
    epochs = 1
    n = 70                 # max training samples are 70

    train_x_images, train_y_images = np.load('train_x_images_rti.npy', 'r')[:min(n, 70)], np.load('train_y_images_rti.npy', 'r')[:min(n, 70)]
    test_x_images, test_y_images = np.load('test_x_images_rti.npy', 'r'), np.load('test_y_images_rti.npy', 'r')
    print('train_x_images.shape, train_y_images.shape, test_x_images.shape, test_y_images.shape', train_x_images.shape, train_y_images.shape, test_x_images.shape, test_y_images.shape)


    def augmentation(train_x_images, train_y_images):
        train_x_images_aug = []; train_y_images_aug = []
        for i in range(train_x_images.shape[0]):
            tx_img = train_x_images[i, 0, ...]
            rx_img = train_x_images[i, 1, ...]
            map_img = train_x_images[i, 2, ...]
            rti_img = train_x_images[i, 3, ...]
            rss_img = train_y_images[i]

            non_zero_pixels = np.nonzero(rx_img)
            non_zero_pixels = np.array(list(zip(non_zero_pixels[0], non_zero_pixels[1])))

            for j in range(num_images):
                selected_pixels = np.random.choice(non_zero_pixels.shape[0], size=num_samples, replace=False)
                selected_pixels = non_zero_pixels[selected_pixels, :]
                selected_pixels = tuple(list(zip(*selected_pixels)))

                rx_img_new = np.zeros_like(rx_img)
                rx_img_new[selected_pixels] = rx_img[selected_pixels]
                rti_img_new = np.zeros_like(rti_img)
                rti_img_new[selected_pixels] = rti_img[selected_pixels]
                new_data = np.stack([tx_img, rx_img_new, map_img, rti_img_new], axis=0)
                train_x_images_aug.append(new_data)

                rss_img_new = np.zeros_like(rss_img)
                rss_img_new[selected_pixels] = rss_img[selected_pixels]
                train_y_images_aug.append(rss_img_new)

                # plotting some sample images
                if gen_plots:
                    fig, axs = plt.subplots(nrows=2, ncols=5, constrained_layout=True)
                    plot_data = [tx_img, rx_img, map_img, rti_img, rss_img, new_data[0, ...], new_data[1, ...],
                                new_data[2, ...], new_data[3, ...], rss_img_new]
                    titles = ['TX', 'RX', 'Map', 'RTI', 'RSS', 'TX', 'RX_samp', 'Map', 'RTI_samp', 'RSS_samp']
                    for idx, ax in enumerate(axs.flat):
                        ax.grid(True)
                        vec = plot_data[idx]; title = titles[idx]

                        im = ax.imshow(vec, aspect='auto', cmap='viridis', interpolation="nearest")
                        cbar = fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.99, aspect=40, pad=0.01)
                        cbar.ax.tick_params()

                        ax.set_xticks(np.arange(0, 50, 5) - 0.5); ax.set_xticklabels(10 * np.arange(0, 50, 5), rotation=45)
                        ax.set_yticks(np.arange(0, 50, 5) + 0.5); ax.set_yticklabels(10 * np.flip(np.arange(0, 50, 5)))
                        # ax.tick_params(labelsize=labelsize)
                        ax.set_title(title)
                    plt.show()
        return [np.array(train_x_images_aug), np.array(train_y_images_aug)]
    def compute_train_error(nunet_obj_rti, train_x_images , train_y_images = train_y_images):
        temp_pred = nunet_obj_rti.predict_rss(train_x_images, True)
        temp_err = np.where(train_y_images == 0.0, train_y_images, np.abs(temp_pred - train_y_images))
        # print('Final train_error', (np.sum(temp_err)) / np.count_nonzero(temp_err))/
        return (np.sum(temp_err)) / np.count_nonzero(temp_err)
    def compute_test_error(nunet_obj_rti, test_x_images = test_x_images, test_y_images = test_y_images):
        nunet_rti_prediction = []

        # mask = np.ones(test_x_images.shape)
        # y_mask = np.ones((test_x_images.shape[0], test_x_images.shape[2], test_x_images.shape[3]))
        # if len(test_x_images.shape) == 4:
        #     for i in range(test_x_images.shape[0]):
        #         mask[i, 1] = np.random.choice([0, 1], size=test_x_images.shape[2:], p=[0.75, 0.25])
        #         mask[i, 3] = mask[i, 1]
        #         y_mask[i] = mask[i, 1]
        #test_x_images_new = test_x_images * mask
        #test_y_images_new = test_y_images * y_mask

        test_x_images_new = test_x_images
        test_y_images_new = test_y_images


        for item, vals in zip(test_x_images_new, test_y_images_new):
            temp_unet = nunet_obj_rti.predict_rss(item)
            nunet_rti_prediction.append(temp_unet)


            # plt.imshow(temp_unet, aspect='auto')
            # plt.show()
            #
            # plt.imshow(vals, aspect='auto')
            # plt.show()

        nunet_rti_prediction = np.reshape(np.array(nunet_rti_prediction), test_y_images_new.shape)
        temp_err = np.where(test_y_images_new == 0.0, test_y_images_new, np.abs(nunet_rti_prediction - test_y_images_new))
        # print('Final test error', np.sum(temp_err) / np.count_nonzero(temp_err))

        # errors = np.array([i for i in nunet_rti_prediction.flatten() if i != 0.0])
        # print(errors.mean(), errors.min(), errors.max())


        return np.sum(temp_err) / np.count_nonzero(temp_err)

    if use_aug:
        train_x_images, train_y_images = augmentation(train_x_images, train_y_images)
        test_x_images, test_y_images = augmentation(test_x_images, test_y_images)
        print('Post augmentation:')
        print('train_x_images.shape, train_y_images.shape, test_x_images.shape, test_y_images.shape',
            train_x_images.shape, train_y_images.shape, test_x_images.shape, test_y_images.shape)

    nunet_obj_rti = UnetClass(para, num_pixels_x, num_pixels_y, train_x_images.shape[1], 0, False) # Global object (Model)

    if w_epochs == 0:
        epochs = 200
        for epoch in range(epochs):
            nunet_obj_rti.training(train_x_images, train_y_images, start_training, worker_epochs=1)    # trd. training
            train_errors.append(compute_train_error(nunet_obj_rti, train_x_images, train_y_images))
            test_errors.append(compute_test_error(nunet_obj_rti, test_x_images, test_y_images))
            print(f'epoch{epoch}: Global train_error {train_errors[-1]}', end = '; ') # compute train error
            print(f'epoch{epoch}: Global test error {test_errors[-1]}\n')     # prediction
    else:
        epochs = nunet_obj_rti.best_epochs//w_epochs
        epochs = 100

        # intialize workers
        workers_nunet_obj_rti = [UnetClass(para, num_pixels_x, num_pixels_y, train_x_images.shape[1], i, False) for i in range(n_workers)]
        workers = [FederatedWorker(i, workers_nunet_obj_rti[i], train_x_images, train_y_images, para, worker_epochs = w_epochs, total_workers = n_workers) for i in range(n_workers)]

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            v = []
            t = []
            for i, worker in enumerate(workers):
                print('worker', i)
                val_loss, train_loss, _ = worker.train_local_model(accelerate = True)        # Training each worker for w_epochs
                print(compute_train_error(worker.nunet_obj, worker.train_x_images, worker.train_y_images))
                
                v.append(val_loss)
                t.append(train_loss)
            val_losses.append(v)
            train_losses.append(t)
            # # compute train + test error for each worker and global model (verbose)
            for i in range(len(workers)):
                print(f'W_{i} epoch{epoch} train_error {compute_train_error( workers[i].nunet_obj, train_x_images, train_y_images)}') # compute train error
                print(f'W_{i} epoch{epoch} test error {compute_test_error( workers[i].nunet_obj, test_x_images, test_y_images)}')      # prediction
            _, _ = FederatedWorker.federated_averaging(nunet_obj_rti, workers)              # Calculating global model from all workers(Avg); and sharing them

            # nunet_obj_rti.model.set_weights(global_weights)
            # for i in range(len(workers)): workers[i].nunet_obj.model.set_weights(avg_weights)

            if (epoch % 10 == 0): nunet_obj_rti.model.save(f'unet_fed(w={n_workers}_e={epoch})_model.keras')    # Save the model for pick up training later

            train_errors.append(compute_train_error(nunet_obj_rti, train_x_images, train_y_images))
            test_errors.append(compute_test_error(nunet_obj_rti, test_x_images, test_y_images))

            print(f'Global train_error {train_errors[-1]}', end = '; ') # compute train error
            print(f'Global test error {test_errors[-1]}\n')     # prediction

            # # Update the description to show the current epoch
            # tqdm.write(f"Epoch {epoch + 1} completed")
            # progress_bar.set_description(f"Epoch {epoch + 1}")
        workers[-1].visualize_divided_data(datapoints = visualize_fed_data, workers = workers)

    print(f'Final train_error {compute_train_error(nunet_obj_rti, train_x_images, train_y_images)}') # compute train error
    print(f'Final test error {compute_test_error(nunet_obj_rti, test_x_images, test_y_images)}')     # prediction

if __name__ == "__main__":
    main()