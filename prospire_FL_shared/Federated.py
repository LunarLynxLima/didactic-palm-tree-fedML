from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import time
import gc

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
        
        # Delete variables and force garbage collection
        del start_time
        gc.collect()
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
        # Delete variables and force garbage collection
        del local_weights, weights, i, avg_weights, worker_obj, worker_objs
        gc.collect()
        
        return global_obj.model.get_weights()#, avg_weights

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
