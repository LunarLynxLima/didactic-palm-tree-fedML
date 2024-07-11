import numpy as np
import timeit, time
from UnetClass import UnetClass
from Parameters import Parameters
from matplotlib import pyplot as plt

dataset = 'wi_outdoor_15_buildings_8_reflections'  # dataset to use. For other available datasets, see Parameters.py
para = Parameters(dataset)
num_pixels_x = int((para.x_max - para.x_min) / para.cell_width)  # image width
num_pixels_y = int((para.y_max - para.y_min) / para.cell_width)  # image height
# for sampling RXs and creating images
num_samples = 40
num_images = 200

gen_plots = False
use_aug = False

start_training = True
n_workers = 2 # number of workers to use for federated learning :: 1 means traditional learning without FL

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

def federated_averaging(global_model, worker_models, weights):
    global_weights = global_model.get_weights()
    num_workers = len(worker_models)
    
    # Initialize averaged weights
    new_weights = [np.zeros_like(w) for w in global_weights]
    
    # Weighted average of local models' weights
    for i, worker_model in enumerate(worker_models):
        local_weights = worker_model.get_weights()
        for j in range(len(new_weights)):
            new_weights[j] += local_weights[j] * weights[i]
    
    # Set the averaged weights to the global model
    global_model.set_weights(new_weights)
    # return global_model

class FederatedWorker:
    def __init__(self, worker_id, nunet_obj, train_x_images, train_y_images, params, worker_epochs=100):
        self.worker_id = worker_id
        self.nunet_obj = nunet_obj
        self.model = nunet_obj.model
        self.train_x_images = train_x_images
        self.train_y_images = train_y_images
        self.worker_epochs = worker_epochs
        self.params = params

    def train_local_model(self):
        # Train local model and return training time, val_loss, and train_loss
        start_time = time.time()
        self.nunet_obj.training(self.train_x_images, self.train_y_images, start_training=True, worker_epochs=self.worker_epochs)
        training_time = time.time() - start_time
        val_loss, train_loss = self.model.get_train_stats()
        return val_loss, train_loss, training_time
    

def main():
    r_train_x_images, r_train_y_images = np.load('train_x_images_rti.npy', 'r'), np.load('train_y_images_rti.npy', 'r')
    test_x_images, test_y_images = np.load('test_x_images_rti.npy', 'r'), np.load('test_y_images_rti.npy', 'r')
    train_x_images, train_y_images = np.array_split(r_train_x_images, n_workers), np.array_split(r_train_y_images, n_workers)
    
    print(f'number of sub-data : {len(train_y_images)}; ', 
          'train_x_images[0].shape : ', train_x_images[0].shape, '; train_y_images[0].shape : ', train_y_images[0].shape, 
          '; test_x_images.shape : ', test_x_images.shape, '; test_y_images.shape : ', test_y_images.shape)    

    if use_aug:
        train_x_images, train_y_images = augmentation(train_x_images, train_y_images)
        test_x_images, test_y_images = augmentation(test_x_images, test_y_images)
        print('Post augmentation:')
        print('train_x_images.shape, train_y_images.shape, test_x_images.shape, test_y_images.shape',
              train_x_images.shape, train_y_images.shape, test_x_images.shape, test_y_images.shape)

    global_nunet_obj_rti = UnetClass(para, num_pixels_x, num_pixels_y, train_x_images[-1].shape[1], False)
    workers_nunet_obj_rti = [UnetClass(para, num_pixels_x, num_pixels_y, train_x_images[-1].shape[1], False) for i in range(n_workers)]

    # intialize workers
    workers = [FederatedWorker(i, workers_nunet_obj_rti[i], train_x_images[i], train_y_images[i], para) for i in range(n_workers)]
    epochs = UnetClass.best_epochs
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}')
        
        # Train local models
        for worker in workers:
            worker.train_local_model()
        
        # Federated averaging
        federated_averaging(global_nunet_obj_rti.model, [worker_obj.model for worker_obj in workers],  [1/n_workers]*n_workers)

    # compute train error
    temp_pred = global_nunet_obj_rti.predict_rss(r_train_x_images, True)
    temp_err = np.where(train_y_images == 0.0, r_train_y_images, np.abs(temp_pred - train_y_images))
    train_error = (np.sum(temp_err)) / np.count_nonzero(temp_err)
    print('train_error', train_error)
    # prediction
    nunet_rti_prediction = []
    for item, vals in zip(test_x_images, test_y_images):
        temp_unet = global_nunet_obj_rti.predict_rss(item)
        nunet_rti_prediction.append(temp_unet)
    nunet_rti_prediction = np.reshape(np.array(nunet_rti_prediction), test_y_images.shape)
    temp_err = np.where(test_y_images == 0.0, test_y_images, np.abs(nunet_rti_prediction - test_y_images))
    print('test error', np.sum(temp_err) / np.count_nonzero(temp_err))


if __name__ == "__main__":
    main()

## main code base ends here