#! /home/shamik/.local/lib/python3.8/site-packages
"""
nvidia-smi
#### to get into environment
source fedml/bin/activate 
python3 prospire_fedml.py > FedML.txt 2>&1

git add --all
git reset HEAD prospire_FL_shared/fedml/
"""

from Federated import FederatedWorker
from Parameters import Parameters
from UnetClass import UnetClass

from matplotlib import pyplot as plt
import timeit, time, os, psutil, sys
import tensorflow as tf
import numpy as np
import argparse

import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # ensure 1 gpu is free automatic alloaction can allcate accros 2 gpus

# Function to check system memory usage
def check_system_memory_usage(threshold=80):
    memory = psutil.virtual_memory()
    used_percent = memory.percent  # This gives the percentage of total memory used
    print(f"Total Memory: {memory.total / (1024 * 1024):.2f} MB, Used Memory: {memory.used / (1024 * 1024):.2f} MB, Used Percent: {used_percent}%")

    if used_percent > threshold:
        print(f"System memory usage is more than {threshold}%, exiting...")
        sys.exit()
    del memory,used_percent, threshold
    return

def main():
    parser = argparse.ArgumentParser(description='Federated Learning Parameters')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--w_epochs', type=int, default=0, help='Number of epochs per worker')
    parser.add_argument('--epochs', type=int, default=200, help='Total number of epochs')

    args = parser.parse_args()

    n_workers = args.n_workers
    w_epochs = args.w_epochs
    epochs = args.epochs

    print(f'Running with {n_workers} workers, {w_epochs} worker epochs, and {epochs} total epochs')

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
    # n_workers = 1           # number of workers to use for federated learning :: 1 means traditional learning without FL
    # w_epochs = 1            # w_epochs = 0 (have trd. training); w_epochs > 0 (have fed. training)
    # epochs = 200            # total number of epochs to run
    n = 70                 # max training samples are 70
    
    print(f"n_workers: {n_workers}, w_epochs: {w_epochs}, epochs: {epochs}")

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
        
        train_error = (np.sum(temp_err)) / np.count_nonzero(temp_err)
        del temp_err  # Free the temporary variable
        return train_error            
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
            # plt.imshow(vals, aspect='auto')
            # plt.show()

        nunet_rti_prediction = np.reshape(np.array(nunet_rti_prediction), test_y_images_new.shape)
        temp_err = np.where(test_y_images_new == 0.0, test_y_images_new, np.abs(nunet_rti_prediction - test_y_images_new))
        test_error = np.sum(temp_err) / np.count_nonzero(temp_err)

        # Free the temporary variables
        del nunet_rti_prediction, test_x_images_new, test_y_images_new, temp_err
        # print('Final test error', np.sum(temp_err) / np.count_nonzero(temp_err))

        # errors = np.array([i for i in nunet_rti_prediction.flatten() if i != 0.0])
        # print(errors.mean(), errors.min(), errors.max())
        # return np.sum(temp_err) / np.count_nonzero(temp_err)
        
        return test_error

    if use_aug:
        train_x_images, train_y_images = augmentation(train_x_images, train_y_images)
        test_x_images, test_y_images = augmentation(test_x_images, test_y_images)
        print('Post augmentation:')
        print('train_x_images.shape, train_y_images.shape, test_x_images.shape, test_y_images.shape',
            train_x_images.shape, train_y_images.shape, test_x_images.shape, test_y_images.shape)

    nunet_obj_rti = UnetClass(para, num_pixels_x, num_pixels_y, train_x_images.shape[1], id = 0, var_loss=False) # Global object (Model)

    if w_epochs <= 0:
        # epochs = 200
        for epoch in range(epochs):
            nunet_obj_rti.training(train_x_images, train_y_images, start_training, worker_epochs=1)    # trd. training
            train_errors.append(compute_train_error(nunet_obj_rti, train_x_images, train_y_images))
            test_errors.append(compute_test_error(nunet_obj_rti, test_x_images, test_y_images))
            print(f'epoch{epoch}: Global train_error {train_errors[-1]}', end = '; ') # compute train error
            print(f'epoch{epoch}: Global test error {test_errors[-1]}\n')     # prediction
    else:
        # epochs = nunet_obj_rti.best_epochs//w_epochs
        # epochs = 1

        # intialize workers
        workers_nunet_obj_rti = [UnetClass(para, num_pixels_x, num_pixels_y, train_x_images.shape[1], id = i+1, var_loss=False) for i in range(n_workers)]
        workers = [FederatedWorker(i, workers_nunet_obj_rti[i], train_x_images, train_y_images, para, worker_epochs = w_epochs, total_workers = n_workers) for i in range(n_workers)]

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            v, t = [], []
            for i, worker in enumerate(workers):
                print('worker', i+1)
                # print(f"Before {i+1}th worker : ")
                # check_system_memory_usage()
                val_loss, train_loss, _ = worker.train_local_model(accelerate = True)        # Training each worker for w_epochs
                # print(f"After {i+1}th worker :  ")
                # check_system_memory_usage()
                print(compute_train_error(worker.nunet_obj, worker.train_x_images, worker.train_y_images))
                
                v.append(val_loss)
                t.append(train_loss)
            val_losses.append(v)
            train_losses.append(t)
            # # compute train + test error for each worker and global model (verbose)
            for i in range(len(workers)):
                print(f'W_{i} epoch{epoch} train_error {compute_train_error( workers[i].nunet_obj, train_x_images, train_y_images)}') # compute train error
                print(f'W_{i} epoch{epoch} test error {compute_test_error( workers[i].nunet_obj, test_x_images, test_y_images)}')     # prediction
            # print(f"Before averaging : ")
            # check_system_memory_usage()
            _ = FederatedWorker.federated_averaging(nunet_obj_rti, workers)              # Calculating global model from all workers(Avg); and sharing them

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
            
            del v,t, val_loss, train_loss
            
            # if system RAM usage increased by 80%, exit the code
            print(f"End of Epoch : ")
            check_system_memory_usage()
        workers[-1].visualize_divided_data(datapoints = visualize_fed_data, workers = workers)

    print(f'Final train_error {compute_train_error(nunet_obj_rti, train_x_images, train_y_images)}') # compute train error
    print(f'Final test error {compute_test_error(nunet_obj_rti, test_x_images, test_y_images)}')     # prediction

    visualize_metrics = True
    if visualize_metrics:
        epochs = [i for i in range(len(train_errors))]
        global_train_errors = train_errors
        global_test_errors = test_errors

        print(len(global_train_errors))
        print(len(global_test_errors))
        print(f"Epochs = {epochs}")

        plt.figure(figsize=(12, 6))
        plt.plot(epochs, global_train_errors, label='Global Train Error', marker='o')
        plt.plot(epochs, global_test_errors, label='Global Test Error', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('Global Train and Test Errors Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()
        

        # Plotting the train and validation losses for each worker
        plt.figure(figsize=(12, 8))
        
        print(train_errors)
        print(test_errors)
        print(val_losses)
        print(train_losses)

        # flattenn losses
        ftrain_losses = [item for sublist in train_losses for item in sublist]
        fval_losses = [item for sublist in val_losses for item in sublist]
        print(ftrain_losses)
        print(fval_losses)
        
        # Provided data
        train_losses = ftrain_losses[:]
        val_losses = fval_losses[:]

        global_train_errors = train_errors
        global_test_errors = test_errors
        
        plt.plot(global_train_errors, marker='+', linestyle='-', color='g', label='Global Train Error')
        plt.plot(global_test_errors, marker='+', linestyle='-', color='r', label='Global Train Error')
        
        
        # Initializing lists for each worker
        worker_train_losses = [[] for _ in range(n_workers)]
        worker_val_losses = [[] for _ in range(n_workers)]

        # Separating losses for each worker
        for i in range(n_workers):
            worker_train_losses[i] = train_losses[i::n_workers]
            worker_val_losses[i] = val_losses[i::n_workers]

        # Example of how to access the losses for plotting (similar to previous code)
        for i in range(n_workers):
            plt.plot(worker_train_losses[i], marker='o', linestyle='-', label=f'Worker {i+1} Train Loss')
            plt.plot(worker_val_losses[i], marker='o', linestyle='--', label=f'Worker {i+1} Val Loss')


        epochs = list(range(1, len(worker_train_losses[0]) + 1))
        print(epochs)
        plt.plot(epochs, train_losses[:len(worker_train_losses[0])], label='Baseline Training Loss')
        plt.plot(epochs, val_losses[:len(worker_train_losses[0])], label='Baseline Validation Loss')

        # Adding titles and labels
        plt.title('Train and Validation Losses for Each Worker')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        # epochs, train_loss, val_loss = epochs[:], train_errors[:150], test_errors[:150]

if __name__ == "__main__":
    main()