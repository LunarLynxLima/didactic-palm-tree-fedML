
 - The main file is MasterFL.py. It has several global variables that specify the simulation scenario.
 - Parameters.py has parameters for the datasets experiment used
 - UnetClass.py is the file that performs the image to image translation


NOTE:

If you run MasterFL.py it will load a saved model and perform prediction. The prediction accuracy will be printed.
MasterFL.py loads saved data. Then it performs the data augmentation. However, for your part, you don't have to do data augmentation.
First, you have to create 4 datasets for 4 different workers.
For each of the workers then you can use the FL framework. Finally, evaluate the prediction accuracy on test set.
Compare the performance with that of the existing model.


