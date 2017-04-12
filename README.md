## Buiding Recognition

### Project Basis

Building recognition Using Local Oriented Features
Jing Li and Nigel Allinson

(see pdf in repository for the paper)


### Project Goal

Implement building recognition algorithm outlined in paper and achieve comparable performance.

### Paper Outline

##### Steerable Filter-Based Building Recognition (SFBR)

SFBR contains four parts:

1. Feature Recognition (Steerable Filters)
2. Feature Pooling (Max Pooling)
3. Dimensionality Reduction (Linear Discrimant Analysis)
4. Classification (Support Vector Classifier)

**Implementation**

Based on steerable filter responses given at various locations.

Steerable filters are a set of basis filters which can synthesize any filter with an arbitrary orientation.  Second-order Gaussian and the Hibert transform will be used.

The basis filters and linear combination coefficients are defined in the paper.

Utilizing openCV to import in grayscale and display images.

centeredFilter - creates a numpy array filled with coordinates of positions with (0,0) as the center of the matrix

steerableGaussian + steerableHilbert - functions to implement steerable Gaussian and steerable Hilbert filter. Uses basis filters and linear combination coefficients defined in paper for preprocessing of the images. Creates a kernel for the linear convolution. Convolves the kernel with the image by way of openCV's filter2d function. Filter response changes based on the input angle.

To create 16 feature maps, the following 8 angles are input to the steerable filters:
0,45,90,135,180,225,270,315.

Each feature map is separated into 4x4 sections for max pooling. Max pooling selects the most active feature in each of the 16 section of each feature map. The output is a 256 dimensional vector describing the original image.

Each 256 dimensional vector is then reduced to 39 dimensions through linear discriminant analysis.

All of the above is done in the data_prep.py file to each image. The output dataset is saved as a pickled numpy array with class labels for each descriptive vector.

The output of data_prep.py is a pickled numpy file used by classifier.py. This is becasue data_prep.py takes a very long time (hours) to run. 

The rest of the work is done in classifier.py.

The data is scaled to be Gaussian distributed with zero mean and unit variance so the support vector classifier acts as expected.

The overall dataset that was prepped in data_prep.py is then randomly split 50/50 into a training and test set. The training set and its label are then passed to the supervised learning algoithm. The program then passes the test set to the trained support vector classifier to predict the class of each building and compares that to the actual labels. Finally the number of correctly predicted images is printed to the console along with the precentage of correct predictions.



**Dataset Info**

The Sheffield Building Image Dataset is used for training and for testing.
https://www.shef.ac.uk/eee/research/iel/research/buildings_data
40 classes of building - each class represents a different building.
within each class, the pictures vary in terms of lighting, orientation and the section the of building pictured.

Below are the number of pictures in the database for each of the 40 classes.

S1 - 334,
S2 - 112,
S3 - 124,
S4 - 70,
S5 - 135,
S6 - 103,
S7 - 100,
S8 - 104,
S9 - 100,
S10 - 134,
S11 - 83,
S12 - 102,
S13 - 99,
S14 - 137,
S15 - 101,
S16 - 98,
S17 - 111,
S18 - 71,
S19 - 149,
S20 - 88,
S21 - 138,
S22 - 124,
S23 - 89,
S24 - 90,
S25 - 91,
S26 - 122,
S27 - 106,
S28 - 64,
S29 - 125,
S30 - 87,
S31 - 72,
S32 - 94,
S33 - 61,
S34 - 35,
S35 - 66,
S36 - 85,
S37 - 67,
S38 - 83,
S39 - 98,
S40 - 126


After testing with a random split of the data, this implementation of the algorithm was between 95.5 ~ 96.8 percent accurate, which is slightly higher than the number in the paper. The reason for this is most likely the sheffield dataset has grown larger since paper has been published. The paper cites 3,192 images in the database but at the time of download, there were 4178 images to train and test the algorithm.
