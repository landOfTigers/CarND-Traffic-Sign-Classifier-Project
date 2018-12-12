# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[distribution]: ./output_images/dist_training_set.png "Distribution"
[grayscale]: ./output_images/grayscale.png "Grayscale"
[augmented]: ./output_images/augmented.png "Augmented"
[12_priority_road]: ./web_images/12_priority_road.jpg "Priority road"
[1_speed_limit_30]: ./web_images/1_speed_limit_30.jpg "30"
[21_double_curve]: ./web_images/21_double_curve.jpg "Double curve"
[25_road_work]: ./web_images/25_road_work.jpg "Road work"
[32_end_of_limits]: ./web_images/32_end_of_limits.jpg "End of limits"
[33_turn_right]: ./web_images/33_turn_right.jpg "Turn right"
[40_roundabout]: ./web_images/40_roundabout.jpg "Roundabout"
[8_speed_limit_120]: ./web_images/8_speed_limit_120.jpg "120"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/landOfTigers/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used standard python and numpy functions to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

My exploratory visualization of the data set can be seen in the report.html file. It displays an example image for each of the 43 classes and sets the class number and name of the traffic sign (extracted from the csv file) as the title.

Addionally, here is a distribution of the classes in the training examples:
![alt text][distribution]

There is a strong variation in the number of training examples for each traffic sign. This might lead to the neural network having a harder time in classifying signs with few examples correctly.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because my first implementation of the neural network was overfitting the data. Converting the images to grayscale reduces the number of input features by a factor of three, which helped reduce the overfitting.

Here is an example of an original image and its conversion to grayscale:

![alt text][grayscale]

As a second step, I normalized and scaled the image data because this way the neural network trains faster.

I decided to generate additional data because my network was still overfitting the original data set even after performing the steps above and implementing dropout. To add more data to the the data set, I transformed the grayscale images by randomly rotating them by up to five degrees and adding noise to them. For rotating, I used an implementation from this [article](https://medium.com/@thimblot/data-augmentation-boost-your-image-dataset-with-few-lines-of-python-155c2dc1baec) , which discusses data augmentation for neural networks. For adding noise, I used the numpy random.normal function with zero mean and a sigma value of 5, as these values yielded good results in reducing the overfitting of the network. In the article mentioned above, the author additionally suggests flipping images horizontally. However, this does not make sense for traffic signs, as most of them are unsymmetric, and some of them are even specific to eiher the left or right side, so I left this step out.

Here is an example of an original grayscale image and its noisy rotated version:

![alt text][augmented]

The augmented data set contains four times the amount of images as the original one, as three modified versions were added for each image: one which was rotated, one where noise was added, and one where both of these steps were performed.  


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Dropout   	      	| keep probability = 0.8 						|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16					|
| Dropout   	      	| keep probability = 0.8 						|
| Flatten				| output 400   									|
| Fully connected		| output 120   									|
| RELU					|												|
| Dropout   	      	| keep probability = 0.8 						|
| Fully connected		| output 84   									|
| RELU					|												|
| Dropout   	      	| keep probability = 0.8 						|
| Fully connected		| output 43   									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer with a learning rate of 0.001. I achieved the best results with a batch size of 32, 15 epochs and a dropout keep-probability of 0.8.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem

My final model results were:
* training set accuracy of 0.990
* validation set accuracy of 0.950 
* test set accuracy of 0.939

The first architecture I chose was the original LeNet function from the character recognition lesson. I saw this well-tested, strong model as a good starting point. With this, I could achieve an accuracy of around 80% in the first shot. As this model was overfitting the training set, I took some measures which I already discussed earlier in this report (grayscale images as input, adding a dropout layer after each layer of the network, augmenting the input data). 

Furthermore I experimented with different batch epoch and keep-probability values and stuck to the ones that gave me the best results. My observation is that the model performs better with lower batch sizes and more epochs than the original design. I tested batch sizes from 16 to 128 in multiples of two steps and found 32 to be the best value. I increased the epoch size from 10 to 15 when I noticed that the model wasn't done training after 10 epochs.

The key factor in reducing the overfitting (around ten per cent lower validation than training accuracy) was to introduce dropout layers, not only after the first convolutional layer, like I did in the beginning, but after each one of the layers in the network. In doing so, I had to increment the keep-probability from 0.5 to 0.8, otherwise the performance of the network would deteriorate in general (meaning that the even training accuracy went down significantly).

I used an Adam optimizer, which means that the learning rate for a weight is divided by a moving average of recent gradients for that weight. This way, the model trains faster as compared to stochastic gradient descent.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][1_speed_limit_30] ![alt text][8_speed_limit_120] ![alt text][32_end_of_limits]
![alt text][12_priority_road]  ![alt text][21_double_curve] ![alt text][25_road_work] 
![alt text][33_turn_right] ![alt text][40_roundabout] 

The road work sign might be difficult to classify because it is scratched. The speed limit signs might also be difficult to classify because they are at an angle. And the roundabout sign might be difficult to classify because it is low in contrast. I had to crop the images so that the sign fills most of the image surface. Otherwise, the predictions were poor for signs that were far away in the image, probably because too much information was lost when downscaling the images to 32 by 32 pixels.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image									| Prediction									| 
|:-------------------------------------:|:---------------------------------------------:| 
| Turn right ahead						| Turn right ahead 								| 
| Speed limit (120km/h)					| No passing for vehicles over 3.5 metric tons	|
| Roundabout mandatory					| Roundabout mandatory							|
| Priority road							| Priority road					 				|
| Road work								| Road work										|
| End of all speed and passing limits	| End of all speed and passing limits			|
| Double curve							| Right-of-way at the next intersection			|
| Speed limit (30km/h)					| Speed limit (30km/h)							|


The model was able to correctly guess 6 of the 8 traffic signs, which gives an accuracy of 75%. This is a decent result for images randomly taken from the web and a the current state of the network. 

Here is my analysis of the two mistakes the model made: 

When looking at the double curve example image that I displayed in the exploratory visualization section of the Jupyter notebook, it becomes obvious that the resolution in the training set is too low to distinguish between a double curve and a right-of-way at the next intersection symbol, even for a human eye. To mitigate this problem, one could implement an algorithm for the pre-processing phase that would automatically detect a bounding box around the traffic sign and crop the image in such a way that the sign covers most of the image space, thus improving the resolution of the symbols.

The 120 km/h speed limit image might have been misclassified because the picture is taken from an angle. This could be compensated by applying a perspective transformation, like we did in the advanced lane lines project.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

When making predictions, the model is relatively certain about it's decisions (all of them have a top softmax probability of over 90 per cent). To illustrate this, here are the top five soft max probabilities for the the eight traffic signs:

##### Turn right ahead                                                                                       ##### Speed limit (120km/h)
| Probability 	| Prediction							| | Probability 	| Prediction									| 
|:-------------:|:-------------------------------------:| |:---------------:|:---------------------------------------------:|
| 1.         	| Turn right ahead   					| | .967 			| No passing for vehicles over 3.5 metric tons	| 
| 0.     		| Right-of-way at the next intersection	| | .016 			| Roundabout mandatory 							|
| 0.			| No entry								| | .008			| Slippery road									|
| 0.	   		| Keep left								| | .004			| No passing									|
| 0.			| Ahead only							| | .002			| Dangerous curve to the left					|

##### Roundabout mandatory
| Probability 			| Prediction											| 
|:---------------------:|:-----------------------------------------------------:| 
| .967         			| Roundabout mandatory   								| 
| .029     				| Priority road 										|
| .026					| End of no passing by vehicles over 3.5 metric tons	|
| .019	      			| Right-of-way at the next intersection					|
| .001				    | Speed limit (100km/h)      							|

##### Priority road
| Probability 			| Prediction											| 
|:---------------------:|:-----------------------------------------------------:| 
| 1.         			| Priority road   										| 
| 0.     				| Roundabout mandatory 									|
| 0.					| Speed limit (100km/h)									|
| 0.	      			| Yield													|
| 0.				    | Speed limit (30km/h)									|

##### Road work
| Probability 			| Prediction											| 
|:---------------------:|:-----------------------------------------------------:| 
| 0.945        			| Road work   											| 
| 0.055    				| Bumpy road 											|
| 0.					| Turn right ahead										|
| 0.	      			| Keep left												|
| 0.				    | Traffic signals										|

##### End of all speed and passing limits
| Probability 			| Prediction											| 
|:---------------------:|:-----------------------------------------------------:| 
| 0.999        			| End of all speed and passing limits					| 
| 0.     				| End of no passing 									|
| 0.					| End of speed limit (80km/h)							|
| 0.	      			| Priority road											|
| 0.				    | Speed limit (30km/h)     								|

##### Double curve
| Probability 			| Prediction											| 
|:---------------------:|:-----------------------------------------------------:| 
| 0.976        			| Right-of-way at the next intersection					|  
| 0.011   				| Beware of ice/snow 									|
| 0.011					| Slippery road											|
| 0.001	      			| Children crossing										|
| 0.				    | Double curve     										|

##### Speed limit (30km/h)
| Probability 			| Prediction											| 
|:---------------------:|:-----------------------------------------------------:| 
| 1.        			| Speed limit (30km/h)   								| 
| 0.					| Speed limit (50km/h) 									|
| 0.					| Speed limit (20km/h)									|
| 0.					| Speed limit (60km/h)									|
| 0.				    | Speed limit (80km/h)     								|
