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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
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


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because my first implementation of the neural network was overfitting the data. Converting the images to grayscale reduces the number of input features by a factor of three, which helped reduce the overfitting.

Here is an example of an original image and its conversion to grayscale:

![alt text][grayscale]

As a second step, I normalized and scaled the image data because this way the neural network trains faster.

I decided to generate additional data because my network was still overfitting the original data set even after performing the steps above and implementing dropout. To add more data to the the data set, I transformed the grayscale images by randomly rotating them by up to ten degrees and adding noise to them. For rotating, I used an implementation from this [article](https://medium.com/@thimblot/data-augmentation-boost-your-image-dataset-with-few-lines-of-python-155c2dc1baec) , which discusses data augmentation for neural networks. For adding noise, I used the numpy random.normal function with zero mean and a sigma value of 10. In the article mentioned above, the author additionally suggests flipping images horizontally. However, this does not make sense for traffic signs, as most of them are unsymmetric, and some of them are even specific to eiher the left or right side, so I left this step out.

Here is an example of an original grayscale image and its noisy rotated version:

![alt text][augmented]

The augmented data set contains four times the amount of images of the original one, as three modified versions added for each image: one which is rotated, one where noise is added, and one where both of these steps are performed.  


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

The first architecture I chose was the original LeNet function from the character recognition lesson. I saw this well-tested, strong model as a good starting point. With this, I could achieve an accuracy of around 80% in the first shot. As this model was overfitting the training set, I took some measures which I already discussed earlier in this report (grayscale images as input, adding a dropout layer after each layer of the network, augmenting the input data). Furthermore I experimented with different batch epoch and keep-probability values and stuck to the ones that gave me the best results.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][12_priority_road] ![alt text][1_speed_limit_30] ![alt text][21_double_curve] ![alt text][25_road_work] 
![alt text][32_end_of_limits] ![alt text][33_turn_right] ![alt text][40_roundabout] ![alt text][8_speed_limit_120]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


