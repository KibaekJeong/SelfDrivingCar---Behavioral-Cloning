# **Behavioral Cloning**
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_img/center_2019_09_06_17_44_22_726.jpg "Center Lane"
[image2]: ./writeup_img/center_2019_09_11_13_24_26_827.jpg "Center Lane"
[image3]: ./writeup_img/center_2019_09_11_13_52_28_713.jpg "Recovery Image"
[image4]: ./writeup_img/center_2019_09_11_13_52_29_712.jpg "Recovery Image"
[image5]: ./writeup_img/center_2019_09_11_13_52_30_481.jpg "Recovery Image"
[image6]: ./writeup_img/center_2019_09_11_13_52_30_947.jpg "Recovery Image"
[image7]: ./writeup_img/center_2019_09_11_14_03_05_814.jpg "Track 2"

---




### Model Architecture and Training Strategy
####[Link to Final Video](https://youtu.be/bkKKgwG_Hjg)

#### 1. Solution Design Approach

For following project, neural network is used to clone vehicle driving behavior. Using data collected from driving simulator distributed by [Udacity](https://udacity.com), model learns how to steer the vehicle.
Neural network model is based on [NVIDIA model](https://devblogs.nvidia.com/paralelforall/deep-learning-self-driving-cars/), which uses convolutional neural networks to drive a vehicle. The NVIDIA model has been proven to perform well to steer the vehicle, with or without lane markings, on both local roads and highways with minimum training data from humans.

Following project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md  summarizing the results

In order to improve the model, I have utilized data augmentation. I first separated dataset into a training and validation set. For training dataset, I have implemented image flip and brightness variation. Also, I have converted image color space from RGB to YUV, which was done in NVIDIA model. Also, I have modified the NVIDA model by adding dropout layer to mitigate overfitting.

#### 2. Model Architecture

Neural network model for following project is based on the [NVIDIA model](https://devblogs.nvidia.com/paralelforall/deep-learning-self-driving-cars/). Various modification had been tried for improvement, and addition of dropout layer after first fully connected layer was chosen for final model.

Model composition is as follow:
- Image normalization
- Cropping Image
- Convolutional 2D : 5 X 5, Filter:24, Strides: 2 X 2, Activation: ELU
- Convolutional 2D : 5 X 5, Filter:36, Strides: 2 X 2, Activation: ELU
- Drop Out Layer: Rate: 0.35
- Convolutional 2D : 5 X 5, Filter:48, Strides: 2 X 2, Activation: ELU
- Convolutional 2D : 3 X 3, Filter:64, Strides: 1 X 1, Activation: ELU
- Drop Out Layer: Rate: 0.35
- Convolutional 2D : 3 X 3, Filter:64, Strides: 1 X 1, Activation: ELU
- Drop Out Layer: Rate: 0.35
- Flatten
- Fully Connected Layer: Neurons: 100, Activation: ELU
- Fully Connected Layer: Neurons: 50, Activation: ELU
- Fully Connected Layer: Neurons: 10, Activation: ELU
- Fully Connected Layer: Neurons: 1


#### 3. Data Collection
Data has been collected from driving simulator distributed by the [Udacity](https://udacity.com). In the [driving simulator](https://github.com/udacity/self-driving-car-sim), there are two different tracks. For data collection, both tracks were used.

To capture good driving behavior, I first recorded two laps on each track while driving at the center of lane. Here are example images of center lane driving:
Track 1:
![alt text][image1]
![alt text][image2]
Track 2:
![alt text][image7]


In addition to good driving behavior, model needs to learn how to recover when vehicle is located at left or right side of the lane. Therefore, I have recorded vehicle recovering from the side which is shown below.

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]


Track one requires driver to make lots of left turn, which would lead us to have lots of left turn data. In order to balance out the data amount I have implemented image flip. In result I was able to solve data bias. Also, as not all of the image is required for driving(background such as mountain or trees at the top of the feed in video), cropping layer in the final model has been implemented.

After the collection process, I had 48197 number of data points.

#### 4. Training
For training, I have used 10 epochs with batch size of 64. Loss was calculated using mean squared error as following project required regression, not classification. For the optimizer, Adam has been chosen with learning rate of 0.0001.
Through each epochs, validation loss were calculated and helped determining whether model is under or overfitted. Also, I was able to evaluate which parameter combination resulted in best result.
Final models validation loss was 0.0142.
