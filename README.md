# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[image_orig]: ./images/image_orig.jpg
[cropped]: ./images/cropped.jpg
[flipped]: ./images/flipped.jpg
[left_offset]: ./images/left_offset.jpg
[right_offset]: ./images/right_offset.jpg

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

To train the model ...
```sh
mkdir ../models
python model.py
```

The model will be placed in the ../models directory. To drive the simulator using this new model ...
```sh
python ../models/model.h5
```

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is a slightly modified version of the Nvidia neural network proposed in the lectures. The only modification was
adding dropout layers after the first 2 fully connected layers

#### 2. Attempts to reduce overfitting in the model

I added 2 dropout layers after the first 2 fully connected layers in an attempt to prevent overfitting. Additionally, using only 3 training epochs helped to prevent overfitting to the data

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was tuned automatically

#### 4. Appropriate training data

Training data was generated on both tracks, traveling in both directions. I augmented the data by using the left/right offset images as well as flipping each image horizontally to gain more training samples and to create a balanced dataset.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I began using the Nvidia model proposed in the lecture which is laid out as follows

- Convolution (5x5 Kernel + RELU)
- Convolution (5x5 Kernel + RELU)
- Convolution (5x5 Kernel + RELU)
- Convolution (3x3 Kernel + RELU)
- Convolution (3x3 Kernel + RELU)
- Dense FC (100)
- Dense FC (50)
- Dense FC (10)

I initially added a final layer to provide a single output for steering angle
- Dense FC (1)

In order to increase robustness I added 2 dropout layers (with 0.5 dropout rate) after the 100 and 50 Dense layers

Finally, I augmented the incoming training data as follows

- Color conversion from BGR to RGB
- Generated additional training samples by flipping L/R
- Generated additional training samples by utilizing left and right offset cameras
- Normalized the color channels from 0<->255 to -1.0<->1.0
- Cropping off the top of the image that doesn't see much roadway

#### 2. Final Model Architecture

- Input Layer (160px x 320px)
- Normalization (160px x 320px)
- Cropping
- Convolution (5x5 Kernel + RELU)
- Convolution (5x5 Kernel + RELU)
- Convolution (5x5 Kernel + RELU)
- Convolution (3x3 Kernel + RELU)
- Convolution (3x3 Kernel + RELU)
- Dense FC (100)
- Dropout (0.5)
- Dense FC (50)
- Dropout (0.5)
- Dense FC (10)
- Dense FC (1)

#### 3. Creation of the Training Set & Training Process

The training sets were generated by driving around the track both directions, attempt to stay as close to the center of the track. 

![alt text][image_orig]

From here, I normalized images and cropped the image to include only the area of interest, primarily the road.

![alt text][cropped]

Finally, the images were flipped L/R and the left offset and right offset camera images were used as well.

![alt text][flipped]

![alt_text][left_offset]

![alt_text][right_offset]

When training the final model, I had access to 39091 raw images which equates finally to 78182 augmented images.

