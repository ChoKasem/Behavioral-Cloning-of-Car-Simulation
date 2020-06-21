# Behavioral Cloning of a Car using Simulation 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[NN]: ./images/nn.png "Architecture"
[loss1]: ./images/loss9.png "Model Visualization"
[loss2]: ./images/loss10.png "Model Visualization"
[loss3]: ./images/loss11.png "Model Visualization"
[Relu]: ./images/relu.png "Relu"
[Elu]: ./images/elu.png "Elu"


## Scripts and models
* model.py : script for creating and training model
* drive.py : script for driving the car and saving video
* model10.h5 : the convolution neural network model
* video.py : convert images to mp4 video 

## Setup and Running Instruction

### Installation
1. Download the simulation from this link
2. Clone this repository

### Running the Model
1. Open the simulation in Autonomous mode
2. Run the following line to run the model in simulation
```sh
python drive.py model.h5
```

### Gathering Data
1. Run the simulation in training mode
2. Press `R` or click on record button, then choose a folder where the data will be save
3. Drive the car around and press `R` again to stop. The images of the camera will be save in the folder along with the steering value in a csv file.

### Training Data
1. Specify the data folder in `model.py` that you want to train the network on
2. Run the following code in terminal to build model. Modify the `input_model`, `output_model`, `data_folder`, and `output_image` variable to specify which model to train, the output model name, the data folder that will be used for training, and the name of output loss graph, respectively. 
```sh
python model.py
```

## Model Architecture and Training Strategy
![NN Image][NN]

The architecture is similar to Nvidia deep learning network, but I experiment with the number of filter and layer.

### Model Summary

|Layer (type)           |      Output Shape        |      Param #  | 
|:--------------------:|:--------------------:|:-------------:|
|lambda_18 (Lambda)       |    (None, 160, 320, 3)   |    0         |
|cropping2d_18 (Cropping2D) |  (None, 65, 320, 3)    |    0         
|conv2d_49 (Conv2D)        | (None, 31, 158, 24)    |   1824      
|conv2d_50 (Conv2D)        | (None, 14, 77, 36)    |    21636     
|conv2d_51 (Conv2D)        | (None, 5, 37, 48)     |    43248     
|conv2d_52 (Conv2D)        |  (None, 3, 35, 64)    |     27712     
|conv2d_53 (Conv2D)        |   (None, 1, 33, 70)    |     40390     
|flatten_12 (Flatten)      |    (None, 2310)        |      0         
|dense_37 (Dense)          |     (None, 100)        |       231100    
|dense_38 (Dense)         |      (None, 50)       |         5050      
|dense_39 (Dense)        |       (None, 10)         |       510       
|dense_40 (Dense)       |        (None, 1)          |       11        


Total params: 371,481

Trainable params: 371,481

Non-trainable params: 0

### Description

#### Lambda

The lambda layer is used for normalizing the image so the value is between 0 and 1.

#### Cropping

The cropping layer remove the sky region and the region that the car hood which doesn't help the model to determine the direction that it should drive toward to. Removing them will increase performance speed.

#### Convolution Neural Network

Multiple layers of Convolution Neural Network are used to detect different feature of the scenary to determine the driving direction. Each layer have different number of filter for detecting different feauture. The CNN have a combination of ReLu and ELU activation as well as dropout.

#### Flatten

Flatten layer convert the 2D input into 1D

#### Dense

Dense Layer slowly find the important feature from 1D data and convert it into the steering angle that is required

### Solution Design Approach

#### Creation of Training Set & Training Process

There are 3 mains dataset in which I train the model on. The first dataset is just the car driving normally along the center. The second dataset focus more on the curve portion, especially the area where there is the gap. Lastly, the final dataset contain images of the car going to the center from the edge. This will teach the model to go back to the center everytimes it stay to close to the edge. The model were trained using generator so it doesn't have to compute all the images before running. Moreover, I augmented the dataset by flipping the images and reverse the steering direction by multiplying by -1.

Originally, the first model is similar to the NVIDIA model, which use only ReLu activation. The model is train using Mean Square Error (MSE) loss and Adam optimizer. The number of epoch for each training is 2 and the validation data is 20% of the entire data. However, the model still cause the car to goes outside of the road. Therefore, it is better if the model punish the negativeafter changing to ELu, the performance increase. The model should punish the steering when it is going toward the edges. Therefore, the Elu is a more appropriate choice.

Comaparison between ReLu and Elu is shown below:

![Relu][Relu]
![Elu][Elu]

Dropouts are used to prevent the model from overfitting. The validation should still be higher than the training model. As I train the model with multiple dataset, the validation loss finally goes beyond the training loss.

Here are the loss graph from the training:

![loss1][loss1]
![loss2][loss2]
![loss3][loss3]

## Discussion

### Problem

Gathering data took a lot of time, as expected for machine learning problem. Also, the type of data gather also effect the model. Therefore, I gather different type of dataset and then tune the model by training it different time. At first when I was using ReLu activation, the model did not perform well. However, after using ELu, it improved much more significantly. The activation function for different model have hugh impact on the performance.

### Further Improvement

In the future, the model will require more data as well as regularization or dropout to prevent overfitting. Currently, it cannot deal with the road in the challenge map. By gather data from both map, it can generalize the lane better and know the steering action the car should use.