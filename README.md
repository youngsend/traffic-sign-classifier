## Project: Build a Traffic Sign Recognition Program
#### install pytorch with conda

- check CUDA version:

  ```bash
  $ cat /usr/local/cuda/version.txt
  CUDA Version 10.2.89
  ```

- according to https://pytorch.org/, the command is:

  ```bash
  conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
  ```


### Data Set Summary & Exploration

#### 1. a basic summary of the data set.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43.

#### 2. an exploratory visualization of the dataset.

- First, I plot a histogram for sample number of each class, in training, validation and test set: ![](/home/sen/senGit/traffic-sign-classifier/examples/histogram.png)
  - we can see that the class distribution is similar in training, validation and test set.

- Second, I plot each traffic sign name and their image, so that I can easily know the meaning of an traffic sign image, as well as which traffic signs are included. ![](/home/sen/senGit/traffic-sign-classifier/examples/label-and-image.png)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

- For preprocessing, I only normalized the range from [0, 255] to [0, 1] by dividing 255.
- I have not generated additional data.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

```bash
LeNetRevised(
  (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  (bn1): BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=43, bias=True)
)
```

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

- I used  `torch.utils.data.DataLoader` which can provide batches when training.
- During training, I used `SGD` optimizer with learning rate of `0.01`.
- I used a batch size of `64` (which is the parameter of `DataLoader`) and epoch number of `70`.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of 100%.
* **test set accuracy of 93%. (which can be seen in `Traffic_Sign_Classifier.ipynb`)**
* I used validation set only during training, and the final training loss is `0.006289869131625678`, the final validation loss is `0.2103424778886066`. It seems the trained model is overfitted.

If a well known architecture was chosen:

* What architecture was chosen?
  * I used `LeNet` as the base model, and in order to mitigate the overfitting, I **added a `BatchNormalization` layer after each convolution layer**. 
  * Besides, I deleted one `Linear` layer to reduce the model parameters. 
  * These two practice all make validation loss less.
* Why did you believe it would be relevant to the traffic sign application?
  * Because `LeNet` is an image classification application.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
  * The accuracy on test set is 93%.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs (and after they are resized to (32, 32)) that I found on the web: ![](/home/sen/senGit/traffic-sign-classifier/examples/five-images.png)

- The 2nd and 4th image may be difficult for the model to predict, because the original image is not square, and after resized to (32, 32), the shape of traffic sign will become unusual.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

- Here are the results of the prediction:

|                 Image                 |              Prediction               |
| :-----------------------------------: | :-----------------------------------: |
|         Speed limit (60km/h)          |         Speed limit (60km/h)          |
|         **Children crossing**         |       **Wild animals crossing**       |
|             Priority road             |             Priority road             |
|               Road work               |               Road work               |
| Right-of-way at the next intersection | Right-of-way at the next intersection |

- The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. Since there are only 5 images, 80% accuracy is comparative to test set accuracy (93%).

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the **last cell** of the Ipython notebook.

- For all five images, the highest probability is always very high:

```bash
# the 5 highest probability for each image (a row is the result of an image)
tensor([[1.0000e+00, 3.4132e-19, 3.2332e-19, 4.2221e-20, 2.6393e-21],
        [9.9988e-01, 7.7522e-05, 3.4883e-05, 7.0886e-06, 3.2687e-06],
        [1.0000e+00, 5.6886e-13, 7.6746e-14, 5.2081e-17, 3.7606e-17],
        [1.0000e+00, 1.1250e-08, 8.6519e-12, 2.5753e-13, 3.8149e-14],
        [1.0000e+00, 4.3172e-12, 1.4027e-12, 6.1071e-14, 2.2836e-14]],
       grad_fn=<TopkBackward>)
# the class label corresponding to 5 highest probabilities shown above.
tensor([[ 3, 19,  5,  2, 23],
        [31, 25, 11, 29, 20],
        [12,  6, 32, 41, 36],
        [25, 21, 22,  4, 20],
        [11, 21, 30, 19, 27]])
```

- It is dangerous that when the model is wrong, the probability is also very high (nearly 1.0).

#### important suggestions from reviewer

- For further improvements, here are some preprocessing techniques that have proven to work on this dataset:
  1. Convertion to grayscale can also help.
  2. Try [Augmenting](https://github.com/aleju/imgaug) your training dataset with rotation, distortion, scaling ,translation  and cropping images with varied contrast/brightness or normalized  histograms) can improve the robustness of the model and give better  results
  3. If images are not very bright, can do pre-processing like [histogram equalization](http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html) help in enhancing.

