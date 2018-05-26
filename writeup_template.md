## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1a]: ./output_images/car.png
[image1b]: ./output_images/notcar.png
[image2a]: ./output_images/car_notcar_hog.PNG
[image2b]: ./output_images/car_notcar_hog2.PNG
[image2c]: ./output_images/car_notcar_hog3.PNG
[image3a]: ./output_images/sliding_window_overlap_0_3.PNG
[image3b]: ./output_images/sliding_window_overlap_0_8.PNG
[image4]: ./output_images/sliding_window.PNG
[image5a]: ./output_images/pipeline_overlapping.png
[image5b]: ./output_images/pipeline_No_overlapping.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in cells 3 and 4 of the IPython notebook. I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1a]

![alt text][image1b]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here are examples using HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2a]

![alt text][image2b]

![alt text][image2c]


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and settled on the indicated values below. The main factor used in making this decision was the accuracy score obtained by testing each of the classifiers resulting from the training process.
color_space = 'YCrCb'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_size = (32,32)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using LinearSVC from sklearn.svm. Images of both cars and notcars were used from the provided training dataset.
'extract_features' (cell 8) is used to extract spatial, histogram and HOG features from each image.
In cell 14, these features were scaled using StandardScaler to account for the different features. LinearSVC is then used to train a support vector machine classifier with a test accuracy of approximately 99%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Cell 9 contains code for slide_window(). This function is used in cell 15 to obtain windows for an overlap of 50%. These windows are then used to search for car images in search_windows (cell 12) using the above trained SVM classifier.

The HOG parameters were fixed to use the values explained previously. The overlap value was emprically chosen after trying various values including 30% and 80%:

Overlap = 30%


![alt text][image3a]


Overlap = 80%


![alt text][image3b]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately the selected HOG parameter values and the 50% overlap value was used to serach for car images in windows in each of the test images (cell 15). 

![alt text][image4]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. But applying a threshold to the heatmap was a double-eged sword. I eliminated all  false positives but also reduced detection rate by around 15 %. So, I decided not to apply a threshold when generating the final result. 

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the entire pipeline on the test images with overlapping boxes:

![alt text][image5a]

Without overlapping boxes (using label):

![alt text][image5b]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. A major problem was false positives in the detection process. These were eliminated by trying out different values for the parameters.
2. Another problem is the processing time for each of the frames. 
3. The tuning process uses a single test video which means that the parameters obtained might not generalize well
4. Differing illumination conditions and car conditions have to be accounted for in the training data.
5. Merging boxes when detecting multiple vehicles could throw off calculation of metrics for individual vehicles.
6. Causal data can be used to improve the detection process. 
7. Detection of the same vehicle across multiple frames can be changed to detection in the first frame and tracking in the following frames. This would make the pipeline faster.

