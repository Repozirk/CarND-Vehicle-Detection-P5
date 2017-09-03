
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/vehicle.png
[image2]: ./examples/non_vehicle.png
[image3]: ./examples/extract_features.png
[image4]: ./examples/pipeline1.png
[image5]: ./examples/pipeline2.png
[image6]: ./examples/pipeline3.png
[image7]: ./examples/pipeline4.png
[image8]: ./examples/pipeline5.png
[image9]: ./examples/pipeline_video1.png
[image9]: ./examples/pipeline_video1_1.png
[image10]: ./examples/pipeline_video2.png
[image11]: ./examples/pipeline_video2_1.png
[image12]: ./examples/pipeline_video3.png
[image13]: ./examples/pipeline_video3_1.png
[image14]: ./examples/pipeline_video4.png
[image15]: ./examples/pipeline_video4_1.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I consider the rubric points individually and describe how I addressed each point in my implementation.  


###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in code cell 2 of the IPython notebook "codeP5.ipynb" the function to extract the veature vector is called `get_hog_features` and calculates a Histogram of Oriented Gradient (HOG). The function `get_hog_features` is applied in code cell 3 at line #46 and #52 by using the function `extract_features` to combine color hist and spatial features

Before calling any function, reading in all the `vehicle` and `non-vehicle` images was necessary. This was done in code cell 3 from line #3.
Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1] ![alt text][image2]


####2. Explain how you settled on your final choice of HOG parameters.

Based on the experience during the Vehicle Detection and Tracking Quizzes, I decide to use the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` and added to the feature vector the results from the `color_hist` and `bin_spatial` functions located in code cell 2.
With using different features for the feature_vector, a normalization of the feature vector was performed in line #68 of code cell 3.

![alt text][image3]



####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the HOG feature `get_hog_features` and `color_hist` and `bin_spatial`. The clasiffier is located in code cell 5 and deliverd following results:

```python
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 8412
10.88 Seconds to train SVC...
Test Accuracy of SVC =  0.9899
My SVC predicts:  [ 0.  1.  1.  0.  1.  0.  0.  0.  0.  0.]
For these 10 labels:  [ 0.  1.  1.  0.  1.  0.  0.  0.  0.  0.]
0.01563 Seconds to predict 10 labels with SVC
```

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The method of sliding window search was applied based on the "Udacity Suggestion" and is realized with the function `find_cars` located in code cell 2.

I decided to search in areas of interest depending on the scale for the sliding windows search (see code cell 7 line 17):

```python
    # Define a row of scale values to get different size of search windows
    # Define the area of interest depending on scale values
    for scale in [1, 1.5, 2, 2.5]:     
        if scale == 1:
            ystart = 400
            ystop = 500
            xstart = 500
            xstop = 1000
        elif scale == 1.5:
            ystart = 400
            ystop = 550
            xstart = 450
            xstop = 1250
        elif scale == 2:
            ystart = 400
            ystop = 650
            xstart = 450
            xstop = 1250
        elif scale == 2.5:
            ystart = 500
            ystop = 650
            xstart = 400
            xstop = 1000
```

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As described I searched on 4 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image6] 

And all different scales combined  at line #42 of code cell 7 resulted following image:
```python 
bboxes=bboxes + bbox_list
``` 

![alt text][image7]

From assigned boxes from heatmap using label function creating followin result in the known test image (with threshold of 0.1:
```python
heat = apply_threshold(heat,0.1)
```
![alt text][image8]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_video.mp4)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections `bboxes` in each frame of the video.  From the history of positive detections with help of function `bboxes_history` I created a heatmap and then thresholded that map to identify vehicle positions with a value of 2, at line #54 in code cell 8 .  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the bounding boxes overlaid on the last frame of video as a result of `scipy.ndimage.measurements.label()`:

### Here are the first 3 frames and the 10th frame and their corresponding heatmaps:

Frame 1: Bounding boxes and heatmap for the single frame and below Bounding boxes and heatmap for the last 10 frames
![alt text][image9]
![alt text][image10]

Frame 2: Bounding boxes and heatmap for the single frame and below Bounding boxes and heatmap for the last 10 frames
![alt text][image10]
![alt text][image11]

Frame 3: Bounding boxes and heatmap for the single frame and below Bounding boxes and heatmap for the last 10 frames
![alt text][image12]
![alt text][image13]

.
.
.

Frame 10: Bounding boxes and heatmap for the single frame and below Bounding boxes and heatmap for the last 10 frames
![alt text][image14]
![alt text][image15]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

At the beginning without regarding the results from previous frames the bounding boxes had a lot of jitter during the video stream and there where false detection from the opposite direction traffic of from shadow areas on the road. The false detections from the opposite direction traffic could be avoided by cutting the area of interest. But the false detections from shadow areas on the road and  jitter of the bounding boxes was not solved.

With implementation of adding bounding boxes over the last 10 frames of the video, the whole pipeline got more robust. The jitter disappeared and there where no false detections. A more professional approach to record bounding boxes would be the use of a class. The storage of previous bounding boxes was done with the "deque" function as suggested by my reviewer of the first submission.

A optimzation of  the classifier wouls increase of robustness as well.
  

