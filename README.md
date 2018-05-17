**Lane Lines Detection and Marking using Multiple Visual Features and Perspective Transformation**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./examples/before-undis.png "Road Transformed before"
[image3]: ./examples/after-undis.png "Road Transformed after"
[image4]: ./examples/combination_selection_step.png "Binary Example"
[image5]: ./examples/wraped_img.png "Warp Example"
[image6]: ./examples/src_dst.png "src and dst points"
[image7]: ./examples/color_fit_lines.jpg "Fit Visual"
[image8]: ./examples/test2.png "Output"
[video1]: ./test_video_output/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 2-th and 3-th code cells of the IPython notebook located in "./P4 Advanced Lane Lines Detection.ipynb".

I start by preparing `object point`,which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x,y) plane at z=0, such that the object points are the same for each calibration image. Thus, `obj` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I sucessfully detect all chessboard corners in a test image. `imagepoints` will be appended with the (x,y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objectpoints` and `imagepoints` to compute the camera calibration and distortion coefficients using the function I have written called `cal_undistort`, it uses the `cv2.calibrateCamera()` function to calculate the undistort parameters, then applied the distortion correction to the test image using `cv2.undistort()` function and used a test image to see the effect:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one: (1) At first I have already got the imagepoints and objectpoints from the calibration images. (2) I read a test image from test_images folder using cv2.imread. (3) Convert the BGR image to RGB scale. (4) Use 'cal_undistort()' function to undistort this image.

The particular process could be checked in the cells with the title "Apply a distortion correction to raw images".

Before undistort:

![alt text][image2]

After undistort:

![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

To create the thresholded binary image, we have many methods (After the cells with title 'Use color transforms, gradients, etc., to create a thresholded binary image').

(1) Absolute Sobel Operator. This method take the derivatiive in the x or y orient of the gray-scale image, x or y could be presupposed by yourself, and apply absolute operation to calculated gradient. Then you will set the minimal and maximal thresholds, the absoluted grdient values which satisfy the thresholds will be judged as 1, otherwise 0, then we get the binary image.

(2) Magnitude of Sobel Operator. This method calculate the magnitude using sobel on x and on y orients. And then it applies the same process as (1).

(3) Gradients direction. Firt calculate the gray-scale image, then calculate gradient on both x and y orients, take absolute values of them. Use arctan to calculate the direction of the gradients, use thresholds to select pixels from the calculated direction mask, note that these thresholds should be angles.

(4) HLS Selection. Simply convert the RGB or BGR image to HLS scale, apply the thresholds to HLS-scaled image.

(5) Combined Selection. This method is intend to use all the selection methods from (1) to (4). You can select two or three even four of the methods above to make the selection. For me I use HLS selection, magnitude of Sobel opertor and Gradient direction selection, then I got 3 binary images, and I can use the logic operation to make the final selection.

I used a combination of HLS, gradient direction and magnitude of Sobel operator selections to generate 3 binary images. Here's an example of my output for this step. (note: this is not actually from one of the test images)

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform located in the cells with title, the code includes a function called `corners_unwarp()`, which appears in the first code cell with the title "Apply a perspective transform to rectify binary image".  The `corners_unwarp()` function takes as inputs an image (`undistorted_gray_img`), as well as source (`src`) and destination (`dst`) points.  

#### Actually I didn't chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

I just chose the `src` and `dst` empirically:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 495,490      | 50,300        | 
| 200,700      | 170,700      |
| 800,490     | 1230,300      |
| 1115,700      | 1110,700       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Here is the illustration of `src` and `dst` points from the images:

![alt text][image6]

Here is the wraped images:

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I use `sliding_window_search` function to find the lane line in the first frame of video. 
Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this (Cells with title "Detect lane pixels and fit to find the lane boundary"):

![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the code cells with title "From now, let's test on videos", the `calculate_curvature_radius` function calculates the curvature of lane line and the function `process_frames` estimates the bias of vehicle's position from center.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in `process_frames` function on every frames of video.  Here is an example of my result on a test image:

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./test_video_output/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

This project I use a combine slection approach to select the binary image of each frame, at the first frame I need to apply a sliding window search of total image, after that I will search for the lane line based on searched results of the first frame.


