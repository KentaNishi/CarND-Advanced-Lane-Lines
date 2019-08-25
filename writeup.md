## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

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

[image1]: ./undistort.png "Undistorted"
[image2]: ./undistort_series.png "Undistorted series"
[image3]: ./binarize.png "Binary"
[image4]: ./warped_binary_image.png "Warp"
[image5]: ./find_lane_points.png "Fit Visual"
[image6]: ./warped_lane_area.png "Output"
[video1]: ./project_videos_output_with_N_5.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first section of the IPython notebook located in "./project_2.ipynb" 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to the test images like this:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at the third section of the IPython notebook ).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform appears in the 4th section of the IPython notebook. To get transform matrix, I used cv2.getPerspectiveTransform function takes as inputs source (`src`) and destination (`dst`) points. And then, I used cv2.warpPerspective.I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([  (200/1280*input_img.shape[1],input_img.shape[0]),
                    (1100/1280*input_img.shape[1],input_img.shape[0]),
                    (700/1280*input_img.shape[1],450/720*input_img.shape[0]),
                    (600/1280*input_img.shape[1],450/720*input_img.shape[0])])
dst = np.float32([  (200/1280*input_img.shape[1],input_img.shape[0]),
                    (1100/1280*input_img.shape[1],input_img.shape[0]),
                    (1100/1280*input_img.shape[1],0),
                    (200/1280*input_img.shape[1],0)])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 720      | 200, 720      | 
| 1100, 720     | 1100, 720     |
| 700, 450      | 1100, 0       |
| 600, 450      | 299, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 16 through 17 in 6th code section in `./project_2.ipynb`

```python
left_fit_real = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
right_fit_real = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
left_curverad = ((1 + (2*left_fit_real[0]*y_eval*ym_per_pix + left_fit_real[1])**2)**1.5) / np.absolute(2*left_fit_real[0])
right_curverad = ((1 + (2*right_fit_real[0]*y_eval*ym_per_pix + right_fit_real[1])**2)**1.5) / np.absolute(2*right_fit_real[0])
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in 7th code section in `./project_2.ipynb` .  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).
My program is following.
The points are,
1. Set lane search window based on most recent detected lane line. 
1. Introduce sanity check.

```python
import sys
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
%matplotlib inline




# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # set the number to check class variables over the last (const_num) iteration
        self.N = 10
        # was the line detected in the last iteration?
        self.detected = np.array([])  
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = np.zeros((720,2))
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = np.zeros((2,3))
        #polynomial coefficients over the last n iterations
        self.recent_fit = []
        #polynomial coefficients for the most recent fit
        self.current_fit =  np.zeros((2,3))
        #radius of curvature of the line over the last n iterations
        self.recent_radius_of_curvature = np.zeros((1,2))
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        
        img_size = (720,1280)
        
        # load the data of calibration
        dist_pickle = pickle.load(open("./camera_cal/pickle.p", "rb" ))
        self.mtx, self.dist = dist_pickle["mtx"], dist_pickle["dist"]
        
        # Set height of windows - based on nwindows above and image shape
        # for perspective transfrom,set 4 points lying along the lines in undistort image and perspective transposed image. 
        src = np.float32([(200/1280*img_size[1],img_size[0]),(1100/1280*img_size[1],img_size[0]),(700/1280*img_size[1],450/720*img_size[0]),(600/1280*img_size[1],450/720*img_size[0])])
        dst = np.float32([(200/1280*img_size[1],img_size[0]),(1100/1280*img_size[1],img_size[0]),(1100/1280*img_size[1],0),(200/1280*img_size[1],0)])

        

        src_x = [src[0][0], src[1][0], src[2][0], src[3][0], src[0][0]]
        src_y = [src[0][1], src[1][1], src[2][1], src[3][1], src[0][1]]
        dst_x = [dst[0][0], dst[1][0], dst[2][0], dst[3][0], dst[0][0]]
        dst_y = [dst[0][1], dst[1][1], dst[2][1], dst[3][1], dst[0][1]]

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)

        
        
        
        
    def detection_update(self,detect):
        self.detected = np.append(self.detected,detect)
        if len(self.detected) > self.N :
            self.detected = self.detected[1:]
        
    
    def fitx_update(self,left_fitx,right_fitx):
        fit_x = np.concatenate((np.reshape(left_fitx,(1,len(left_fitx),1)),np.reshape(right_fitx,(1,len(right_fitx),1))),axis=2)
        if len(self.recent_xfitted)==0:
            self.recent_xfitted = fit_x
        else:
            self.recent_xfitted = np.concatenate((self.recent_xfitted,fit_x),axis=0)
        if len(self.recent_xfitted) > self.N:
            self.recent_xfitted = self.recent_xfitted[1:]
        #endif
        self.bestx = np.mean(self.recent_xfitted,axis=0).reshape(len(right_fitx),2)
    
    def fit_update(self,left_fit,right_fit):
        self.current_fit[0,:] = left_fit
        self.current_fit[1,:] = right_fit
        if len(self.recent_fit)==0:
            self.recent_fit = np.reshape(self.current_fit,(1,2,3))
        else:
            self.recent_fit = np.concatenate((self.recent_fit,np.reshape(self.current_fit,(1,2,3))),axis=0)
        if len(self.recent_fit) > self.N:
            self.recent_fit = self.recent_fit[1:]
        #endif
        self.best_fit = np.mean(self.recent_fit,axis=0)
                                         
    def curventure_update(self,curventure):
        if len(self.recent_radius_of_curvature)<=0:
            self.recent_radius_of_curvature = np.array(curventure).reshape((1,2))
        else:
            self.recent_radius_of_curvature = np.concatenate((self.recent_radius_of_curvature,np.array(curventure).reshape((1,2))),axis=0)
        if len(self.recent_radius_of_curvature) > self.N:
            self.recent_radius_of_curvature = self.recent_radius_of_curvature[1:]

    def process_image(self,image):
        img_size = (image.shape[1],image.shape[0])
        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 60
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # from pixel world to real world
        ym_per_pix = 30/720  #[m/pixel] in y dimension
        xm_per_pix = 3.7/700 #[m/pixel] in x dimension
        ## undistort image
        undist = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

        ## threshold and generate binary image
        mag_and_dir_binary = mag_and_dir_thresh(undist)
        # color threshold
        # Convert to HLS color space and separate the V channel
        s_thresh = (170, 255)
        hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_thresh[0] <= s_channel) & (s_channel <= s_thresh[1])] = 1
        # conbine gradient and color threshold
        threshold_binary_img = np.zeros_like(mag_and_dir_binary)
        threshold_binary_img[(s_binary==1) | (mag_and_dir_binary==1)] = 1

        ## perspective transform
        warped_binary_img = cv2.warpPerspective(threshold_binary_img, self.M, img_size, flags=cv2.INTER_LINEAR)
        window_height = np.int(warped_binary_img.shape[0]//nwindows)
        #the bottom of the image
        ploty = np.linspace(0, warped_binary_img.shape[0]-1, warped_binary_img.shape[0] )
        y_eval = np.max(ploty)
        nonzero = warped_binary_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        #########################################################
        """The first point I said in this explanation is as follows."""
        #########################################################
        if ((len(self.detected)==0)|(self.detected.any()==False)):
            ## Detect lane pixels and fit to find the lane boundary
            # calculate histgram
            bottom_binary_img =  warped_binary_img[warped_binary_img.shape[0]//3*2:,:]
            histogram = np.sum(bottom_binary_img,axis=0)
            # split the histgram for the left and right line and find the peak of the left and right halves of the histogram
            midpoint = np.int(histogram.shape[0]//2)
            # set the first position of search window.
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint
            # Current positions to be updated later for each window in nwindows
            leftx_current = leftx_base
            rightx_current = rightx_base
        else:
            if self.detected[-1]:
                leftx_current = self.current_fit[0,0]*(y_eval)**2 + self.current_fit[0,1]*(y_eval) + self.current_fit[0,2]
                rightx_current = self.current_fit[1,0]*(y_eval)**2 + self.current_fit[1,1]*(y_eval) + self.current_fit[1,2]
            else:
                leftx_current = self.best_fit[0,0]*(y_eval)**2 + self.best_fit[0,1]*(y_eval) + self.best_fit[0,2]
                rightx_current = self.best_fit[1,0]*(y_eval)**2 + self.best_fit[1,1]*(y_eval) + self.best_fit[1,2]
            #endif


        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
            win_y_low = warped_binary_img.shape[0] - (window+1)*window_height
            win_y_high = warped_binary_img.shape[0] - window*window_height
            # Find the four below boundaries of the window ###
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Identify the nonzero pixels in x and y within the window ###
            if ((self.current_fit==0).any()|(self.detected.any()==False)):
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            else:
                good_left_inds = ((nonzerox > (self.current_fit[0,0]*(nonzeroy**2) + self.current_fit[0,1]*nonzeroy + 
                        self.current_fit[0,2] - margin)) & (nonzerox < (self.current_fit[0,0]*(nonzeroy**2) + 
                        self.current_fit[0,1]*nonzeroy + self.current_fit[0,2] + margin))).nonzero()[0]
                good_right_inds= ((nonzerox > (self.current_fit[1,0]*(nonzeroy**2) + self.current_fit[1,1]*nonzeroy + 
                        self.current_fit[1,2] - margin)) & (nonzerox < (self.current_fit[1,0]*(nonzeroy**2) + 
                        self.current_fit[1,1]*nonzeroy + self.current_fit[1,2] + margin))).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window #
            # (`right` or `leftx_current`) on their mean position #
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        #endfor
        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # it a second order polynomial to each using `np.polyfit`
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, warped_binary_img.shape[0]-1, warped_binary_img.shape[0] )
        ploty_inverse = np.linspace(warped_binary_img.shape[0]-1, 0, warped_binary_img.shape[0] )
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty_inverse**2 + right_fit[1]*ploty_inverse + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        # for overray lane area
        lane_area = np.zeros_like(undist)
        warped_area = cv2.warpPerspective(lane_area, self.M, img_size, flags=cv2.INTER_LINEAR)

        left_fit_real = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_real = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        left_curverad = ((1 + (2*left_fit_real[0]*y_eval*ym_per_pix + left_fit_real[1])**2)**1.5) / np.absolute(2*left_fit_real[0])
        right_curverad = ((1 + (2*right_fit_real[0]*y_eval*ym_per_pix + right_fit_real[1])**2)**1.5) / np.absolute(2*right_fit_real[0])
        print(left_curverad, 'm', right_curverad, 'm')
        left_bottom_x = left_fit_real[0]*(y_eval*ym_per_pix)**2 + left_fit_real[1]*(y_eval*ym_per_pix) + left_fit_real[2]
        right_bottom_x = right_fit_real[0]*(y_eval*ym_per_pix)**2 + right_fit_real[1]*(y_eval*ym_per_pix) + right_fit_real[2]
        print(left_bottom_x, 'm', right_bottom_x, 'm')
        left_middle_x = left_fit_real[0]*(y_eval/2*ym_per_pix)**2 + left_fit_real[1]*(y_eval/2*ym_per_pix) +left_fit_real[2]
        right_middle_x = right_fit_real[0]*(y_eval/2*ym_per_pix)**2 + right_fit_real[1]*(y_eval/2*ym_per_pix)+right_fit_real[2]
        print(left_middle_x, 'm', right_middle_x, 'm')
        

        #########################################################
        """The second point I said in this explanation is as follows."""
        #########################################################
        # the width between the lines is about 4.5[m], so ,if it isn't,it seems that lane lines are not detected.
        if ((np.abs(4.5-np.abs(left_bottom_x-right_bottom_x))<=1.0)&(np.abs(4.5-np.abs(left_middle_x-right_middle_x))<=1.0)):
            self.detection_update(True)
            self.fit_update(left_fit,right_fit)
            self.fitx_update(left_fitx,right_fitx)
            self.curventure_update([left_curverad,right_curverad])
            pts_left = np.reshape(np.transpose(np.stack((left_fitx, ploty),axis=0)),(1,ploty.shape[0], 2))
            pts_right = np.reshape(np.transpose(np.stack((right_fitx, ploty_inverse),axis=0)),(1,ploty.shape[0], 2))
            pts = np.concatenate((pts_left, pts_right),axis=1)
            lane_area_overrayed = cv2.fillPoly(warped_area,np.int_([pts]), (0, 255, 0))
            unwarped_lane_area_overrayed_img = cv2.warpPerspective(lane_area_overrayed, self.M_inv, img_size, flags=cv2.INTER_LINEAR)
            lane_area_overrayed_img = cv2.addWeighted(undist, 1, unwarped_lane_area_overrayed_img, 0.2, 0)
        else:
            self.detection_update(False)
            print("detection_false")
            if self.bestx.any()==0:
                self.bestx[:,0] = left_fitx
                self.bestx[:,1] = right_fitx
            #endif
            pts_left = np.reshape(np.transpose(np.stack((self.bestx[:,0], ploty),axis=0)),(1,ploty.shape[0], 2))
            pts_right = np.reshape(np.transpose(np.stack((self.bestx[:,1], ploty_inverse),axis=0)),(1,ploty.shape[0], 2))
            pts = np.concatenate((pts_left, pts_right),axis=1)
            lane_area_overrayed = cv2.fillPoly(warped_area,np.int_([pts]), (0, 255, 0))
            unwarped_lane_area_overrayed_img = cv2.warpPerspective(lane_area_overrayed, self.M_inv, img_size, flags=cv2.INTER_LINEAR)
            lane_area_overrayed_img = cv2.addWeighted(undist, 1, unwarped_lane_area_overrayed_img, 0.2, 0)
        self.line_base_pos = (img_size[0]/2*xm_per_pix-(left_bottom_x+right_bottom_x)/2)#from center to the right
        text = "Vehicle is " + "{:.2f}".format(self.line_base_pos) + "[m] right of center"
        lane_area_overrayed_img = cv2.putText(lane_area_overrayed_img, text, (40,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
        text = "Curventure is " + "{:.2f}".format(np.mean(self.recent_radius_of_curvature[-1])) + "[m]"
        lane_area_overrayed_img = cv2.putText(lane_area_overrayed_img, text, (40,150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

        return lane_area_overrayed_img

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

project_output = 'project_videos_output_complete_ver_with_N_10.mp4'
file_name = "project_video.mp4"
clip = VideoFileClip(file_name)
finding_lane_line = Line()
project_clip = clip.fl_image(finding_lane_line.process_image)

%time project_clip.write_videofile(project_output, audio=False)
```
Here's a ![link to my video result][video1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 

In my program , it is assumed that there is no vehicle cover the lane.
So if there are some vehicle cover the lane, my program doesn't work well.
To solve this problem, I think it is valid that detect vehicle and set search window reshape without vehicle covering area.

Another problem is occured when the curventure is intense.
I think one of the solution is change seach window margin according to the second order fit value. 

