import pickle
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from scipy.misc import imsave
from collections import deque

def get_calibration_parameters(calibration_image_path, nx, ny):
    # get image size
    img_size = cv2.imread(calibration_image_path[0]).shape[:2]
    # list to store object points and image points of all the images
    imagepoints = [] # 2d points in image plane 图像上的坐标
    objectpoints = [] # 3d points in real world space 现实中的坐标

    # Creating object points like ([0, 0, 0], [5, 5, 0], ...)
    obj = np.zeros([nx*ny, 3], np.float32)
    obj[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape([-1, 2]) 

    for image_path in calibration_image_path:
        # read each image
        img = cv2.imread(image_path)

        # Convert image to gray scale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chestboard coners
        # flags == None, ret: if this function detected corners or not
        ret, corners = cv2.findChessboardCorners(gray_img, (nx,ny), None) # corners includes all coordinates

        # if corners detected, add to objects points and image points
        if ret == True:
            objectpoints.append(obj)
            imagepoints.append(corners)

    # img = cv2.drawChessboardCorners(c1, (9,6), corners, ret)
    # plt.imshow(img)

    objectpoints = np.array(objectpoints)
    imagepoints = np.array(imagepoints)

    return objectpoints, imagepoints
def video_abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
    elif orient == 'y':
        sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobelx = np.absolute(sobel_x)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    uint8_sobelx = np.uint8(255 * abs_sobelx/np.max(abs_sobelx))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(uint8_sobelx)
    # 6) Return this mask as your binary_output image
    binary_output[(uint8_sobelx >= thresh_min) & (uint8_sobelx <= thresh_max)] = 1

    return binary_output



def video_mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Calculate the magnitude 
    magnitude = np.absolute(np.square(sobel_x) + np.square(sobel_y))

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    uint8_mag = np.uint8(255 * magnitude/np.max(magnitude))

    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(uint8_mag)

    # 6) Return this mask as your binary_output image
    binary_output[(uint8_mag >= mag_thresh[0]) & (uint8_mag <= mag_thresh[1])] = 1

    return binary_output


def video_dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
        # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobel_x)
    abs_sobely = np.absolute(sobel_y)
        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    dires = np.arctan2(abs_sobely, abs_sobelx)
        # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(dires)
        # 6) Return this mask as your binary_output image
    binary_output[(dires >= thresh[0]) & (dires <= thresh[1])] = 1
    return binary_output



def video_hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(hls_img[:,:,2])
        # 3) Return a binary image of threshold result
    binary_output[(hls_img[:,:,2] > thresh[0]) & (hls_img[:,:,2] <= thresh[1])] = 1
    return binary_output


def video_select_combined_binary(undis_img):
    # Convert to HLS color space and make the HLS selection
    HLS_selected = video_hls_select(undis_img, thresh=(90, 255))

    # Gradients direction selection
    Gradient_direction_selected = video_dir_threshold(undis_img, sobel_kernel=9, thresh=(0.7, 1.3))

    # Magnitude of Sobel operator
    Mag_sobel_selected = video_mag_thresh(undis_img, sobel_kernel=9, mag_thresh=(90, 250))


    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( HLS_selected, Gradient_direction_selected, Mag_sobel_selected )) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(HLS_selected)
    combined_binary[(HLS_selected == 1) | (Gradient_direction_selected == 1) & (Mag_sobel_selected == 1)] = 1

    return combined_binary



def corners_unwarp(undistorted_gray_img, src, dst):
    '''
    src: source points ( np.float32([[,],[,],[,],[,]]) )        
    dst: destination points ( np.float32([[,],[,],[,],[,]]) )
    '''
    # 1) Convert to grayscale
    img_size = (undistorted_gray_img.shape[1], undistorted_gray_img.shape[0])
            # offset = 100 # offset for dst points

            # dst = np.float32([[offset,offset],[img_size[0],offset],
            #   [img_size[0]-offset, img_size[1]-offset], 
            #   [offset, img_size[1]-offset]])

    # 2) use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # 3) use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(undistorted_gray_img, M, img_size, flags=cv2.INTER_LINEAR)

    # 4) inverse PerspectiveTransform
    Minv = cv2.getPerspectiveTransform(dst, src)

    return warped, M, Minv

class Line:
    def __init__(self, side):
        self.side = side
        # Was the line found in the previous frame?
        self.found = False

        # Remember x and y values of lanes in previous frame
        self.X = None
        self.Y = None

        # Store recent x intercepts for averaging across frames
        self.x_int = deque(maxlen=10)
        self.top = deque(maxlen=10)

        # Remember previous x intercept to compare against current one
        self.lastx_int = None
        self.last_top = None

        # Remember radius of curvature
        self.radius = None

        # Store recent polynomial coefficients for averaging across frames
        self.fit0 = deque(maxlen=10)
        self.fit1 = deque(maxlen=10)
        self.fit2 = deque(maxlen=10)
        self.fitx = None
        self.pts = []

        # Count the number of frames
        self.count = 0



    def sliding_window_search(self, warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        if self.found == False:
            histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)

            # Create an output image to draw on and  visualize the result
            # out_img = np.dstack((warped, warped, warped))*255

            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0]//2)
            # 首先随便找两个整个图中按列求和最大的点
            if self.side == 'left':
                x_base = np.argmax(histogram[:midpoint])
            else:
                x_base = np.argmax(histogram[midpoint:]) + midpoint

            # Choose the number of sliding windows
            nwindows = 9
            # Set height of windows
            window_height = np.int(warped.shape[0]/nwindows)
            # Identify the x and y positions of all nonzero pixels in the image
            # all nonzero pixels inside current frame 该帧所有的非零点的坐标
            nonzero = warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Current positions to be updated for each window
            # 把之前找的初始车道线的点赋给存储新车道线坐标的变量
            x_current = x_base
            
            # Set the width of the windows +/- margin
            margin = 100
            # Set minimum number of pixels found to recenter window
            # 设定一个探测到的非零点pixel数目的最小判定数，若大于此数则认定为就是车道线
            minpix = 50
            # Create empty lists to receive left and right lane pixel indices
            # 用来存储左右框框里圈到的所有可能是车道线的点的坐标
            lane_line_inds = []

            # Step through the windows one by one
            # 从图的最下面往上搜索
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = warped.shape[0] - (window+1)*window_height
                win_y_high = warped.shape[0] - window*window_height

                # confirm the X indice of line
                win_x_low = x_current - margin
                win_x_high = x_current + margin

                # Identify the nonzero pixels in x and y within the window
                # ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                # (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)) ======>
                # array([ True, False, False, False, False,  True, False, ...], dtype=bool)

                # 返回非框框内的非零点的X坐标
                nonzero_idxs = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                  (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]

                # Append these indices to the lists
                # 把这些检测到的框框内所有非零点的x坐标添加到一个序列里面
                lane_line_inds.append(nonzero_idxs)
                
                # If you found > minpix pixels, recenter next window on their mean position
                if len(nonzero_idxs) > minpix:
                # 如果框框内检测到的非零点有大于我们设的门槛数值，则计算出一个平均值作为新的车道线的X坐标，并更新给车道线坐标变量
                    x_current = np.int(np.mean(nonzerox[nonzero_idxs]))


            # Concatenate the arrays of indices
            # 添加新的非零点进来，会从第一个小方块开始累加起来，用来fit出一个二次函数表示这条车道线
            lane_line_inds = np.concatenate(lane_line_inds)
            
            # Extract line pixel positions
            x_pos = nonzerox[lane_line_inds]
            y_pos = nonzeroy[lane_line_inds] 

            if np.sum(x_pos) > 0:
                self.found = True
            else:
                y_pos = self.Y
                x_pos = self.X
        # Fit a second order polynomial to each
        # fit出左右各一条二次函数来表示左右车道线
            # left_fit = np.polyfit(lefty, leftx, 2)
            # right_fit = np.polyfit(righty, rightx, 2)

        return np.array(x_pos), np.array(y_pos), nonzerox, nonzeroy, self.found

    def found_based_search(self, wraped, margin=50):
        """
        this search is based on previous found, if you already found lines on previous frame,         
        it's not necessary to make slide windows search against.
        """
        if self.found == True:
            nonzero = wraped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            lane_line_inds = ((nonzerox > (np.mean(self.fit0)*(nonzeroy**2) + np.mean(self.fit1)*nonzeroy + np.mean(self.fit2) - margin)) & (nonzerox < (np.mean(self.fit0)*(nonzeroy**2) + np.mean(self.fit1)*nonzeroy + np.mean(self.fit2) + margin)))
            lane_line_inds = np.array(lane_line_inds)
            # Again, extract line pixel positions
            x_pos = nonzerox[lane_line_inds]
            y_pos = nonzeroy[lane_line_inds]

        if np.sum(x_pos) == 0:
            self.found = False
        return np.array(x_pos), np.array(y_pos), nonzerox, nonzeroy, self.found




    def calculate_curvature_radius(self, x_pos, y_pos):

        '''
        this function is used to calculate curvature in meters
        '''
        # set evaluate point
        y_eval = np.max(y_pos)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(y_pos*ym_per_pix, x_pos*xm_per_pix, 2)

        # Calculate the new radii of curvature
        curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])

        return curverad



    def calculate_intercept(self, poly, img_y):
        '''
        calculate the x values of fitted polynomial using bottom and top position inside a frame
        '''
        bottom_x = poly[0] * img_y ** 2 + poly[1] * img_y + poly[2]

        top_x = poly[0] * 0 ** 2 + poly[1] * 0 + poly[2]

        return bottom_x, top_x


    def x_y_pos_sorted(self, x_pos, y_pos):
        '''
        sorted the detected x and y positions in order
        按从小到大排序
        '''
        # np.argsort 给出从小到大排序的idxs
        sorted_idx = np.argsort(y_pos)
        sorted_y_pos = y_pos[sorted_idx]
        sorted_x_pos = x_pos[sorted_idx]
                                                         
        return sorted_x_pos, sorted_y_pos
