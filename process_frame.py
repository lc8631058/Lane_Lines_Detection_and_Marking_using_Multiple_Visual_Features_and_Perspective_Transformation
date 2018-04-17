import pickle
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from scipy.misc import imsave
from collections import deque

# def cal_undistort(img, objectpoints, imagepoints):
#     # Use cv2.undistort()
#     # calculate the calibration matrix and distortion coefficients
#     img_size = img.shape[:2]
#     ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectpoints, imagepoints, img_size, None, None)
#     undist = cv2.undistort(img, mtx, dist, None, mtx)
#     return undist

def draw_lines(warped, undis_img, img_size, nonzerox, nonzeroy, left_fit, right_fit, leftx_pos, rightx_pos, Minv, margin=15):
    # Draw lines 
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    window_img = np.zeros_like(color_warp)
    
    # Color in left and right line pixels
    window_img[nonzeroy[np.int_(leftx_pos)], nonzerox[np.int_(leftx_pos)]] = [255, 0, 0]
    window_img[nonzeroy[np.int_(rightx_pos)], nonzerox[np.int_(rightx_pos)]] = [0, 0, 255]
    
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
      ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
      ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,0, 255))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,0, 255))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp_1 = cv2.warpPerspective(color_warp, Minv, img_size) 
    newwarp_2 = cv2.warpPerspective(window_img, Minv, img_size) 
    
    # Combine the result with the original image
    color_line = cv2.addWeighted(newwarp_1, 1, newwarp_2, 1, 0)
    result = cv2.addWeighted(undis_img, 1, color_line, 0.3, 0)
    # result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)a

    return result


def process_frames(img):
    # get image size
    img_size = (img.shape[1], img.shape[0])
    
    # undistort image
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectpoints, imagepoints, (img_size[1], img_size[0]), None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # binary select 
    combined_binary = video_select_combined_binary(undist)

    # warp image
    warped, M, Minv = corners_unwarp(combined_binary, src, dst)

    if Left.found == True: 
        leftx_pos, lefty_pos, nonzerox, nonzeroy, Left.found = Left.found_based_search(warped, margin=50)

    if Left.found == False: 
        leftx_pos, lefty_pos, nonzerox, nonzeroy, Left.found = Left.sliding_window_search(warped)

    if Right.found == True: 
        rightx_pos, righty_pos, nonzerox, nonzeroy, Right.found = Right.found_based_search(warped, margin=50)

    if Right.found == False: 
        rightx_pos, righty_pos, nonzerox, nonzeroy, Right.found = Right.sliding_window_search(warped)

    # Calculate polynomial fit based on points
    left_fit = np.polyfit(lefty_pos, leftx_pos, 2)
    right_fit = np.polyfit(righty_pos, rightx_pos, 2)

    # Calculate intercepts of top and bottom position
    left_bott, left_top = Left.calculate_intercept(left_fit, img_size[1])
    right_bott, right_top = Right.calculate_intercept(right_fit, img_size[1])

    # Store recent intercepts and average them
    Left.x_int.append(left_bott)
    Left.top.append(left_top)
    Right.x_int.append(right_bott)
    Right.top.append(right_top)
    # average and update
    left_bott = np.mean(Left.x_int)
    left_top = np.mean(Left.top)
    right_bott = np.mean(Right.x_int)
    right_top = np.mean(Right.top)
    # update to last intercepts
    Left.lastx_int = left_bott
    Left.last_top = left_top
    Right.lastx_int = right_bott
    Right.last_top = right_top

    # Add updated intercepts to x and y points
    leftx_pos = np.append(leftx_pos, left_bott)
    lefty_pos = np.append(lefty_pos, img_size[1])
    leftx_pos = np.append(leftx_pos, left_top)
    lefty_pos = np.append(lefty_pos, 0)

    rightx_pos = np.append(rightx_pos, right_bott)
    righty_pos = np.append(righty_pos, img_size[1])
    rightx_pos = np.append(rightx_pos, right_top)
    righty_pos = np.append(righty_pos, 0)

    # sort points in order
    leftx_pos, lefty_pos = Left.x_y_pos_sorted(leftx_pos, lefty_pos)
    rightx_pos, righty_pos = Right.x_y_pos_sorted(rightx_pos, righty_pos)

    # Save detected lane line points in previous frame
    Left.X = leftx_pos
    Left.Y = lefty_pos
    Right.X = rightx_pos
    Right.Y = righty_pos

    # Now calculate the polynomial again and store it to recent polynomial coefficients
    left_fit = np.polyfit(lefty_pos, leftx_pos, 2)
    Left.fit0.append(left_fit[0])
    Left.fit1.append(left_fit[1])
    Left.fit2.append(left_fit[2])
    left_fit = [np.mean(Left.fit0), np.mean(Left.fit1), np.mean(Left.fit2)]

    right_fit = np.polyfit(righty_pos, rightx_pos, 2)
    Right.fit0.append(right_fit[0])
    Right.fit1.append(right_fit[1])
    Right.fit2.append(right_fit[2])
    right_fit = [np.mean(Right.fit0), np.mean(Right.fit1), np.mean(Right.fit2)]


    # Calculate the x values in fitted polynomial of corresponidng y value
    left_fitx = left_fit[0] * lefty_pos ** 2 + left_fit[1] * lefty_pos + left_fit[2]
    Left.fitx = left_fitx
    right_fitx = right_fit[0] * righty_pos ** 2 + right_fit[1] * righty_pos + right_fit[2]
    Right.fitx = right_fitx

    # Calculate radius of curvature in meters
    left_curverad = Left.calculate_curvature_radius(leftx_pos, lefty_pos)
    right_curverad = Right.calculate_curvature_radius(rightx_pos, righty_pos)

    # Store curvature to class every 2 frames
    if Left.count % 2 == 0:
        Left.radius = left_curverad
        Right.radius = right_curverad

    # Calculate the car position
    car_pos = (left_bott + right_bott) / 2
    bias_from_center = abs(car_pos - img_size[0]/2) * 3.7 / 700 # in meters

    # Draw detected lines on every frame
    result = draw_lines(warped, undist, img_size, nonzerox, nonzeroy, left_fit, right_fit, leftx_pos, rightx_pos, Minv, margin=15)


    # Print the car_pos and radius of curvature 
    if car_pos > img_size[0]/2:
        cv2.putText(result, 'The Vehicle is {:.2f}m left of center'.format(bias_from_center), (100, 80),
            fontFace=10, fontScale=1, color=(255, 255, 255), thickness=2)
    else:
        cv2.putText(result, 'The Vehicle is {:.2f}m right of center'.format(bias_from_center), (100, 80),
            fontFace=10, fontScale=1, color=(255, 255, 255), thickness=2)

    # Print radius of curvature on video
    cv2.putText(result, 'Radius of Curvature {}(ms)'.format(int((Left.radius + Right.radius) / 2)), (140, 160),
        fontFace=10, fontScale=1, color=(255, 255, 255), thickness=2)

    Left.count += 1

    return result