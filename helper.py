import os
import numpy as np
import cv2
import matplotlib.pylab as plt
import matplotlib
import matplotlib.image as mpimg
import pickle



def calculate_corners(cal_direction):
    img_lst = os.listdir(cal_direction)
    images = [ cv2.imread(cal_direction + x )for x in img_lst]
    objpoints = []   # List to save Grids  in 3D spaces.
    imgpoints = []   # List to save coordinates of corners.
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2) #grid of objectpoints coordinate making z coordinate 0


    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret,corners = cv2.findChessboardCorners(gray, (9,6), None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
    return objpoints,imgpoints

def undistort_img(objpoints,imgpoints,img):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img[0].shape[-2::-1],None,None)
    undistorted_road = cv2.undistort(road_images[1],mtx,dist,None,mtx)
    return undistorted_road


def perspective_transform(undistorted,src,dst,mtx,dist):
    img_size  = (undistorted.shape[1], undistorted.shape[0])
    M = cv2.getPerspectiveTransform(src,dst)
    warped = cv2.warpPerspective(undistorted,M,img_size,flags = cv2.INTER_LINEAR)
    return warped, M




def pipeline(img, s_thresh=(80, 255), sx_thresh=(10, 100),kernel_size =5):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0,kernel_size) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    combined_binary = np.zeros_like(sxbinary)
    #combined_binary[s_binary ==1] = 1
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary


def calculate_curvature(combined,left_fit,right_fit):

    ploty = np.linspace(0, combined.shape[0]-1, combined.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    y_eval = np.max(ploty)
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    return left_curvread, right_curvread
def find_window_centroids(combined, window_width, window_height, margin):

    window_centroids = []
    y_points = []
    window = np.ones(window_width) # Create our window template that we will use for convolutions

    #Get the initial centroid by using 1/4 part of image.
    left_combined = np.zeros_like(combined)
    right_combined = np.zeros_like(combined)
    left_combined[int(3*combined.shape[0]/4):,:int(combined.shape[1]/2)] = 1
    right_combined[int(3*combined.shape[0]/4):,int(combined.shape[1]/2):] = 1

    l_sum = np.sum(combined[int(3*combined.shape[0]/4):,:int(combined.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(combined[int(3*combined.shape[0]/4):,int(combined.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(combined.shape[1]/2)

    left_y = np.argwhere(left_combined[:,int(l_center)])
    left_y = left_y[int(len(left_y)/2)][0]
    right_y = np.argwhere(right_combined[:,int(r_center)])
    right_y = right_y[int(len(right_y)/2)][0]
    window_centroids.append((l_center,r_center))
    y_points.append((left_y,right_y))
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(combined.shape[0]/window_height)):
        empty_combined = np.zeros_like(combined)
        empty_combined[int(combined.shape[0]-(level+1)*window_height):int(combined.shape[0]-level*window_height),:] = 1

        # convolve the window into the vertical slice of the image
        image_layer = np.sum(combined[int(combined.shape[0]-(level+1)*window_height):int(combined.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,combined.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset

        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,combined.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer


        left_y = np.argwhere(empty_combined[:,int(l_center)])
        left_y = left_y[int(len(left_y)/2)][0]
        right_y = np.argwhere(empty_combined[:,int(r_center)])
        right_y = right_y[int(len(right_y)/2)][0]
        window_centroids.append((l_center,r_center))
        y_points.append((left_y,right_y))

    return window_centroids,y_points


def draw_transformed_img(window_centroids,y_points,src,dst,undistorted_road):
    left_x = np.asarray(window_centroids)[:,0]
    right_x = np.asarray(window_centroids)[:,1]
    left_y = np.asarray(y_points)[:,0]
    right_y = np.asarray(y_points)[:,1]
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)
    ploty = np.linspace(0, undistorted_road.shape[0]-1, undistorted_road.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    warp_zero = np.zeros_like(undistorted_road).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undistorted_road[1].shape[1], undistorted_road[1].shape[0]))
    left_curvread,right_curvread = calculate_curvature(combined,left_fit,right_fit)
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_road, 1, newwarp, 0.3, 0)
    return result


def final_pipeline(img):
    objpoints = np.load('./objpoints.npy')
    objpoints = list(objpoints)
    imgpoints = np.load('./imgpoints.npy')
    imgpoints = list(imgpoints)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[-2::-1],None,None)
    undistorted = cv2.undistort(img,mtx,dist,None,mtx)
    top_left = [585,450]
    top_right = [690,450]
    bottom_left = [255,660]
    bottom_right = [1050,660]
    src = np.asarray([top_left,top_right,bottom_right,bottom_left])
    src = np.float32(src)
    top_left = [255,100]
    top_right = [1050,100]
    bottom_left = [255,700]
    bottom_right = [1050,700]
    dst = np.float32([top_left,top_right,bottom_right,bottom_left])
    warped,M = perspective_transform(undistorted,src,dst,mtx,dist)
    combined = pipeline(warped,kernel_size =21)
    window_centroids,y_points = find_window_centroids(combined,50,80,40)
    result = draw_transformed_img(window_centroids,y_points,src,dst,undistorted)
    return result
