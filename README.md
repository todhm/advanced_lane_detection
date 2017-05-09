## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


[image1]: ./model_structure.001.png
[image2]: ./output_images/undistort_img.png "Before undistort image based on camera calibration"
[image3]: ./output_images/img_threshold.png "Before flipping"
[image4]: ./output_images/src_dst.png "Before flipping"
[image5]: ./output_images/warp_transformation.png "Before flipping"
[image6]: ./output_images/centroid_img.png "Shadowed image"
[image7]: ./output_images/final_transform.png "Shadowed image"
[image8]: ./output_images/video.png "Shadowed image"
[image9]: ./output_images/augment_img.png "Shadowed image"

---
In this project I will build a lane detecting pipeline to detect lane on the load. 

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use  gradients, direction of gradient,color space conversion to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels by using linear convolution and fit the line based on x and y points.
* Determine the curvature of the lane and vehicle position with respect to center and difference between centroid points. 
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.



Camera Calibration
---
* To correct the distortion of camera caused by 

* First With the Chess board image from various angle we calculate the camera matrix and distortion coefficient. 
    * Calculating camera matrix and distortion coefficient of camera consists of two seperate steps. 
    * (1) First Detect corners on each chessboard from image. 
    * (2) Considering detected corners and 3D space of the objects in image calculate camera matrix & distortion coefficient. 
    
* Based on camera matrix & distortion coefficient from previous steps make undistorted image. 

![alt text][image2]

Augment Data
---
* To make more robust lane detecting on various condition adjust brightness and shadows to make more road images. 
![alt text][image9]

Threshold Detection.
---
* Based on gradient & color space conversion technique we convert undistorted image to binary image. 
* There are 4 kind of thresh-holding techniques to detect line on the road were used. 
    * (1) Calculate the derivative of image in horizontal or vertical  direction and find the area of image that meet certain threshold of color threshold.
    * (2) Calculate overall magnitude of gradient and find the area that meet certain color threshold. 
    * (3) Calculate the direction of the gradient by arctangent of ygradient and xgradient and find the area that meet certain color threshold. 
    * (4) Convert color space of image from RBG to HLS and extract saturation channel of image. Find the area of image that meet certain color threshold. 
    
* By combining area from image meet (1) Condition or ((2) and (3)) condition  or (4) Condition return binary image. 

![alt text][image3]

Perspective transform. 
---
* After we have binary image that detect lane part from image transform the view point of image from the sky(bird-eye view) so the lanes are seem parallel like real world.  
* This steps are also divided into 2 steps
    * (1) Defining source points and destination point where we want to transform. 
    ![alt text][image4]
    
    * (2) Using cv2.warpPerspective function convert image sourcepoint to destination point. 
    ![alt text][image5]

Locate lane lines and fit polynomial by sliding window from image. 
---
* (1) Find the Base line x point by finding strongest signed on binary warp image. 
* (2) Divide the image area by pre-defined window size. 
* (3) As we slide upward Find the points located inside the pre-defind margin. 

![alt text][image6]


Determine the curvature of the lane and vehicle position with respect to center.
---
* By using fit of the polynomial from the previous step we calculate radius of curvature and vehicle position with respect to center. 
* I could verify that if the lane lines were straight lane the radius of curvature of left and right line  show great difference although lanes were detected correctly. On the other hand if the original lane were curved lane lines radius of curvature show smaller difference if it detected correctly. 

Restore Original image with the area inside the lane are colored. 
---
* From the Functions from previous steps construct final pipeline
* When the lane detection didn't meet two specific condition we using average of former 10 lines in beforehand. 
    * (1) When radius of curvature of new lane is not greater than 100 times former lane line and also not smaller than 0.01 times former lane. . 
    * (2) When maximum distance between pixel value of base x points less than 10.
![alt text][image7]


Final Video. 
---
[![alt text][image8]](https://youtu.be/9cCetfGwfew)


Final Discussion. 
---
* It took much time to finish off this project. 
* The main Problem I had was to use to small x and y point to fit the line. 
* The sanity check also have crucial role to finish off this project.