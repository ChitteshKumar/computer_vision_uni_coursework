import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

#load image
kitty_image = cv2.imread('kitty.bmp')

cv2.namedWindow('Source Image')
cv2.imshow('Source Image', kitty_image)

# Check for success of opening of the image
if kitty_image is None:
    print('Error: failed to open', kitty_image)
    sys.exit()

# Convert to greyscale and save it

# #using opencv
gray_kitty_image = cv2.cvtColor(kitty_image, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('Grayscaled Image')
cv2.imshow('Grayscaled Image', gray_kitty_image)
    
#using PIL library 
# grey_image = 

#---------CONVOLUTION--------------
def convolution(image, kernel):
    height, width = image.shape

    #padding and converting to 64F
    k_height, k_width = kernel.shape
    pad_size_x = k_height // 2
    pad_size_y = k_width // 2
    padded_image = np.pad(image, ((pad_size_x,pad_size_x),(pad_size_y,pad_size_y)), mode='constant', constant_values = 0)
    padded_image = cv2.normalize(padded_image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)


    # gaussianresult = np.zeros_like(image)
    # gaussianresult = np.zeros((height, width), dtype=np.float32)
    result = np.zeros((height, width), dtype=np.float32)

    # for i in range(height):
    #     for j in range(width):  
    #         neighbours = padded_image[i:i+size, j:j+size]
    #         gaussianresult[i,j] = np.sum(neighbours * weighted_mean_kernel)

    for i in range(height):
        for j in range(width):
            value = 0.0  
            for x in range(k_height):
                for y in range(k_width):
                    value = value + padded_image[i + x, j+y] * kernel[x,y]
            result[i,j] = value
    
    return result

#--------------SIMPLE THRESHOLDING--------------
def simpleThresholding(threshold_value, image):
    thresholded_result = np.zeros_like(image).astype(np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > threshold_value:
                thresholded_result[i,j] = 255
            else:
                thresholded_result[i,j] = 0
    
    return  thresholded_result

#--------------AVERAGE SMOOTHING--------------
def averageKernel(size):
    average_kernel = np.ones((size,size), dtype = np.float32) / (size*size)
    return average_kernel

#making average kernel 
average_kernel = averageKernel(size=7)
#calling average convolved image
average_smooth_image = convolution(gray_kitty_image,average_kernel )
#to normalize from [0,255] arrays to uint8 to display image
average_smooth_image = cv2.normalize(average_smooth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


along_x = np.array([[-1,0,1],
                    [-2,0,2],
                    [-1,0,1]], dtype=np.float32)
along_y = np.array([[-1,-2,-1],
                    [0,0,0],
                    [1,2,1]], dtype=np.float32)

#horizontal and vertical gradients
avg_gradient_x_image = convolution(average_smooth_image, along_x)
avg_gradient_y_image = convolution(average_smooth_image, along_y)

#edge strength after combined image  
avg_gradient_magnitude = np.sqrt((avg_gradient_x_image.astype(np.float32) **2) + (avg_gradient_y_image.astype(np.float32) **2) )

#normalizing 
avg_gradient_magnitude = cv2.normalize(avg_gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#to normalize from [0,255] arrays to uint8 to display image
avg_gradient_x_image = cv2.normalize(avg_gradient_x_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
avg_gradient_y_image = cv2.normalize(avg_gradient_y_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)



cv2.namedWindow('Gradient Along x Image')
cv2.imshow('Gradient Along x Image', avg_gradient_x_image)
cv2.namedWindow('Gradient Along y Image')
cv2.imshow('Gradient Along y Image', avg_gradient_y_image)
cv2.namedWindow('Combined Gradient Image')
cv2.imshow('Combined Gradient Image', avg_gradient_magnitude)

#gaussian smoothed image display
cv2.namedWindow('Average Convolved Image')
cv2.imshow('Average Convolved Image', average_smooth_image)


#--------------SIMPLE THRESHOLDING --------------
threshold_value = 21 # 18,25,35,30,40,50,60
thresholded_image = simpleThresholding(threshold_value, avg_gradient_magnitude)
# Display the Thresholded Image
cv2.imshow('Simple Thresholded Image', thresholded_image)


#--------------WEIGHTED-AVERAGE (GAUSSIAN) SMOOTHING--------------
def gaussianKernel(size, sigma):
    #gaussian kernel
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = ( 1 / ( 2 * np.pi * sigma**2 )) * np.exp( - ( (x**2 + y**2) / (2* sigma**2) ) )
    # g= np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-size//2)**2 + (y-size//2)**2)/(2*sigma**2)), (size, size))
    weighted_mean_kernel = g / np.sum(g)

    return weighted_mean_kernel

#making guassian kernel 
gaussian_kernel = gaussianKernel(size=7,sigma=2.5)
#calling gaussian smoothed image
gaussian_smooth_image = convolution(gray_kitty_image, gaussian_kernel)
gaussian_smooth_image = cv2.normalize(gaussian_smooth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

#horizontal and vertical gradients
gaussian_gradient_x_image = convolution(gaussian_smooth_image, along_x)
gaussian_gradient_y_image = convolution(gaussian_smooth_image, along_y)

# #edge strength after combined image  
gaussian_gradient_magnitude = np.sqrt((gaussian_gradient_y_image.astype(np.float32) **2) + (gaussian_gradient_y_image.astype(np.float32) **2) )

#normalizing 
gaussian_gradient_x_image = cv2.normalize(gaussian_gradient_x_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
gaussian_gradient_y_image = cv2.normalize(gaussian_gradient_y_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
gaussian_gradient_magnitude = cv2.normalize(gaussian_gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# cv2.namedWindow('Gradient Along-x Gaussian Image')
cv2.imshow('Gradient Along x Gaussian Image', gaussian_gradient_x_image)
# cv2.namedWindow('Gradient Along-y Gaussian Image')
cv2.imshow('Gradient Along y Gaussian Image', gaussian_gradient_y_image)
cv2.namedWindow('Combined Gradient for Gaussian Image')
cv2.imshow('Combined Gradient for Gaussian Image', gaussian_gradient_magnitude)

#gaussian smoothed image display
cv2.namedWindow('Weighted - Average Convolved Image')
cv2.imshow('Weighted - Average Convolved Image', gaussian_smooth_image)

#--------------SIMPLE THRESHOLDING-----------------
threshold_value = 21 # 18,25,35,30,40,50,60
thresholded_gimage = simpleThresholding(threshold_value, gaussian_gradient_magnitude)
# Display the Thresholded Image
cv2.imshow('Simple Thresholded Gaussian Image', thresholded_gimage)


#--------------HISTOGRAM (Gradient magnitude) - AVERAGE SMOOTHING Vs WEIGHTED AVERAGE SMOOTHING--------------

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

axs[0].hist(avg_gradient_magnitude.ravel(), 256, [0,256], color='blue', alpha=0.7)
axs[0].set_title("Average Smoothing & Sobel")

axs[1].hist(gaussian_gradient_magnitude.ravel(), 256, [0,256], color='green', alpha=0.7)
axs[1].set_title("Weighted Average Smoothing & Sobel")

plt.show()


# Wait for spacebar press before closing,
# otherwise window will close without you seeing it
while True:
    k = cv2.waitKey(1)
    if k == ord(' ') or k == ord('\t'):
        print('Saving average_smooth_image.png')
        cv2.imwrite('average_smooth_image.png', average_smooth_image)
        print('Saving avg_gradient_x_image.png')
        cv2.imwrite('avg_gradient_x_image.png', avg_gradient_x_image)
        print('Saving avg_gradient_y_image.png')
        cv2.imwrite('avg_gradient_y_image.png', avg_gradient_y_image)
        print('Saving avg_gradient_magnitude.png')
        cv2.imwrite('avg_gradient_magnitude.png', avg_gradient_magnitude)
        #saving the graph 
        print('Saving img_mean_hist.png')
        fig.savefig("histogram.png")
        print('Saving thresholded_image.png')
        cv2.imwrite('thresholded_image.png', thresholded_image)
        print('Saving gaussian_smooth_image.png')
        cv2.imwrite('gaussian_smooth_image.png', gaussian_smooth_image)
        print('Saving gaussian_gradient_x_image.png')
        cv2.imwrite('gaussian_gradient_x_image.png', gaussian_gradient_x_image)
        print('Saving gaussian_gradient_y_image.png')
        cv2.imwrite('gaussian_gradient_y_image.png', gaussian_gradient_y_image)
        print('Saving gaussian_gradient_magnitude.png')
        cv2.imwrite('gaussian_gradient_magnitude.png', gaussian_gradient_magnitude)
        print('Saving thresholded_gaussian_image.png')
        cv2.imwrite('thresholded_gaussian_image.png', thresholded_gimage)
        # print('Saving img_edge_comparison.png')
        # cv2.imwrite('img_edge_comparison.png', img_edge_comparison)
        break

cv2.destroyAllWindows()
