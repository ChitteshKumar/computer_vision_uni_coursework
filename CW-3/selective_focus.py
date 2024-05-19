import numpy as np
import cv2
import sys
import disparity

# Function to get the disparity map based on Canny edge images
def getDisparityMapFromCanny(imL, imR, canny_vals, numDisparities, blockSize):
    canny_images = [cv2.Canny(im, canny_vals[0], canny_vals[1]) for im in [imL, imR]]
    disparity_img = disparity.getDisparityMap(canny_images[0], canny_images[1], numDisparities, blockSize)
    disparity_img = (disparity_img - disparity_img.min()) / (disparity_img.max() - disparity_img.min()) * 255
    disparity_img = disparity_img.astype(np.uint8)
    return disparity_img

# Function to update disparity map based on Canny edge detection parameters
def updateDisparityMapFromCanny(canny_vals, numDisparities, blockSize):
    global imgL, imgR
    disparity_img = getDisparityMapFromCanny(imgL, imgR, canny_vals, numDisparities, blockSize)
    cv2.imshow('Disparity', disparity_img)

# Trackbar callback functions for adjusting Canny edge detection parameters
def changeCanny1(x):
    canny_vals[0] = x
    updateDisparityMapFromCanny(canny_vals, numDisparities, blockSize)

def changeCanny2(x):
    canny_vals[1] = x
    updateDisparityMapFromCanny(canny_vals, numDisparities, blockSize)

def changeNumDisparities(x):
    global numDisparities
    numDisparities = x
    updateDisparityMapFromCanny(canny_vals, numDisparities, blockSize)

def changeBlockSize(x):
    global blockSize
    blockSize = x
    updateDisparityMapFromCanny(canny_vals, numDisparities, blockSize)

# # Function to calculate depth image
# def calculateDepthImage(disparity_img, k):
#     # Invert disparity (closer = higher value)
#     depth_img = 255 - disparity_img
#     # Apply depth formula (without scale)
#     depth_img = np.reciprocal(depth_img + k)
#     # Normalize depth to 0-255 range
#     depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min()) * 255
#     depth_img = depth_img.astype(np.uint8)
#     return depth_img

# Function to calculate depth image
def calculateDepthImage(disparity_img, k):
    with np.errstate(divide='ignore'):
        depth_img = 1.0 / (disparity_img + k)
        depth_img[disparity_img==0] = 0

    norm_depth_img =  cv2.normalize(depth_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return norm_depth_img.astype(np.uint8)

def applySelectiveFocus(original_image, depth_image):
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    # Simple threshold to create a mask for the background
    _, background_mask = cv2.threshold(depth_image, 120, 255, cv2.THRESH_BINARY_INV)
    foreground_mask = cv2.bitwise_not(background_mask)

    # Create background, could be blurred or changed to grayscale
    blurred_background = cv2.medianBlur(original_image, 21)  # Using median blur instead of Gaussian
    grayscale_background = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    grayscale_background = cv2.cvtColor(grayscale_background, cv2.COLOR_GRAY2BGR)
    # Apply mask
    foreground = cv2.bitwise_and(original_image, original_image, mask=foreground_mask)
    background = cv2.bitwise_and(grayscale_background, grayscale_background, mask=background_mask)

    # Combine the focused foreground with the modified background
    selectively_focused_img = cv2.add(foreground, background)
    return selectively_focused_img

# Function to update and display disparity and depth images
def update(k):
    global imgL, imgR, numDisparities, blockSize, canny_vals
    disparity_img = getDisparityMapFromCanny(imgL, imgR, canny_vals, numDisparities, blockSize)
    depth_img = calculateDepthImage(disparity_img, k)
    focused_image = applySelectiveFocus(cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR), depth_img)

    cv2.imshow('Selective Focus', focused_image)

def changeK(x):
    global k
    k = x
    update(k)

# Main function
if __name__ == '__main__':
    # Load left and right images
    imgL = cv2.imread('girlL.png', cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread('girlR.png', cv2.IMREAD_GRAYSCALE)
    if imgL is None or imgR is None:
        print('\nError: failed to open one or both images.\n')
        sys.exit()
    
    canny_vals = [90, 158]
    numDisparities = 64
    blockSize = 5

    # Create a window to display the disparity map
    cv2.namedWindow('Selective Focus', cv2.WINDOW_NORMAL)
    update(1)

    # Create trackbars for Canny edge detection parameters
    cv2.createTrackbar('Canny1', 'Selective Focus', canny_vals[0], 255, changeCanny1)
    cv2.createTrackbar('Canny2', 'Selective Focus', canny_vals[1], 255, changeCanny2)
    cv2.createTrackbar('Number of disparities', 'Selective Focus', numDisparities, 255, changeNumDisparities)
    cv2.createTrackbar('Block Size', 'Selective Focus', blockSize, 255, changeBlockSize)
    # cv2.createTrackbar('k (Depth Scale)', 'Selective Focus', 1, 255, lambda x: update(x))
    cv2.createTrackbar('k (Depth Scale)', 'Selective Focus', 1, 255, changeK)

    # Display initial disparity map
    # updateDisparityMapFromCanny(canny_vals, numDisparities, blockSize)

    # Wait for key press to exit
    while True:
        key = cv2.waitKey(1)
        if key == ord(' ') or key == 27:  # Spacebar or Esc key
            break

    cv2.destroyAllWindows()
