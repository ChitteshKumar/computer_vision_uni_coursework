import numpy as np
import cv2
import sys
import disparity
import matplotlib.pyplot as plt

cam0 = [(5806.559, 0, 1429.219),
        (0, 5806.559, 993.403),
        (0, 0, 1)]
cam1 = [(5806.559, 0, 1543.51),
        (0, 5806.559, 993.403),
        (0, 0, 1)]
sensor = [22.2, 14.8]
resolution = [3088, 2056]
doffs = 114.291
baseline = 174.019
# width = 2960
# height = 2016
width = 740
height = 505

# Function to get the disparity map based on Canny edge images
def getDisparityMapFromCanny(imL, imR, canny_vals, numDisparities, blockSize):
    canny_images = [cv2.Canny(im, canny_vals[0], canny_vals[1]) for im in [imL, imR]]
    disparity_img = disparity.getDisparityMap(canny_images[0], canny_images[1], numDisparities, blockSize)
    disparity_img = (disparity_img - disparity_img.min()) / (disparity_img.max() - disparity_img.min()) * 255
    disparity_img = disparity_img.astype(np.uint8)
    cv2.imwrite(f'output/Disparity.png', disparity_img)
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

# Main function
if __name__ == '__main__':


    # FOCAL LENGTH CALCULATION
    # focal length(mm) = focal length (px) / (sensor width (mm) x image width (px))
    focal_length = cam0[0][0] * (sensor[0] / resolution[0])
    print(f"focal length "
        f"= focal length ({cam0[0][0]}px) * (sensor width ({sensor[0]}mm) / image width ({resolution[0]}mm))"
        f"\nf={focal_length}mm")


    # Load left and right images
    imgL = cv2.imread('umbrellaL.png', cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread('umbrellaR.png', cv2.IMREAD_GRAYSCALE)
    if imgL is None or imgR is None:
        print('\nError: failed to open one or both images.\n')
        sys.exit()

    # Create a window to display the disparity map
    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)

    # Initialize Canny edge detection parameters
    canny_vals = [90, 158]

    # Initialize StereoBM parameters
    numDisparities = 64
    blockSize = 5

    # Create trackbars for Canny edge detection parameters
    cv2.createTrackbar('Canny1', 'Disparity', canny_vals[0], 255, changeCanny1)
    cv2.createTrackbar('Canny2', 'Disparity', canny_vals[1], 255, changeCanny2)
    cv2.createTrackbar('Number of disparities', 'Disparity', numDisparities, 255, changeNumDisparities)
    cv2.createTrackbar('Block Size', 'Disparity', blockSize, 255, changeBlockSize)

    # Display initial disparity map
    updateDisparityMapFromCanny(canny_vals, numDisparities, blockSize)


    #Task1.3:
    # Camera parameters
    baseline = 174.019  # in mm
    doffs = 114.291  # x-difference of principal points
    cam0 = [(5806.559, 0, 1429.219), (0, 5806.559, 993.403), (0, 0, 1)]
    width, height = 740, 505  # Width and height of the resized images
    imgL = cv2.resize(imgL, (width, height))
    imgR = cv2.resize(imgR, (width, height))

    # Compute the disparity map
    disparity_img = getDisparityMapFromCanny(imgL, imgR, canny_vals, numDisparities, blockSize)

    # Calculate the depth (Z) for each pixel
    disparity_img_float = disparity_img.astype(np.float32)
    Z = baseline * cam0[0][0] / (disparity_img_float + doffs)

    # Generate meshgrid for pixel coordinates
    X = np.tile(np.arange(width), (height, 1))
    Y = np.tile(np.arange(height), (width, 1)).T

    # Convert pixel coordinates to world coordinates
    X_world = (X - cam0[0][2]) * Z / cam0[0][0]
    Y_world = (Y - cam0[1][2]) * Z / cam0[0][0]

    # Mask to remove infinite and NaN values for visualization
    mask = np.isfinite(Z) & (Z > 0)

    disparity.plot(X_world)

    # Create figures for visualization
    plt.figure()
    plt.scatter(X_world[mask], Y_world[mask], c=Z[mask], cmap='viridis')
    plt.title('Top View of the Scene')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.colorbar(label='Depth (mm)')
    plt.savefig('output/top_view.png')
    plt.show()

    plt.figure()
    plt.scatter(X_world[mask], Z[mask], c=Z[mask], cmap='viridis')
    plt.title('Side View of the Scene')
    plt.xlabel('X (mm)')
    plt.ylabel('Depth (mm)')
    plt.colorbar(label='Depth (mm)')
    plt.savefig('output/side_view.png')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_world[mask], Y_world[mask], Z[mask], c=Z[mask], cmap='viridis', marker='.')
    ax.set_title('3D View of the Scene')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Depth (mm)')
    plt.savefig('output/3d_view.png')
    plt.show()

    # Wait for key press to exit
    while True:
        key = cv2.waitKey(1)
        if key == ord(' ') or key == 27:  # Spacebar or Esc key
            break

    cv2.destroyAllWindows()
