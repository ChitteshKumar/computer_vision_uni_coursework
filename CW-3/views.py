import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import disparity
import sys

# Camera parameters
baseline = 174.019  # in mm
doffs = 114.291  # x-difference of principal points
cam0 = [(5806.559, 0, 1429.219), (0, 5806.559, 993.403), (0, 0, 1)]
width, height = 740, 505  # Width and height of the resized images

# Function to compute the disparity map from canny edge images
def getDisparityMapFromCanny(imL, imR, canny_vals, numDisparities, blockSize):
    canny_images = [cv2.Canny(im, canny_vals[0], canny_vals[1]) for im in [imL, imR]]
    disparity_img = disparity.getDisparityMap(canny_images[0], canny_images[1], numDisparities, blockSize)
    disparity_img = (disparity_img - disparity_img.min()) / (disparity_img.max() - disparity_img.min()) * 255
    disparity_img = disparity_img.astype(np.uint8)
    return disparity_img

# Main execution
if __name__ == '__main__':
    # Load left and right images and ensure they are the correct size
    imgL = cv2.imread('umbrellaL.png', cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread('umbrellaR.png', cv2.IMREAD_GRAYSCALE)
    if imgL is None or imgR is None:
        print('\nError: failed to open one or both images.\n')
        sys.exit()

    # Resize images if necessary
    imgL = cv2.resize(imgL, (width, height))
    imgR = cv2.resize(imgR, (width, height))

    # Initialize Canny edge detection and StereoBM parameters
    canny_vals = [90, 158]
    numDisparities = 64
    blockSize = 5

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

    # Create figures for visualization
    plt.figure()
    plt.scatter(X_world[mask], Y_world[mask], c=Z[mask], cmap='viridis')
    plt.title('Top View of the Scene')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.colorbar(label='Depth (mm)')
    plt.savefig('top_view.png')
    plt.show()

    plt.figure()
    plt.scatter(X_world[mask], Z[mask], c=Z[mask], cmap='viridis')
    plt.title('Side View of the Scene')
    plt.xlabel('X (mm)')
    plt.ylabel('Depth (mm)')
    plt.colorbar(label='Depth (mm)')
    plt.savefig('side_view.png')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_world[mask], Y_world[mask], Z[mask], c=Z[mask], cmap='viridis', marker='.')
    ax.set_title('3D View of the Scene')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Depth (mm)')
    plt.savefig('3d_view.png')
    plt.show()
