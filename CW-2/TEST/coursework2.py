import cv2
import numpy as np 
import scipy.ndimage
import os 
 
# Task 1
def harris_points_detector(image, sigma = 0.5, k=0.059, window_size = 7, threshold = 0.01):
    """
    Detects Harris corner points with the specified order of operations.

    Args:
        image (numpy.ndarray): The input grayscale image.
        window_size (int): The size of the window for calculating Harris matrix.
        k (float): Harris corner constant.
        threshold (float): Threshold for filtering interest points.

    Returns:
        list: A list of cv2.KeyPoint objects for detected Harris interest points.
    """    
    
    # converting the image to float32 to avoid overflow
    image = image.astype('float32')
    
    # blurring the image to get better results - less number of unneccessary points 
    image = scipy.ndimage.gaussian_filter(image, sigma)
    
    # 1. Computing image gradients
    dx = scipy.ndimage.sobel(image, axis=0, mode='reflect')
    dy = scipy.ndimage.sobel(image, axis=1, mode='reflect')

    
    # Calculating the orientation
    orientations = np.degrees(np.arctan2(dx, dy))
    
    # Calculating Ix^2, Iy^2 and Ixy
    Ixx = dx**2
    Iyy = dy**2
    Ixy = dx*dy

    # 2. Compute Harris matrix components
    Ixx = scipy.ndimage.gaussian_filter(Ixx, sigma, mode='reflect')
    Iyy = scipy.ndimage.gaussian_filter(Iyy, sigma, mode='reflect')
    Ixy = scipy.ndimage.gaussian_filter(Ixy, sigma, mode='reflect')
    

    # 3. Compute corner strength (Response)
    detM = Ixx * Iyy - Ixy**2
    traceM = Ixx + Iyy
    R = detM - k * traceM**2
    
    # 4. Non-maximum suppression 
    R_nms = non_maximum_suppression(R, window_size, threshold*R.max())
    
    
    keypoints = []
    rx, ry = R_nms.shape
    for i in range(0, rx):
        for j in range(0, ry):
            if R_nms[i, j] > 0:
                keypoints.append(cv2.KeyPoint(j, i, window_size, orientations[i, j]))    
    
    
  
    cv2.waitKey(0)
    return R, R_nms, keypoints


def non_maximum_suppression(harris_response, window_size,threshold):
    """Performs basic non-maximum suppression on detected keypoints""" 
    p_window_size = window_size//2
    # Padding the    matrix to deal with edges and corners
    padded = cv2.copyMakeBorder(harris_response, p_window_size, p_window_size, p_window_size, p_window_size, borderType=cv2.BORDER_REFLECT)
        
    dim_x, dim_y = harris_response.shape
    suppressed_keypoints = np.zeros((dim_x,dim_y))

    for i in range(0, dim_x):
        for j in range(0, dim_y):
            if harris_response[i,j] >= threshold and harris_response[i,j] >= np.max(padded[i:i+window_size,j:j+window_size]):
                suppressed_keypoints[i, j] = harris_response[i, j]
        

    return suppressed_keypoints

# Task 2
def compute_orb_keypoints(image):
    orb = cv2.ORB_create()
    keypoints = orb.detect(image, None)
    return keypoints

def compute_orb_descriptors(image, keypoints):
    orb = cv2.ORB_create()
    _, descriptors = orb.compute(image, keypoints)
    return descriptors


def display_keypoints_count(harris_keypoints, orb_keypoints):
    # Print the number of keypoints and descriptors
    print(f"Harris Keypoints: {len(harris_keypoints)}")
    print(f"ORB Keypoints: {len(orb_keypoints)}")

# Task 3
def ssd_feature_matcher(descriptors1, descriptors2):
    """
    Matches features using Sum of Squared Differences (SSD).

    Args:
        descriptors1 (numpy.ndarray): Descriptors of the first image.
        descriptors2 (numpy.ndarray): Descriptors of the second image.

    Returns:
        list: A list of cv2.DMatch objects.
    """
    distances = scipy.spatial.distance.cdist(descriptors1, descriptors2, 'sqeuclidean')
    matches = []
    for i, row in enumerate(distances):
        match_idx = np.argmin(row)
        matches.append(cv2.DMatch(i, match_idx, row[match_idx]))

    return matches

def ratio_feature_matcher(descriptors1, descriptors2, ratio_threshold=0.75):
    """
    Matches features using the ratio test.

    Args:
        descriptors1 (numpy.ndarray): Descriptors of the first image.
        descriptors2 (numpy.ndarray): Descriptors of the second image.
        ratio_threshold (float): Threshold for the ratio test.

    Returns:
        list: A list of cv2.DMatch objects.
    """
    distances = scipy.spatial.distance.cdist(descriptors1, descriptors2, 'sqeuclidean')
    matches = []
    for i, row in enumerate(distances):
        sorted_indices = np.argsort(row)
        best_match, second_best_match = sorted_indices[:2]
        if row[best_match] / row[second_best_match] < ratio_threshold:
            matches.append(cv2.DMatch(i, best_match, row[best_match]))

    return matches

def visualize_matches(image1, keypoints1, image2, keypoints2, matches, display=True, save_path=None):
    """
    Visualizes feature matches using OpenCV's drawMatches function.

    Args:
        image1 (numpy.ndarray): The first image.
        keypoints1 (list): Keypoints in the first image.
        image2 (numpy.ndarray): The second image.
        keypoints2 (list): Keypoints in the second image.
        matches (list): A list of cv2.DMatch objects.
        display (bool): If True, displays the results using cv2.imshow.
        save_path (str): Optional path to save the visualized image.
    """
    result_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, outImg=None, flags=2)

    if display:
        cv2.imshow("Feature Matches", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save_path:
        cv2.imwrite(save_path, result_img)

    return result_img


    
def load_image(image_path):
    return cv2.imread(image_path)

# def display_images():
    
    


def resize_image(image, max_height=800, max_width=1000):
    height, width = image.shape[:2] 
    scale_height = max_height / height
    scale_width = max_width / width

    scale_factor = min(scale_height, scale_width)  # Choose the smaller scale factor 

    if scale_factor < 1: 
        new_size = (int(width * scale_factor), int(height * scale_factor))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    else:
        return image 
    
    
def get_filename(filepath, type):
    if type == "harris":
        path = os.path.splitext(filepath)[0]
        return path + "_ssd_matches.jpg", path + "_ratio_matches.jpg"
    elif type == "orb":
        path = os.path.splitext(filepath)[0]
        return path + "_orb_ssd_matches.jpg", path + "_orb_ratio_matches.jpg"
    
    





# ------------------------------------------MAIN------------------------------------------------------------------------------------------

bernie_img = load_image('bernieSanders.jpg')
bernie_gray = cv2.cvtColor(bernie_img, cv2.COLOR_BGR2GRAY)

# Compute Harris keypoints and ORB descriptors
R, R_nms, harris_keypoints = harris_points_detector(bernie_gray)
harris_descriptors = compute_orb_descriptors(bernie_gray, harris_keypoints)

# Compute ORB keypoints and descriptors
orb_keypoints = compute_orb_keypoints(bernie_gray)
orb_descriptors = compute_orb_descriptors(bernie_gray, orb_keypoints)

# printing number of keypoints
display_keypoints_count(harris_keypoints, orb_keypoints)
# plotting the keypoints from both techniques on the original image 
keypoints_harris = cv2.drawKeypoints(bernie_img, harris_keypoints, 0, (0, 0, 255))
keypoints_orb = cv2.drawKeypoints(bernie_img, orb_keypoints, 0, (0, 255, 0))

# Visualize results
# cv2.imshow("Original Image", resize_image(bernie_img))
cv2.imshow("Response", resize_image(R))
cv2.imshow("Response after NMS", resize_image(R_nms))

comparison_harris_orb = np.concatenate((keypoints_harris, keypoints_orb), axis = 1)
cv2.imwrite("Harris keypoints VS Orb keypoints.jpg", comparison_harris_orb)

test_images = [
    # "bernie180.jpg",                    #0
    # "bernieBenefitBeautySalon.jpeg",    #1
    "BernieFriends.png",                #2
    # "bernieMoreblurred.jpg",            #3
    # "bernieNoisy2.png",                 #4
    # "berniePixelated2.png",             #5
    # "bernieShoolLunch.jpeg",            #6
    # "brighterBernie.jpg",               #7
    # "darkerBernie.jpg"                  #8
]


for test_image in test_images:
    test = load_image(test_image)
    test_grey = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    _, _ , test_keypoints = harris_points_detector(test_grey)
    test_descriptors = compute_orb_descriptors(test_grey, test_keypoints)
    ssd_matches = ssd_feature_matcher(harris_descriptors, test_descriptors)
    ratio_matches = ratio_feature_matcher(harris_descriptors, test_descriptors)
    filename_ssd, filename_ratio = get_filename(test_image, type = "harris")
    visualize_matches(bernie_gray, harris_keypoints, test_grey, test_keypoints, ssd_matches, display=True, save_path=filename_ssd)
    visualize_matches(bernie_gray, harris_keypoints, test_grey, test_keypoints, ratio_matches, display=True, save_path=filename_ratio)

for test_image in test_images:
    test = load_image(test_image)
    test_grey = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    test_keypoints = compute_orb_keypoints(test_grey)
    test_descriptors = compute_orb_descriptors(test_grey, test_keypoints)
    ssd_matches = ssd_feature_matcher(orb_descriptors, test_descriptors)
    ratio_matches = ratio_feature_matcher(orb_descriptors, test_descriptors)
    filename_ssd, filename_ratio = get_filename(test_image, type = "orb")
    visualize_matches(bernie_gray, harris_keypoints, test_grey, test_keypoints, ssd_matches, display=True, save_path=filename_ssd)
    visualize_matches(bernie_gray, harris_keypoints, test_grey, test_keypoints, ratio_matches, display=True, save_path=filename_ratio)

cv2.waitKey(0)

