import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial.distance import cdist

def normalization(array):
    min = array.min()
    max = array.max()
    range = max-min
    return (255*(array - min) / range).astype('uint8')

def nonMaximumSuppression(response, window_size,thresh):
    
    p = window_size//2
    padded = cv2.copyMakeBorder(response, p, p, p, p, borderType=cv2.BORDER_REFLECT)
        
    height, width = response.shape
    local_maxima = np.zeros((height,width))

    for i in range(height):
        for j in range(width):
            if response[i,j] >= thresh and response[i,j] >= np.max(padded[i:i+window_size,j:j+window_size]):
                local_maxima[i, j] = response[i, j]
    
    return local_maxima


# function to detect harris corner with orientation 
def harrisCornerDetection(image, sigma=1.7, k=0.06, thresh = 0.01):

    image = ndimage.gaussian_filter(image, sigma=sigma, mode='reflect')
    image = image.astype('float32')

    #gradients 
    Ix = ndimage.sobel(image, axis=0, mode='reflect')
    Iy = ndimage.sobel(image, axis=1, mode='reflect')

    #normalization
    Ix = normalization(Ix)
    Iy = normalization(Iy)

    Ixx = ndimage.gaussian_filter(Ix*Ix, sigma=sigma, mode='reflect')
    Ixy = ndimage.gaussian_filter(Ix*Iy, sigma=sigma, mode='reflect')
    Iyy = ndimage.gaussian_filter(Iy*Iy, sigma=sigma, mode='reflect')

    orientation = np.arctan2(Iy , Ix) / (180/np.pi)

    #constructing response matrix for each pixel
    height, width = image.shape
    response = np.zeros((height,width))
    M = np.zeros((height,width,2,2))
    for i in range(height):
        for j in range(width):
            M[i,j] = [[Ixx[i,j], Ixy[i,j]], [Ixy[i,j], Iyy[i,j]]]
            detM = M[i,j,0,0] * M[i,j,1,1] - M[i,j,0,1] * M[i,j,1,0]   
            traceM = M[i,j,0,0] + M[i,j,1,1] 
            response[i,j] = detM - k * (traceM**2)

    
    local_maxima = ndimage.maximum_filter(response, size=7)
    
    interest_points = []
    Rheight, Rwidth = local_maxima.shape
    for i in range(Rheight):
        for j in range(Rwidth):
            if  local_maxima[i,j] == response[i,j] and local_maxima[i,j] > thresh*response.max():
                interest_points.append(cv2.KeyPoint(j,i,7,orientation[i,j]))
    return response, interest_points


def cornerDetectionWithORB(image):
    #ORB for harris 
    orb_harris = cv2.ORB_create(scoreType = cv2.ORB_HARRIS_SCORE) 
    harris_keypoints = orb_harris.detect(image, None)
    harris_keypoints, harris_descriptor = orb_harris.compute(image, harris_keypoints )
    harris_result_keypoints = cv2.drawKeypoints(image, harris_keypoints, None, color=(0,255,0), flags=0)

    #ORB for fast 
    orb_fast = cv2.ORB_create(scoreType = cv2.ORB_HARRIS_SCORE) 
    fast_keypoints = orb_fast.detect(image, None)
    fast_keypoints, fast_descriptor = orb_harris.compute(image, fast_keypoints )
    fast_result_keypoints = cv2.drawKeypoints(image, fast_keypoints, None, color=(0,255,0), flags=0)
    
    cv2.imwrite(f"OUTPUT/ORB_Harris_KeyPoints.jpg", harris_result_keypoints)
    cv2.imwrite(f"OUTPUT/ORB_Fast_KeyPoints.jpg", fast_result_keypoints)
    print("NUMBER OF KEY", len(harris_keypoints) )

    return  harris_keypoints, harris_descriptor, fast_keypoints, fast_descriptor

def matchDescriptorsSSD(descriptor1, descriptor2):
    distances = cdist(descriptor1, descriptor2, metric='sqeuclidean')
    dmatch = []
    for match_idx in range(len(distances)):
        min_idx  = np.argmin(distances[match_idx])
        dmatch.append(cv2.DMatch(match_idx, min_idx, distances[match_idx][min_idx]))
    return dmatch

def matchDescriptorsRatio(descriptor1, descriptor2):
    distances = cdist(descriptor1, descriptor2, metric='sqeuclidean')
    nn1 = np.argsort(distances, axis=1)[:, 0]
    nn2 = np.argsort(distances, axis=1)[:, 1]

    # Apply the ratio test
    matches = []
    for i in range(len(descriptor1)):
        if distances[i, nn1[i]] / distances[i, nn2[i]] < 0.7:
            match = cv2.DMatch(i, nn1[i], distances[i, nn1[i]])
            matches.append(match)

    return matches

#Image visualisation
def draw_corners_on_image(image, corners):
    # Make a copy of the original image to draw on
    image_with_corners = np.copy(image)
    for corner in corners:
        x, y = corner[1], corner[0]  # Swap the order because OpenCV uses (x, y) format
        # Draw a circle at each corner location (red circle with small radius)
        cv2.circle(image_with_corners, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
    return image_with_corners

#load image
bernie_image = cv2.imread('bernieSanders.jpg')
gray_bernie_image = cv2.cvtColor(bernie_image, cv2.COLOR_BGR2GRAY)

#WITH ORB 
ref_keypoints_with_orb_harris, ref_descriptor_with_orb_harris, ref_keypoints_with_orb_fast, ref_descriptor_with_orb_fast = cornerDetectionWithORB(gray_bernie_image)

#WITH HARRIS CORNER
orb = cv2.ORB_create()

response, ref_keypoints = harrisCornerDetection(gray_bernie_image)
print("NUMBER OF KEY POINTS", len(ref_keypoints))
ref_keypoints, ref_descriptors = orb.compute(gray_bernie_image, ref_keypoints )
result_keypoints = cv2.drawKeypoints(gray_bernie_image, ref_keypoints, None, color=(0,255,0), flags=0)
# cv2.imshow("Testing Image KEYPOINTS", result_keypoints)
# cv2.imwrite(f"OUTPUT/REF_keypoints.jpg", result_keypoints)



# other_images = [
#     "bernie180.jpg",                    #0
#     "bernieBenefitBeautySalon.jpeg",    #1
#     "BernieFriends.png",                #2
#     "bernieMoreblurred.jpg",            #3
#     "bernieNoisy2.png",                 #4
#     "berniePixelated2.png",             #5
#     "darkerBernie.jpg" ,            #6
#     "brighterBernie.jpg",               #7
#     "bernieShoolLunch.jpeg"                 #8
# ]
other_images = [
    "bernieShoolLunch.jpeg"   ,
                  'bernieSanders.jpg'              #8
]

for other_image_path in other_images:
    testing_image = cv2.imread(other_image_path)
    gray_testing_image = cv2.cvtColor(testing_image, cv2.COLOR_BGR2GRAY)

    #WITH ORB 
    test_keypoints_with_orb_harris, test_descriptor_with_orb_harris, test_keypoints_with_orb_fast, test_descriptor_with_orb_fast = cornerDetectionWithORB(gray_testing_image)

    #WITH HARRIS CORNER
    test_response, test_harris_keypoints = harrisCornerDetection(gray_testing_image)
    test_harris_keypoints, test_descriptors = orb.compute(gray_testing_image, test_harris_keypoints )
    result_test_keypoints = cv2.drawKeypoints(gray_testing_image, test_harris_keypoints, None, color=(0,255,0), flags=0)
    cv2.imwrite(f"OUTPUT/schoollunch_keypoints.jpg", result_test_keypoints)
    # cv2.imshow("Testing Image KEYPOINTS", result_test_keypoints)


    ssd_match = matchDescriptorsSSD(ref_descriptors, test_descriptors)
    ssd_match = sorted(ssd_match, key=lambda x: x.distance)
    gray_ssd_match = cv2.drawMatches(gray_bernie_image, ref_keypoints, gray_testing_image, test_harris_keypoints, ssd_match[:80], outImg=None, flags=2)
    cv2.imwrite(f"OUTPUT/TEST/{other_image_path}_harris_match.jpg", gray_ssd_match)
    



cv2.waitKey(0)
cv2.destroyAllWindows()
