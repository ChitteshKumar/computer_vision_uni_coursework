import cv2
import numpy as np
import disparity

# Camera matrix for the left camera (cam0)
cam0 = np.array([[5806.559, 0, 1429.219],
                 [0, 5806.559, 993.403],
                 [0, 0, 1]])

# Sensor size and resolution in medium resolution mode for a Canon DSLR camera (EOS 450D)
sensor_size_mm = np.array([22.2, 14.8])  # Sensor size in mm (width x height)
resolution_px = np.array([3088, 2056])   # Image resolution in pixels (width x height)

# Focal length in pixels is given by the camera matrix (fx for cam0)
focal_length_px = cam0[0, 0]

# Calculate the scale factor for width
scale_factor = sensor_size_mm[0] / resolution_px[0]

# Calculate the focal length in millimeters
focal_length_mm = focal_length_px * scale_factor

print(f"The focal length of the cameras is approximately {focal_length_mm:.2f} mm")

