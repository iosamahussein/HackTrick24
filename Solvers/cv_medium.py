import cv2
import numpy as np


def find_image_coordinates(larger_image , patch_image):
    larger_gray = cv2.cvtColor(larger_image, cv2.COLOR_BGR2GRAY)
    patch_gray = cv2.cvtColor(patch_image, cv2.COLOR_BGR2GRAY)

    # Find the key points and descriptors with SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(larger_gray, None)
    kp2, des2 = sift.detectAndCompute(patch_gray, None)

    # Use a FLANN based matcher to find matches between the descriptors
    flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})
    matches = flann.knnMatch(des1, des2, k=2)

    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        h, w = patch_gray.shape

        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        transformed_pts = cv2.perspectiveTransform(pts, M)
        top_left = np.min(transformed_pts, axis=0).astype(int).ravel()
        bottom_right = np.max(transformed_pts, axis=0).astype(int).ravel()

        # roi = larger_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        return top_left, bottom_right

    else:
        print("Not enough matches found.")
        return None

def cv_medium(larger_image, patch_image, add_value=16):
    result = find_image_coordinates(larger_image, patch_image)
    if result is not None:
        top_left_coord, bottom_right_coord = result

        # Calculate the dimensions of the scaled-up region
        height, width, _ = larger_image.shape
        added_height = min(add_value, (bottom_right_coord[1] - top_left_coord[1]) // 2)
        added_width = min(add_value, (bottom_right_coord[0] - top_left_coord[0]) // 2)

        # Calculate the top-left coordinate for the scaled-up region, ensuring it stays within image boundaries
        added_top_left_y = max(0, top_left_coord[1] - added_height)
        added_top_left_x = max(0, top_left_coord[0] - added_width)

        # Calculate the bottom-right coordinate for the scaled-up region, ensuring it stays within image boundaries
        added_bottom_right_y = min(height, bottom_right_coord[1] + added_height)
        added_bottom_right_x = min(width, bottom_right_coord[0] + added_width)

        # Create the scaled-up mask
        mask = np.zeros_like(larger_image)
        mask[added_top_left_y:added_bottom_right_y, added_top_left_x:added_bottom_right_x] = 255

        result_image = cv2.inpaint(larger_image, mask[:, :, 0], inpaintRadius=8, flags=cv2.INPAINT_TELEA)

        return result_image.tolist()
    else:
        return None
