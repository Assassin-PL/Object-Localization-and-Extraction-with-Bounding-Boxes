import cv2
import numpy as np

class Obraz:

    def __init__(self):
        self.original_image = None
        self.modified_image = None

    def load_images(self, original_path, modified_path):
        self.original_image = cv2.imread(original_path)
        self.modified_image = cv2.imread(modified_path)

    def display_images(self):
        cv2.imshow('Original Image', self.original_image)
        cv2.imshow('Modified Image', self.modified_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_modified_image_with_bboxes(self, output_path, bounding_boxes, skip_boxes=[]):
        # Clone the original image to avoid modifying it directly
        modified_with_bboxes = self.original_image.copy()

        # Create a mask for the bounding boxes
        mask = np.zeros_like(self.original_image)

        for bbox in bounding_boxes:
            x, y, w, h = bbox
            cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), thickness=cv2.FILLED)

        # Darken the background outside the bounding boxes
        darkened_background = cv2.addWeighted(self.original_image, 0.1, mask, 0.5, 0)

        # Draw bounding boxes on the darkened background
        for bbox in bounding_boxes:
            x, y, w, h = bbox
            cv2.rectangle(darkened_background, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Save the modified image with bounding boxes and darkened background
        cv2.imwrite(output_path, darkened_background)


    def save_difference_image(self, output_path, bounding_boxes):
        # Create a 4-channel image with an alpha channel
        difference_image = np.zeros((self.original_image.shape[0], self.original_image.shape[1], 4), dtype=np.uint8)

        for bbox in bounding_boxes:
            x, y, w, h = bbox
            roi_original = self.original_image[y:y+h, x:x+w]
            roi_modified = self.modified_image[y:y+h, x:x+w]

            # Compute the absolute difference between the two regions
            diff = cv2.absdiff(roi_original, roi_modified)

            # Create an alpha channel based on the differences
            alpha_channel = np.max(diff, axis=2)
            alpha_channel[alpha_channel > 0] = 255

            # Create a 4-channel image with RGB channels and an alpha channel
            difference_region = np.zeros((h, w, 4), dtype=np.uint8)
            difference_region[:, :, :3] = diff
            difference_region[:, :, 3] = alpha_channel

            # Copy the difference region to the corresponding location in the final image
            difference_image[y:y+h, x:x+w, :] = difference_region

        # difference_image = cv2.cvtColor(difference_image, cv2.COLOR_BGR2RGB)
        # Save the image with the differences and a transparent background
        cv2.imwrite(output_path, difference_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])


    def find_changes(self):
        # Compute the absolute difference between the two images
        diff = np.abs(self.original_image.astype(np.int32) - self.modified_image.astype(np.int32))

        # Convert the difference to grayscale
        gray_diff = np.max(diff, axis=2)

        # Threshold the grayscale difference
        threshold_value = 30
        binary_diff = (gray_diff > threshold_value).astype(np.uint8) * 255

        # Find contours manually
        contours = find_contours(binary_diff)

        # Extract bounding boxes from contours
        bounding_boxes = []
        for contour in contours:
            x, y, w, h = bounding_rect(contour)
            bounding_boxes.append((x, y, w, h))

        return bounding_boxes

def find_contours(binary_image):
    contours = []
    rows, cols = binary_image.shape

    # Define 8-connectivity neighbors
    neighbors = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1] if (i != 0 or j != 0)]

    # Iterate through each pixel in the binary image
    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            # If the pixel is part of a contour
            if binary_image[row, col] == 255:
                contour = []
                stack = [(row, col)]

                # Depth-first search to find contour pixels
                while stack:
                    current_pixel = stack.pop()
                    contour.append(current_pixel)

                    # Mark the current pixel as visited
                    binary_image[current_pixel] = 0

                    # Check 8-connectivity neighbors
                    for neighbor in neighbors:
                        neighbor_pixel = (current_pixel[0] + neighbor[0], current_pixel[1] + neighbor[1])

                        # If the neighbor is within bounds and part of the contour
                        if 0 <= neighbor_pixel[0] < rows and 0 <= neighbor_pixel[1] < cols and binary_image[neighbor_pixel] == 255:
                            stack.append(neighbor_pixel)

                contours.append(np.array(contour))

    return contours

def bounding_rect(contour):
    min_row = np.min(contour[:, 0])
    max_row = np.max(contour[:, 0])
    min_col = np.min(contour[:, 1])
    max_col = np.max(contour[:, 1])

    return min_col, min_row, max_col - min_col, max_row - min_row
