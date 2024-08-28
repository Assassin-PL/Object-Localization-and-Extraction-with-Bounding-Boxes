# Object-Localization-and-Extraction-with-Bounding-Boxes
This project focuses on localizing a specified object within an image, enclosing it with a bounding box, and extracting the object by removing the surrounding background, leaving only the object with a transparent background. The project uses only NumPy and OpenCV libraries for image processing tasks.

This project focuses on localizing a specified object within an image, enclosing it with a bounding box, and extracting the object by removing the surrounding background, leaving only the object with a transparent background. The project uses only NumPy and OpenCV libraries for image processing tasks.

## Project Overview

The primary goal of this project is to demonstrate the following tasks:

1. **Localization**: Locate the specified object within the image using simple image processing techniques.
2. **Bounding Box**: Draw a bounding box around the detected object.
3. **Background Removal**: Extract the object from the image by removing the background, leaving only the object with a transparent background.

## Project Structure

The project is organized as follows:

- **Klasa.py**: Contains the `Obraz` class, which provides methods for loading images, detecting changes between two images, and saving images with bounding boxes or with the background removed. The method for detecting bounding boxes involves applying masks to the image using the OpenCV library, which helps identify regions of difference between the original image and a modified version (e.g., where a cola can has been added).
- **KlasaTest.py**: Similar to `Klasa.py`, but may contain variations or testing versions of the methods.
- **main.py**: The main script that demonstrates the functionality of the `Obraz` class, including loading images, detecting changes, and saving the processed images.
## Requirements

- Python 3.x
- NumPy
- OpenCV

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git

2. Install the required dependencies:
    ```bash
    pip install numpy opencv-python
## Running the Project

1. **Load Images**:
   Load the original and modified images using the `load_images` method.

   ```python
   image_handler = Obraz()
   image_handler.load_images('original_image.jpg', 'modified_image.jpg')
   
2. **Find Changes and Draw Bounding Boxes** :
Detect the changes between the images and draw bounding boxes around the detected differences.
    ```python
      bounding_boxes = image_handler.find_changes()
      image_handler.save_modified_image_with_bboxes('output_with_bboxes.png', bounding_boxes)

3. Save the Image with Transparent Background: Extract the object within the bounding box and save the result with a transparent background.
    ```python
     image_handler.save_difference_image('output_cut.png', bounding_boxes)
  ## Example Usage

The project includes an example script in `main.py` that demonstrates how to use the `Obraz` class to perform all the steps described above. Simply run the script after setting the paths to your images.

```bash
python main.py

