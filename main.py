from KlasaTest import *

print("Hello world")

original_image_path = 'london.jpg'
modified_image_path = 'london_ed.jpg'
destination_path = 'londonTest.png'
difference_path = 'difference.jpg'
output_cut_path = 'london_cut.png'


image_handler = Obraz()
image_handler.load_images(original_image_path, modified_image_path)
# image_handler.display_images()
skip_boxes = []  # Specify the bounding boxes to skip

bounding_boxes = image_handler.find_changes()
if bounding_boxes:
    print("Znalezione różnice. Bounding box'y:", bounding_boxes)
    # obraz z wycietymi Bounding boxami
    image_handler.save_modified_image(output_cut_path, bounding_boxes)
    # obraz z zostawionymi Bounding boxami
    output_modified_path = destination_path
    image_handler.save_modified_image_with_bboxes(output_modified_path, bounding_boxes)
    # obraz z samymi Bounding boxami
    output_difference_path = difference_path
    image_handler.save_difference_image(output_difference_path, bounding_boxes)
else:
    print("Brak różnic.")
