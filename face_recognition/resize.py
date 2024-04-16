import cv2
import numpy as np

def resize_image(input_path, output_path, desired_size=(256, 256)):
    # Load the image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Could not read the image: {input_path}")
        return False

    # Optional: Correct orientation based on metadata
    # image = correct_image_orientation(image)

    # Resize the image
    h, w = image.shape[:2]
    print(f"Original Size: {w}x{h}")

    # Calculate the scale to use for resizing while maintaining aspect ratio
    scale = min(desired_size[0] / w, desired_size[1] / h)

    # Calculate the new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize the image
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Optionally add padding to maintain the desired size
    delta_w = desired_size[0] - new_w
    delta_h = desired_size[1] - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]  # The color for padding; black in this case
    final_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    # Save the final image
    cv2.imwrite(output_path, final_image)
    print(f"Resized image saved as: {output_path}")
    return True

# Replace 'input_directory' with the appropriate path to your 'ab.png' file
# Ensure 'output_directory' points to a valid output directory
input_directory = "person_images"
output_directory = "modded_images"
file_name = "ab.png"
input_path = f"{input_directory}/{file_name}"
output_path = f"{output_directory}/resized_{file_name}"

# Call the function to resize the image
resize_image(input_path, output_path)