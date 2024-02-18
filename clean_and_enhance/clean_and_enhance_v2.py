import os
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import time
import shutil
import sys

# Function to compare version numbers
def versiontuple(v):
    return tuple(map(int, (v.split("."))))

# Check if the current OpenCV version meets the requirement
required_version = '4.5.5.62'
current_version = cv2.__version__

if versiontuple(current_version) < versiontuple(required_version):
    print(f"Warning: Your OpenCV version is {current_version}, which is not compatible with this script.")
    print(f"To ensure compatibility, please install OpenCV version {required_version} by running the following command:")
    print(f"pip install opencv-contrib-python=={required_version}")
    sys.exit(1)

image_directory = '.'
file_number = 0
processed_files = 0
target_size = 2048
max_pre_sr_size = 1024

sharpness_factor = 1.2
contrast_factor = 1.1
color_factor = 1.1

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Calculate the total number of image files to process
all_files = [f for f in sorted(os.listdir(image_directory))
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
total_files = len(all_files)

# Create an empty list to store processing times
processing_times = []

# Create an SR object for LapSRN
sr = cv2.dnn_superres.DnnSuperResImpl_create()
path_to_lap_srn_model = "LapSRN_x8.pb"
sr.readModel(path_to_lap_srn_model)
sr.setModel('lapsrn', 8)

print('Starting to process images...')

for filename in all_files:
    start_time = time.time()
    file_number += 1  
    remaining_images = total_files - file_number
    if processing_times:
        avg_time_per_image = sum(processing_times) / len(processing_times)
        estimated_time_left = avg_time_per_image * remaining_images
        estimated_minutes = int(estimated_time_left // 60)
        estimated_seconds = int(estimated_time_left % 60)
        print(f"\nProcessing file: {filename}...")
        print(f"Remaining images: {remaining_images}")
        print(f"Estimated time to completion, based on average of previous images: {estimated_minutes} mins {estimated_seconds} seconds")
    else:
        print(f"\nProcessing file: {filename}...")
        print("Calculating time estimates after this image...")


    old_filepath = os.path.join(image_directory, filename)
    new_filename = f'cleaned_and_enhanced_{file_number}.png'
    new_filepath = os.path.join(image_directory, new_filename)

    with Image.open(old_filepath) as img:
        orig_size = img.size
        print(f'Original size: {orig_size}')
        
        # Resize for pre-super-resolution if needed
        if max(img.size) > max_pre_sr_size:
            print('Resizing image for super-resolution...')
            resize_factor = max_pre_sr_size / max(img.size)
            new_dimensions = (int(img.size[0] * resize_factor), int(img.size[1] * resize_factor))
            img = img.resize(new_dimensions, Image.LANCZOS)
            print(f'Resized to: {new_dimensions}')
        
        # Convert image to grayscale for face detection
        gray_img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # You could apply specific enhancements to faces here, if desired.
            # For example, sharpening just the face:
            face_region = img.crop((x, y, x + w, y + h))
            enhancer = ImageEnhance.Sharpness(face_region)
            enhanced_face = enhancer.enhance(2.0)  # Apply stronger sharpness to face
            img.paste(enhanced_face, (x, y))
        
        # Convert back to color image
        img = img.convert('RGB')
        
        # Apply general enhancements
        img = ImageEnhance.Sharpness(img).enhance(sharpness_factor)
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)
        img = ImageEnhance.Color(img).enhance(color_factor)
        
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        print('Starting super-resolution...')
        img_cv = sr.upsample(img_cv)
        print('Super-resolution completed.')
        
        img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        if max(img.size) > target_size:
            img.thumbnail((target_size, target_size), Image.LANCZOS)
        
        img.save(new_filepath, 'PNG')
        
        print(f'Processed {filename}: saved as {new_filename}')
        
        unedited_images_directory = os.path.join(image_directory, "unedited_images")
        os.makedirs(unedited_images_directory, exist_ok=True)
        shutil.move(old_filepath, os.path.join(unedited_images_directory, filename))
            
    end_time = time.time()
    processing_time = end_time - start_time
    processing_times.append(processing_time)
    processed_files += 1
    # Remember to include the remainder of the for-loop where images are processed

# After processing all images
print(f'\nTotal images processed: {processed_files}')
total_time_taken = sum(processing_times)
total_minutes = int(total_time_taken // 60)
total_seconds = int(total_time_taken % 60)
print(f'Total time taken: {total_minutes} mins {total_seconds} seconds')