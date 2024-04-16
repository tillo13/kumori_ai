import cv2
import os
import time
import numpy as np
import dlib
import datetime
from datetime import datetime
import re

GLOBAL_SURROUNDING_AROUND_FACE=1.75
MODEL_DIR = "facial_landmarks_model"
PROTOTXT = os.path.join(MODEL_DIR, "deploy.prototxt.txt")
MODEL = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
INPUT_DIRECTORY = 'person_images'
OUTPUT_DIRECTORY = 'found_faces'
DESIRED_SIZE = (256, 256)
PADDING = 20
CONFIDENCE_THRESHOLD = 0.5
LANDMARKS_MODEL_PATH = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")


class FacePreprocessor:
    def __init__(self, prototxt_path, model_path, input_dir, output_dir, desired_size, padding, confidence_threshold, landmarks_model_path):
        # Initialize other attributes here...
        self.directly_saved_files = 0  # Counter for directly saved files
        # Initialize the deep learning model for face detection
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        # Setup local variables for paths and preprocessing parameters
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.desired_size = desired_size
        self.padding = padding
        self.confidence_threshold = confidence_threshold
        self.valid_extensions = {'.jpeg', '.jpg', '.png'}
        os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

        self.landmarks_detector = dlib.shape_predictor(landmarks_model_path)
        self.face_detector = dlib.get_frontal_face_detector()

    def get_landmarks(self, image, rectangle):
        return np.matrix([[p.x, p.y] for p in self.landmarks_detector(image, rectangle).parts()])

    def adjust_bounding_box(self, image, x, y, w, h):
        dlib_rectangle = dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)
        landmarks = self.get_landmarks(image, dlib_rectangle)

        # Start with the initial bounding box values
        new_x, new_y, new_w, new_h = x, y, w, h

        # If landmarks couldn't be detected, return the initial bounding box
        if landmarks.size == 0:
            return (new_x, new_y, new_w, new_h)  # No landmarks were found
        
        # Heuristic to adjust the bounding box based on landmarks for side views
        jaw_points = landmarks[0:17]  # Jawline points, assuming a 68-point landmark model
        # Check if the face is turned by comparing distances between points on jawline
        dist_left = np.linalg.norm(jaw_points[0] - jaw_points[3])
        dist_right = np.linalg.norm(jaw_points[16] - jaw_points[13])

        # Determine which side of the face is visible
        if dist_left < dist_right:
            # Face turned to the left, expand the bounding box on the left side
            expansion = w // 2
            new_x = max(x - expansion, 0)
            new_w = w + expansion
        elif dist_right < dist_left:
            # Face turned to the right, expand the bounding box on the right side
            expansion = w // 2
            new_w = min(w + expansion, image.shape[1] - x)

        # Adjust the bounding box height to include the whole head if needed
        head_height_expansion = h // 4
        new_y = max(y - head_height_expansion, 0)
        new_h = min(h + head_height_expansion, image.shape[0] - y)

        return (new_x, new_y, new_w, new_h) 



    def is_processing_needed(self, x, y, w, h, img_width, img_height, enlargement_factor=GLOBAL_SURROUNDING_AROUND_FACE):
        """
        Checks if the detected face meets the criteria to skip cropping and resizing.
        """
        # Calculate expected dimensions and position based on enlargement factor
        expected_width = int(w * enlargement_factor)
        expected_height = int(h * enlargement_factor)
        
        center_x, center_y = x + w // 2, y + h // 2
        new_x = max(center_x - expected_width // 2, 0)
        new_y = max(center_y - expected_height // 2, 0)
        
        # Check if the new dimensions are significantly different from the desired size
        if abs(expected_width - self.desired_size[0]) < 20 and abs(expected_height - self.desired_size[1]) < 20:
            # Check if the face is roughly centered
            if abs(new_x - (img_width - expected_width) // 2) < 50 and abs(new_y - (img_height - expected_height) // 2) < 50:
                return False
        return True

    def save_directly(self, image, x, y, w, h, output_path, enlargement_factor=GLOBAL_SURROUNDING_AROUND_FACE):
        
        """
        Handles minimal adjustments and directly saves the specified face region
        as identified by the bounding box coordinates (x, y, w, h).
        """
        # Apply enlargement factor to the detected face region if necessary
        center_x, center_y = x + w // 2, y + h // 2
        new_w = int(w * enlargement_factor)
        new_h = int(h * enlargement_factor)
        new_x = max(center_x - new_w // 2, 0)
        new_y = max(center_y - new_h // 2, 0)

        # Ensure the adjusted region is within image boundaries
        new_w = min(new_w, image.shape[1] - new_x)
        new_h = min(new_h, image.shape[0] - new_y)

        # Crop the adjusted region
        cropped_image = image[new_y:new_y+new_h, new_x:new_x+new_w]

        # Resize the cropped image to the desired output size if necessary
        resized_image = cv2.resize(cropped_image, self.desired_size, interpolation=cv2.INTER_AREA)

        # Save the resulting image
        cv2.imwrite(output_path, resized_image)
        print(f"Directly saved minimally adjusted image as {output_path}")

        # Increment the counter
        self.directly_saved_files += 1

    def process_images(self, max_retry_count=3, enlarge_factor=1.1):
        start_time = time.time()
        faces_found, faces_not_found, processed_files = 0, 0, 0
        
        # Process each file in the input directory
        for file_name in os.listdir(self.input_dir):
            # Use regex to replace any non alphanumeric characters (excluding '.') with underscore.
            # The updated regex specifically avoids replacing '.' used in file extensions.
            file_name = re.sub(r'[^\w.]+', '_', file_name)  # Modification to replace non-alphanumeric characters
            
            extension = os.path.splitext(file_name)[1].lower()
            if extension in self.valid_extensions:
                processed_files += 1
                image_path = os.path.join(self.input_dir, file_name)
                # Load the image from the disk
                image_path = os.path.join(self.input_dir, file_name)
                if not os.path.exists(image_path):
                    print(f"The file {image_path} does not exist. Skipping...")
                    faces_not_found += 1
                    continue

                image = cv2.imread(image_path)
                if image is None:
                    print(f"Could not read the image {file_name}. Skipping...")
                    faces_not_found += 1
                    continue

                output_path = os.path.join(self.output_dir, file_name)
                # Check if the file already exists
                if os.path.exists(output_path):
                    # Split the filename from its extension
                    name, extension = os.path.splitext(file_name)
                    # Generate a datetime string
                    datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                    # Append datetime before the file extension and reconstruct output_path
                    new_filename = f"{name}_{datetime_str}{extension}"
                    output_path = os.path.join(self.output_dir, new_filename)

                    
                # Load the image from the disk
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Could not read the image {file_name}. Skipping...")
                    faces_not_found += 1
                    continue

                # Get the dimensions of the image
                h, w = image.shape[:2]

                # Detect faces using dlib
                print(f"Processing {file_name} with dlib...")
                best_face = self.detect_faces_with_dlib(image)

                # If dlib doesn't detect any faces, use the deep learning model
                if best_face is None:
                    print("No face detected with dlib, using deep learning model.")
                    enlarged_image = cv2.copyMakeBorder(
                        image,
                        top=int(h * (enlarge_factor - 1)),
                        bottom=int(h * (enlarge_factor - 1)),
                        left=int(w * (enlarge_factor - 1)),
                        right=int(w * (enlarge_factor - 1)),
                        borderType=cv2.BORDER_CONSTANT,
                        value=[0, 0, 0]
                    )
                    best_face = self.detect_faces_with_attempts(enlarged_image)

                    

                # set it
                if best_face:
                    x, y, w, h = best_face

                    # Check if we need to process this face
                    if self.is_processing_needed(x, y, w, h, image.shape[1], image.shape[0]):
                        # Proceed with cropping and resizing
                        if self.crop_and_resize_face(image, x, y, w, h, output_path):
                            print(f"Saved preprocessed image as {output_path}")
                    else:
                        # If the image doesn't need extensive processing, save with minimal adjustment
                        self.save_directly(image, x, y, w, h, output_path)

                    # Confirm the presence of the face after processing
                    # This is crucial to ensuring the adjustments didn't affect face detection
                    face_detected = self.verify_face_presence(output_path, image, max_retry_count)
                    if not face_detected:
                        print(f"Face not detected after adjustments for {output_path}. Marking for review.")
                        faces_not_found += 1
                    else:
                        faces_found += 1
                else:
                    print(f"No face found in {file_name}. Skipping...")
                    faces_not_found += 1


        # Display processing summary
        elapsed_time = time.time() - start_time
        print("Image preprocessing is complete.")
        print(f"Processed a total of {processed_files} files.")
        print(f"Faces found and processed in {faces_found} images.")
        print(f"Faces not found in {faces_not_found} images.")
        print(f"Files saved directly with minimal processing: {self.directly_saved_files}")
        print(f"Time taken: {elapsed_time:.2f} seconds.")

    # Method to detect the largest face in the image using the deep learning model
    def detect_faces_with_attempts(self, image):
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        best_face = None
        max_area = 0
        image_center = np.array([w / 2, h / 2])

        # Iterate through detections and keep the largest face
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face_area = (endX - startX) * (endY - startY)
                face_center = np.array([(startX + endX) / 2, (startY + endY) / 2])
                distance = np.linalg.norm(face_center - image_center)

                if face_area > max_area and distance < image_center[0]:
                    max_area = face_area
                    best_face = (startX, startY, endX - startX, endY - startY)

                # After confidence check
                if confidence > self.confidence_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    # Modify this part:
                    (startX, startY, endX, endY) = self.adjust_bounding_box(image, startX, startY, endX-startX, endY-startY)
                    face_area = (endX - startX) * (endY - startY)


        return best_face

    # Method to check for the presence of a face in a saved image
    def recheck_image_for_face(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return False

        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        
        # Check each detection for a high enough confidence score
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                return True  # Face found
        return False  # No face found

    # Method to crop and resize the face region from an image
    def crop_and_resize_face(self, image, x, y, w, h, output_path, enlargement_factor=GLOBAL_SURROUNDING_AROUND_FACE):
        print(f"Original Image Dimensions: Width {image.shape[1]}, Height {image.shape[0]}")
        original_aspect_ratio = image.shape[1] / image.shape[0]
        
        # Define desired dimensions and calculate aspect ratio
        desired_aspect_ratio = self.desired_size[0] / self.desired_size[1]
        print(f"Desired Aspect Ratio: {desired_aspect_ratio}, Original Aspect Ratio: {original_aspect_ratio}")
        
        # Apply enlargement around the detected face region
        center_x, center_y = x + w // 2, y + h // 2
        new_w = int(w * enlargement_factor)
        new_h = int(h * enlargement_factor)
        new_x = max(center_x - new_w // 2, 0)
        new_y = max(center_y - new_h // 2, 0)
        
        cropped_image = image[new_y:new_y+new_h, new_x:new_x+new_w]
        cropped_aspect_ratio = cropped_image.shape[1] / cropped_image.shape[0]
        print(f"Cropped Image Dimensions: Width {cropped_image.shape[1]}, Height {cropped_image.shape[0]}")

        # Calculate scaling factor based on the aspect ratio comparison
        if cropped_aspect_ratio > desired_aspect_ratio:
            # Width is the constraining dimension
            scale_factor = self.desired_size[0] / cropped_image.shape[1]
        else:
            # Height is the constraining dimension
            scale_factor = self.desired_size[1] / cropped_image.shape[0]
        
        resized_width = int(cropped_image.shape[1] * scale_factor)
        resized_height = int(cropped_image.shape[0] * scale_factor)
        
        resized_image = cv2.resize(cropped_image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        print(f"Resized Image Dimensions: Width {resized_width}, Height {resized_height}")
        
        # Add padding to meet the desired size
        delta_w = self.desired_size[0] - resized_width
        delta_h = self.desired_size[1] - resized_height
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        colored_resized_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        print(f"Final Image Dimensions: Width {colored_resized_image.shape[1]}, Height {colored_resized_image.shape[0]}")

        cv2.imwrite(output_path, colored_resized_image)
        print(f"Image saved as: {output_path}")
        return True

    def _crop_resize_util(self, image, x, y, w, h, output_path):
        # Crop and resize logic remains the same
        x1, y1 = max(x - self.padding, 0), max(y - self.padding, 0)
        x2, y2 = min(x + w + self.padding, image.shape[1]), min(y + h + self.padding, image.shape[0])

        cropped_image = image[y1:y2, x1:x2]
        if cropped_image.size == 0:
            print(f"Failed to crop a valid region for {output_path}.")
            return False
        
        aspect_ratio = w / h
        desired_width = self.desired_size[0]
        desired_height = int(desired_width / aspect_ratio) if aspect_ratio > 1 else self.desired_size[1]
        resized_image = cv2.resize(cropped_image, (desired_width, desired_height), interpolation=cv2.INTER_AREA)

        if desired_height != self.desired_size[1]:
            delta_height = max(self.desired_size[1] - desired_height, 0)
            top, bottom = delta_height // 2, delta_height - (delta_height // 2)
            resized_image = cv2.copyMakeBorder(resized_image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        cv2.imwrite(output_path, resized_image)
        return True

    # Method to attempt face detection with retries and potentially using a multi-scale approach
    def verify_face_presence(self, output_path, image, max_retry_count):
        retry_count = 0
        while retry_count < max_retry_count:
            if self.recheck_image_for_face(output_path):
                return True  # Face found, no need to retry.
            elif retry_count == max_retry_count - 1:  # Fallback on the last retry
                print(f"Trying multi-scale detection for {output_path}...")
                best_face = self.multi_scale_face_detection(image)
                if best_face:
                    x, y, w, h = best_face
                    if self.crop_and_resize_face(image, x, y, w, h, output_path) and self.recheck_image_for_face(output_path):
                        return True  # Face found with multi-scale detection
            retry_count += 1
            print(f"Retrying detection (attempt {retry_count}) on {output_path}...")
        return False  # Face not found after retries

    # Method to perform face detection at multiple scales
    def multi_scale_face_detection(self, image, scales=[0.5, 0.75, 1.0, 1.5, 2.0]):
        (h0, w0) = image.shape[:2]
        best_face = None
        max_area = 0

        for scale in scales:
            width = int(w0 * scale)
            height = int(h0 * scale)
            resized_image = cv2.resize(image, (width, height))
            
            blob = cv2.dnn.blobFromImage(resized_image, 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.net.setInput(blob)
            output_detections = self.net.forward()

            # Process detections and find the best face
            for i in range(0, output_detections.shape[2]):
                confidence = output_detections[0, 0, i, 2]
                if confidence > self.confidence_threshold:
                    box = output_detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (startX, startY, endX, endY) = box.astype("int")

                    face_area = (endX - startX) * (endY - startY)
                    if face_area > max_area:
                        max_area = face_area
                        best_face = (int(startX / scale), int(startY / scale), 
                                     int((endX - startX) / scale), int((endY - startY) / scale))

        return best_face

    # Method to detect the largest face in the image using dlib
    def detect_faces_with_dlib(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray, 1)
        
        if len(faces) == 0:
            return None

        # Find the face that is largest and most central
        largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
        return (largest_face.left(), largest_face.top(), largest_face.width(), largest_face.height())
    


# Now instantiate the FacePreprocessor class and process the images
preprocessor = FacePreprocessor(PROTOTXT, MODEL, INPUT_DIRECTORY, OUTPUT_DIRECTORY,
                                DESIRED_SIZE, PADDING, CONFIDENCE_THRESHOLD, LANDMARKS_MODEL_PATH)
preprocessor.process_images()