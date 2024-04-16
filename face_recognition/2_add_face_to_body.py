from PIL import Image, ImageDraw
import os
import cv2

# Import the detect_gender function from gender_detect module
from gender_detect import detect_gender

GLOBAL_SHOW_IMAGE_PREVIEW_AFTER = False
GLOBAL_GENDER_CONFIDENCE_THRESHOLD = 0.85
PREPROCESSED_DIR = 'found_faces'
CONSTANT_OUTPUT_DIR = 'full_body_images'

# both man_suit.png and woman_suit.png are in templates folder
body_configs = {
    'man_suit.png': {
        'path': 'templates/man_suit.png',
        'size': (522, 313, 634, 288, 438, 106, 589, 78),
        'scale_factor_multiplier': 1.5, 
        'y_offset_adjustment': 0.2, 
        'x_offset_adjustment': 0,
    },
    'woman_suit.png': {
        'path': 'templates/woman_suit.png',
        'size': (231, 78, 297, 144, 201, 39, 321, 21),
        'scale_factor_multiplier': 2.6,
        'y_offset_adjustment': 0.45,
        'x_offset_adjustment': 0.05,
    },
}

def create_ellipse_mask(size, center, radius, fill):
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([center[0] - radius[0], center[1] - radius[1], center[0] + radius[0], center[1] + radius[1]], fill=fill)
    return mask

def place_faces_on_body(preprocessed_dir, output_dir):
    # Initializing counters
    men_count = 0
    women_count = 0
    processed_images_count = 0
    total_faces_detected = 0  # Counter for total number of faces detected

    os.makedirs(output_dir, exist_ok=True)

    for face_filename in os.listdir(preprocessed_dir):
        if face_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            face_image_path = os.path.join(preprocessed_dir, face_filename)
            frame = cv2.imread(face_image_path)
            
            # Get detailed gender and age detection
            chosen_suit, gender, gender_confidence, age, age_confidence, numberOfFacesDetected = detect_gender(frame)
            total_faces_detected += numberOfFacesDetected  # Update total faces detected
            
            # Log details
            print("====IMAGE START====")
            print(f"Image name: {face_filename}")
            print(f"Number of faces detected: {numberOfFacesDetected}")
            print(f"Gender: {gender}")
            print(f"Gender confidence: {gender_confidence * 100:.2f}%")
            print(f"Age: {age} years")
            print(f"Age confidence: {age_confidence * 100:.2f}%")

            # Analyze gender and decide on the suit to use
            if numberOfFacesDetected > 0:
                unsure_prefix = ""  # Initialize unsure_prefix to be empty
                if gender == 'Female' and gender_confidence >= GLOBAL_GENDER_CONFIDENCE_THRESHOLD:
                    chosen_suit_config = 'woman_suit.png'
                    women_count += 1
                else:
                    chosen_suit_config = 'man_suit.png'
                    men_count += 1
                    # Here, adjust unsure_prefix based on confidence level
                    if gender_confidence < GLOBAL_GENDER_CONFIDENCE_THRESHOLD:
                        unsure_prefix = "unsure_"

                processed_images_count += 1
                body_config = body_configs[chosen_suit_config]
                body_image = Image.open(body_config['path']).convert("RGBA")
                # Pass unsure_prefix to add_to_body
                add_to_body(body_image, face_image_path, face_filename, body_config, output_dir, gender=("Female" if chosen_suit_config == 'woman_suit.png' else "Male"), gender_confidence=gender_confidence, unsure_prefix=unsure_prefix)
            else:
                # No faces detected handling
                print(f"Skipped {face_filename}: No faces detected.")
            print("====IMAGE END====\n")
    # Print summary
    print("====SUMMARY====")
    print(f"Total faces detected: {total_faces_detected}")
    print(f"Men detected: {men_count}")
    print(f"Women detected: {women_count}")
    print(f"Total processed images: {processed_images_count}")

def add_to_body(body_image, face_image_path, face_filename, body_config, output_dir, gender, gender_confidence, unsure_prefix):
    body_w, body_h = body_image.size
    
    size_params = body_config['size']
    bottom_center_neck_y = (size_params[1] + size_params[3]) // 2
    top_center_head_y = (size_params[5] + size_params[7]) // 2
    head_neck_height = bottom_center_neck_y - top_center_head_y

    scale_factor_multiplier = body_config['scale_factor_multiplier']
    y_offset_adjustment = body_config['y_offset_adjustment']
    x_offset_adjustment = body_config['x_offset_adjustment']

    face_image = Image.open(face_image_path).convert("RGBA")
    face_w, face_h = face_image.size

    scale_factor = (head_neck_height / face_h) * scale_factor_multiplier

    scaled_face_w = int(face_w * scale_factor)
    scaled_face_h = int(face_h * scale_factor)
    face_image = face_image.resize((scaled_face_w, scaled_face_h), Image.Resampling.LANCZOS)

    x_offset = (body_w - scaled_face_w) // 2 + int(body_w * x_offset_adjustment)
    y_offset = bottom_center_neck_y - scaled_face_h + int(scaled_face_h * y_offset_adjustment)

    body_with_face = body_image.copy()
    mask = create_ellipse_mask((scaled_face_w, scaled_face_h), (scaled_face_w // 2, scaled_face_h // 2), (scaled_face_w // 2, scaled_face_h // 2), 255)
    body_with_face.paste(face_image, (x_offset, y_offset), mask)


    # Modified gender prefix to incorporate unsure_prefix
    gender_str = "male" if gender == "Male" else "female"  # Simplify the gender string
    gender_prefix = f"{unsure_prefix}{gender_str}_{round(gender_confidence * 100)}_"
    
    output_image_path = os.path.join(output_dir, f"{gender_prefix}{face_filename}")

    # Converting and saving the image code remains the same
    if output_image_path.lower().endswith('.jpg') or output_image_path.lower().endswith('.jpeg'):
        body_with_face = body_with_face.convert("RGB")
    body_with_face.save(output_image_path)
    print(f"Processed and saved: {output_image_path}")

if __name__ == "__main__":
    place_faces_on_body(PREPROCESSED_DIR, CONSTANT_OUTPUT_DIR)