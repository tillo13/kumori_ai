from PIL import Image, ImageDraw
import os

GLOBAL_SHOW_IMAGE_PREVIEW_AFTER = False
PREPROCESSED_DIR = 'found_faces'
CONSTANT_OUTPUT_DIR = 'full_body_images'


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
        'scale_factor_multiplier': 2.2, 
        'y_offset_adjustment': 0.45, 
        'x_offset_adjustment': 0.05, 
    },
}

def create_ellipse_mask(size, center, radius, fill):
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([center[0] - radius[0], center[1] - radius[1], center[0] + radius[0], center[1] + radius[1]], fill=fill)
    return mask

def place_faces_on_body(preprocessed_dir, body_config, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    body_image_path = body_config['path']
    body_image = Image.open(body_image_path).convert("RGBA")
    body_w, body_h = body_image.size
    
    size_params = body_config['size']
    bottom_center_neck_y = (size_params[1] + size_params[3]) // 2
    top_center_head_y = (size_params[5] + size_params[7]) // 2
    head_neck_height = bottom_center_neck_y - top_center_head_y

    scale_factor_multiplier = body_config['scale_factor_multiplier']
    y_offset_adjustment = body_config['y_offset_adjustment']
    x_offset_adjustment = body_config['x_offset_adjustment']

    for face_filename in os.listdir(preprocessed_dir):
        if face_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            face_image_path = os.path.join(preprocessed_dir, face_filename)
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

            output_image_path = os.path.join(output_dir, f"body_{body_image_path.split('/')[-1].split('.')[0]}_with_{face_filename}")
            if output_image_path.lower().endswith('.jpg') or output_image_path.lower().endswith('.jpeg'):
                body_with_face = body_with_face.convert("RGB")
            body_with_face.save(output_image_path)
            print(f"Processed and saved: {output_image_path}")

for body_image_name, config in body_configs.items():
    print(f"Processing {body_image_name}...")
    place_faces_on_body(PREPROCESSED_DIR, config, CONSTANT_OUTPUT_DIR)