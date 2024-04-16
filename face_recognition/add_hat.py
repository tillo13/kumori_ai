import cv2
from PIL import Image
import numpy as np  # Make sure to import numpy

# Load the image where you want to add a hat.
person_image_path = 'person_images/img_2045.png'
person_image = cv2.imread(person_image_path)

# Check if the person image was successfully loaded
if person_image is None:
    print("Could not read the person image.")
    exit()

# Convert image to RGB (OpenCV uses BGR by default)
person_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)

# Load the hat image with an alpha channel (make sure it's a PNG with transparency).
hat_image_path = 'hat.png'
hat_image = Image.open(hat_image_path).convert("RGBA")

# Load the pre-trained Haar Cascade for face detection provided by OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Convert Image to grayscale because face detection works on grayscale images
gray = cv2.cvtColor(person_image, cv2.COLOR_RGB2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# If faces are detected


if len(faces) > 0:
    for (x, y, w, h) in faces:
        # Resize hat to fit the width of the face
        hat_width = w
        hat_height = int(hat_width * hat_image.size[1] / hat_image.size[0])
        hat_resized = hat_image.resize((hat_width, hat_height), Image.Resampling.LANCZOS)



        # Define position for placing the hat on top of the head (basic positioning)
        hat_position = (x, y - hat_height)

        # Convert OpenCV image to PIL image for manipulation
        person_pil_image = Image.fromarray(person_image)
        person_pil_image.paste(hat_resized, hat_position, mask=hat_resized)

        # Convert back to OpenCV format image
        person_image = cv2.cvtColor(np.array(person_pil_image), cv2.COLOR_RGB2BGR)

# Save or display the result
cv2.imwrite('person_with_hat.jpg', person_image)
cv2.imshow('Output', person_image)
cv2.waitKey(0)
cv2.destroyAllWindows()