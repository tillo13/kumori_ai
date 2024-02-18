# Face Recognition 

The core objective is to accurately detect a face in an image, preprocess that image, and then save it for additional analysis.  The idea was clean an image for facial recogntion automation to then pass to any facial recog software to better detect/give % likeliness of the human in the picture with best fidelity.

## Repository Structure
```bash
face_recog/
├── add_hat.py
├── remove_hat.py
├── hat.png
├── model
│   ├── deploy.prototxt.txt     # Needed for deploying the model
│   ├── res10_300x300_ssd_iter_140000.caffemodel     # Pretrained Caffe-based face detector
│   └── shape_predictor_68_face_landmarks.dat   # Landmarks model for detecting facial landmarks
├── person_images   # Directory for the input images
├── preprocess_images.py   # Python script for preprocessing the images
└── preprocessed_images   # Saved directory for the output images 
```

## Key Features
1. **Face Detection**: Using dlib's and Deep Learning model to detect the face.
2. **Bounding Box Adjustment**: Leveraging dlib's facial landmarks feature to adjust the bounding box encompassing the face.
3. **Face Verification**: Implementing a multi-scale approach and additional verification attempts for more accurate face detection.
4. **Face Preprocessing**: Checking the cropped face part for the presence of a face, resizing and storing the side face into `preprocessed_images` directory.
5. **Add Hat to Detected Face**: The script `add_hat.py` uses OpenCV's Haar cascades to detect faces in the provided image and then place a hat on each of the detected faces. The hat's position and size are adjusted for it to fit the head.
6. **Remove Hat from Detected Face**: The script `remove_hat.py` uses OpenAI's InstructGPT to remove hat from the provided image using the prompt "remove hat".

## How to Use
To run the face recog script:
```bash
python recognize_face.py
```
To add a hat to detected faces:
```bash
python addhat.py
```
To remove a hat from detected faces:
```bash
python remove_hat.py
```

## Next Steps & Future Work
The repository currently serves as the initial layer needed for face-based pipelines, like emotion detection, facial feature assessment, or face recognition tasks. In future, the following features can be included:
1. Full body detection support.
2. Additional model types for face detection.
3. Improved heuristics for bounding box adjustment.
4. Better handling for images with multiple faces.
5. Add % likeliness of person in overall groups in overlay

---
