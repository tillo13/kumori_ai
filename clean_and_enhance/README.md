# Image Enhancement and Super-Resolution Script
This script is designed to carry out cleaning, enhancement, and super-resolution processing on a folder of images. It applies sharpening, contrast, color enhancements, face detection based sharpening, and super resolution via LapSRN on the fetched images. The script provides feedback about its progress, with estimates for the time remaining until completion.

## Requirements
- **Python** (recommended 3.6 and above)
- **OpenCV** (version 4.5.5.62 required)
- **PIL (Python Imaging Library)**  

You can install the specific version of OpenCV with this command:
```
pip install opencv-contrib-python==4.5.5.62
```

## Variables
- `image_directory` : The directory where your images are located.
- `target_size` : The maximum size the images should be after super-resolution.
- `max_pre_sr_size` : The maximum size an image can be before super-resolution is applied.
- `sharpness_factor`, `contrast_factor`, `color_factor` : Enchancement factors for image corrections.

## How to Run
1. Make sure all the libraries installed.
2. Set the `image_directory` variable with the path of directory containing your images
3. Adjust the `target_size`, `max_pre_sr_size`, and various enhancement factors according to your preference.
4. Run the script, and it will process all the images in the directory, creating a cleaned, enhanced and super-resolution version of each.
5. After processing, the original images will be moved to the 'unedited_images' folder inside `image_directory`.

```
## Examples
This script can be useful in applications like,
1. Enhancing old photographs
2. Preprocessing images for machine learning tasks
3. Improving images for digital displays