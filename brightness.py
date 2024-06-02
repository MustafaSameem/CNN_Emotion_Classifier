import os
from PIL import Image, ImageEnhance

def increase_brightness(input_directory, output_directory, brightness_factor):
    # check if directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # go through all images
    for filename in os.listdir(input_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # open image 
            img = Image.open(os.path.join(input_directory, filename))

            # enhance the brightness
            enhancer = ImageEnhance.Brightness(img)
            img_enhanced = enhancer.enhance(brightness_factor)

            # save image
            output_path = os.path.join(output_directory, filename)
            img_enhanced.save(output_path)

    print("Brightness increased for all images in the directory.")

#change directory for each Class
input_dir = './images/Angry'
output_dir = './imageClasses/Angry'
brightness = 1.5  


increase_brightness(input_dir, output_dir, brightness)
