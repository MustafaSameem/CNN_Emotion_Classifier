import os
from PIL import Image, ImageEnhance, ImageStat

def is_image_too_dark(image, threshold=120):
    
    stat = ImageStat.Stat(image)
    brightness = stat.mean[0]
    return brightness < threshold

def increase_brightness_if_dark(input_directory, brightness_factor=1.5, brightness_threshold=120):
    # go through all images
    for filename in os.listdir(input_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # open image 
            img = Image.open(os.path.join(input_directory, filename))

            # check if the image is too dark
            if is_image_too_dark(img, brightness_threshold):
                # increase the brightness
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(brightness_factor)

                # save the image in the same directory
                img.save(os.path.join(input_directory, filename))

    print("Done!")

# change directory depending on the emotion
input_dir = './imageSamples/Angry'
brightness_factor = 1.5  
brightness_threshold = 120  # threshold for detecting dark images


increase_brightness_if_dark(input_dir, brightness_factor, brightness_threshold)
