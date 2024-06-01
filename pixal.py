import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

#Working Directory
current_directory = os.getcwd()
print(f"Current working directory: {current_directory}")

#Define images path
images_folder = os.path.join(current_directory, 'imageSamples')

#Image Classes
classes = ['Angry', 'Engaged', 'Happy', 'Neutral']


class_pixel_distributions = {}
#Calculate pixel intensity 
for class_name in classes:
    class_path = os.path.join(images_folder, class_name)
    if os.path.isdir(class_path):
        pixel_values = []
        for filename in os.listdir(class_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(class_path, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                pixel_values.extend(image.flatten())
        class_pixel_distributions[class_name] = pixel_values

#Plot
plt.figure(figsize=(12, 8))
for i, (class_name, pixel_values) in enumerate(class_pixel_distributions.items()):
    plt.subplot(2, 2, i+1)
    plt.hist(pixel_values, bins=256, range=(0, 256), color='gray', alpha=0.7, label='Total Intensity')
    plt.title(class_name)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()

plt.tight_layout()
plt.show()
