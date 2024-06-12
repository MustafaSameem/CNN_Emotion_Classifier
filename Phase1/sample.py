import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

#Working Directory
current_directory = os.getcwd()
print(f"Current working directory: {current_directory}")

#Define images path
images_folder = os.path.join(current_directory, '../imageSamples')

#Image Classes
classes = ['Angry', 'Engaged', 'Happy', 'Neutral']

#Function to plot 
def plot_images_with_histograms(images, titles, rows, cols, figsize=(15, 10)):
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    for i, (image, title) in enumerate(zip(images, titles), 1):
        ax = axs[(i - 1) % rows, (i - 1) // rows]  
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')

        #plot histogram
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        ax_hist = ax.twinx()
        ax_hist.plot(histogram, color='gray', alpha=0.7)
        ax_hist.set_yticks([])

    plt.tight_layout()
    plt.show()

#Choosing 15 random pictures
for class_name in classes:
    class_path = os.path.join(images_folder, class_name)
    if os.path.isdir(class_path):
        images = []
        titles = []
        for _ in range(15):
            filename = random.choice(os.listdir(class_path))
            image_path = os.path.join(class_path, filename)
            image = cv2.imread(image_path)
            images.append(image)
            titles.append(f'{class_name} - {filename}')
        plot_images_with_histograms(images, titles, 5, 3)
