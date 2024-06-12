import os
import matplotlib.pyplot as plt

#Working Directory
current_directory = os.getcwd()
print(f"Current working directory: {current_directory}")

#Define images path
images_folder = os.path.join(current_directory, '../imageSamples')

#Image Classes
classes = ['Angry', 'Engaged', 'Happy', 'Neutral']


class_counts = {}
#Count images
for class_name in classes:
    class_path = os.path.join(images_folder, class_name)
    if os.path.isdir(class_path):
        files = os.listdir(class_path)
        image_files = [f for f in files if os.path.isfile(os.path.join(class_path, f))]
        class_counts[class_name] = len(image_files)
    else:
        class_counts[class_name] = 0
        print(f"The folder {class_path} does not exist.")

#Print count for each class
for class_name, count in class_counts.items():
    print(f"{class_name}: {count} imageSamples")

#Create bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(class_counts.keys(), class_counts.values(), color=['red', 'blue', 'green', 'orange'])
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Class Distribution')
plt.xticks(rotation=45)
plt.ylim(0, 510)  
plt.grid(axis='y')

#Add labels inside bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval - 10, int(yval), ha='center', va='top', color='white')

#Plot
plt.show()
