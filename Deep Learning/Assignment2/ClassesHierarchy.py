import pandas as pd
import os
import shutil

data = pd.read_csv('Ocular_Disease_Classification/train/_classes.csv')
destination_directory = 'Ocular_Disease_Classification/TrainHierarchy'  
os.makedirs(destination_directory, exist_ok=True)

class_labels = data.columns[2:]  
for label in class_labels:
    label_folder = os.path.join(destination_directory, label)
    os.makedirs(label_folder, exist_ok=True)

for index, row in data.iterrows():
    filename = row['filename']
    is_unlabeled = row[' Unlabeled']

    if is_unlabeled == 1:
        continue
    image_file_path = os.path.join('Ocular_Disease_Classification/train', filename)
    image_labels = [label for label in class_labels if row[label] == 1]

    if len(image_labels) == 0:
        continue  
    destination_folder = os.path.join(destination_directory, image_labels[0]) 
    destination_path = os.path.join(destination_folder, filename)
    shutil.copy(image_file_path, destination_path)