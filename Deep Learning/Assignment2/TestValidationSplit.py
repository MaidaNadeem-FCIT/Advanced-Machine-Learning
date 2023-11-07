import os
import random
import shutil

dataset = 'Ocular_Disease_Classification/TrainHierarchy'  
train = 'Ocular_Disease_Classification/Train_Test_Validate/train'
validation = 'Ocular_Disease_Classification/Train_Test_Validate/validate'
test = 'Ocular_Disease_Classification/Train_Test_Validate/test'

train_ratio = 0.48
validation_ratio = 0.26
test_ratio = 0.26

os.makedirs(train, exist_ok=True)
os.makedirs(validation, exist_ok=True)
os.makedirs(test, exist_ok=True)

class_folders = os.listdir(dataset)

for class_folder in class_folders:
    class_path = os.path.join(dataset, class_folder)
    images = os.listdir(class_path)
    random.shuffle(images)

    num_images = len(images)
    num_train = int(train_ratio * num_images)
    num_validation = int(validation_ratio * num_images)

    train_images = images[:num_train]
    validation_images = images[num_train:num_train + num_validation]
    test_images = images[num_train + num_validation:]

    for image in train_images:
        src = os.path.join(class_path, image)
        dst = os.path.join(train, class_folder, image)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

    for image in validation_images:
        src = os.path.join(class_path, image)
        dst = os.path.join(validation, class_folder, image)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

    for image in test_images:
        src = os.path.join(class_path, image)
        dst = os.path.join(test, class_folder, image)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)