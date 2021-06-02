"""
    TODO: Read all images from raw/road and raw/mtb
    TODO: Apply random transformations to generate "new" images
    TODO: Generate class attribute csv file
    TODO: Split into test/train datasets and move to bikes_test and bikes_train
"""
from PIL import Image
import os
import random
import csv

class_dirs = ["data/raw/road", "data/raw/mtb"]
samples = 16

def RandomHorizontalFlip(image):
    if(random.random() > 0.5):
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image

def RandomVerticalFlip(image):
    if(random.random() > 0.5):
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return image

def RandomCrop(image, size):
    x, y = image.size
    x1 = random.randrange(0, x-size)
    y1 = random.randrange(0, y-size)
    return image.crop((x1, y1, x1+size, y1+size))


train_writer = csv.writer(open('data/bikes_train/bikes.csv', 'w', newline=''))
# test_writer = csv.writer(open('data/bikes_test/bikes.csv', 'w', newline=''))

for cl in range(len(class_dirs)):
    dir = class_dirs[cl]
    files = os.listdir(dir)
    for img_name in files:
        img_raw = Image.open(dir + "/" + img_name)
        img_resize = img_raw.resize((350, 350))
        for i in range(samples):
            img = RandomHorizontalFlip(img_resize)
            img = RandomVerticalFlip(img)
            img = RandomCrop(img, 224)
            filename = str(i) + "-" + img_name
            train_writer.writerow([filename, cl])
            img.save("data/bikes_train/" + filename)

        # test_writer.writerow([img_name, cl]) 
        # img_raw.save("data/bikes_test/" + img_name)


