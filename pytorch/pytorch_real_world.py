#import all the helping libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#supporing library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#to open image file and extract data
import os
from PIL import Image
from IPython.display import display

import warnings
warnings.filterwarnings('ignore')
#setting the path for the data
#with Image.open('./data/CATS_DOGS/test/CAT/9394.jpg') as im:
#    display(im)
#    im.show()
#create the data frame for the images
path = ('./data/CATS_DOGS/')
img_names = []

for folder,subfolders,filenames in os.walk(path):
    for (img) in filenames:
        img_names.append(folder+'/'+img)
#print(len(img_names))
#to store the size of the images as the images are of variable length
img_sizes = []
rejected = []
for item in img_names:
    try:
        with Image.open(item) as img:
            img_sizes.append(img.size)

    except:
        rejected.append(item)

#print(img_sizes[0])
#conversion to the data frame
df = pd.DataFrame(img_sizes)
#print(df.head())
#getting the one image
dog = Image.open('./data/CATS_DOGS/train/DOG/14.jpg')
#dog.show()
print(dog.getpixel((0,0)))
#transfor the data
#we can pass in a varity of transformations via the list
transform = transforms.Compose([
transforms.ToTensor(),
#transforms.Resize((1000,1500)),
#transforms.Resize(250),
#transforms.CenterCrop(250),
#probablity can be comtrolled as p is the probablity
#transforms.RandomHorizontalFlip(p=1)
transforms.RandomRotation(30)

])
im = transform(dog)
#print(type(im))
#print(im.shape)

#plotting the imgae
#first we have to make the dimenson in the wrond dimension
#matplot lib wants a data that is in the shape like - (400,400,3) length, width,channel

#first convert them to numpy and then plot it
plt.imshow(np.transpose(im.numpy(),(1,2,0)))
plt.show()
