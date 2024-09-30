# Dog vs Cat Classification using Transfer Learning

In this project, we will be building a dog vs. cat classification system using Transfer Learning. 

**Transfer Learning** is a type of learning in which we use a pre-trained model, which is already trained to perform one task, as a starting point to perfom a **similar task**. By using the learned features from the first task as a starting point, the pre-trained model can learn more quickly and effectively on the second task with a smaller dataset.

Transfer learning gives more accuracy compared to training models from scratch.

We will be using **MobileNetV2** model for building our classification system.

## Code by Code Explaination :-

### 1. Configuring the path of Kaggle.json file
```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```
This block of code is for setting up access to the Kaggle API using a Kaggle authentication token which is present in the *kaggle.json* file.

2.`!mkdir -p ~/.kaggle`: It creates a hidden directory named *kaggle* in my home directory (`~`) if it doesn't already exist. The `-p` option ensures that no error is thrown if the directory already exists.

1.`!cp kaggle.json ~/.kaggle/` : It copies the *kaggle.json* file that I have already uploaded into the newly created *kaggle* folder.

3.`!chmod 600 ~/.kaggle/kaggle.json` : It changes the permissions of the kaggle.json file so that only the owner can read and write it.

**`600` Permission Mode:**
The number `600` is a permission code that defines the access level for the owner, group, and others:
- **6** for the owner: This means the owner can read (4) and write (2) the file. (4 + 2 = 6)
- **0** for the group: No one in the group can read, write, or execute the file.
- **0** for others: No other users can read, write, or execute the file.

### 2. Downloading the Dog vs Cat dataset
```python
!kaggle competitions download -c dogs-vs-cats
```
The initial `kaggle` is a Kaggle API command-line interface tool. It is used to directly interact with Kaggle for various activies like downloading datasets, submitting models, competitions, etc.

`competitions download ` is a subcommand of the Kaggle CLI, used to download files related to a specific Kaggle competition.

Using `-c dogs-vs-cats`, we are specifying the exact competition we are referring to by providing the competition's name.

Once this code is executed a zip file named `dogs-vs-cats.zip` will be downloaded in our environment. 

### 3. Extracting the compressed dataset
```python
from zipfile import ZipFile

dataset = '/content/dogs-vs-cats.zip'

with ZipFile(dataset, 'r') as zip:
  zip.extractall()
  print("Extracted")
```
The `dogs-vs-cats.zip` compresssed file contains 3 files, `sampleSubmission.csv`, `test1.zip` and `train.zip`. Once we run this block of code, all the 3 files will be extracted to our home directory.

### 4. Extracting the train and test data
```python
dataset = '/content/test1.zip'
with ZipFile(dataset, 'r') as zip:
  zip.extractall()

dataset = '/content/train.zip'
with ZipFile(dataset, 'r') as zip:
  zip.extractall()
```
Here, we are extracting the `test1` folder and `train` folder from `test1.zip` and `train.zip` respectively. The `train` folder contains 25000 labelled images of dogs and cats, and `test1` contains 12500 unlabelled images of dogs and cats.

### 5. Importing all the modules
```python
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from google.colab.patches import cv2_imshow
```
Here we are just importing all the neccessary modules for the image and data handling and pre processing.

### 6. Counting the number of images in test folder
```python
file_names = os.listdir('/content/test1')
print(file_names)

print("Number of Images :", len(file_names))
```

`os.listdir()` returns a list containing the names of all files and directories in the specified path. Here we have specified the `test1` folder path and hence, it will return a list containing names of all the files present inside `test1` and save it in `file_name` variable. Then we are just printing the total length of the list, which will give us the total number of files present inside `test1`.

### 7. Displaying Images of Dogs and Cats
```python
img = mpimg.imread('/content/train/dog.11636.jpg')
plt.imshow(img)

img = mpimg.imread('/content/train/dog.11636.jpg')
plt.imshow(img)
```
`mpimg.imread()` function is a `matplotlib.image` function which is used to convert the specified image into an array format. Here, each element of the array represents a pixel in the image.

We store the array format of the image `dog.11636.jpg` inside the variable `img`. 

The function `plt.imshow()` will take the array format of the image as the input and render the image accordingly.

### 8. Counting the number of dogs and cats images in the train folder
```python
train_fld = os.listdir('/content/train/')

dogs, cats = 0, 0

for img in train_fld:
  name = img[0:3]
  if name == "dog":
    dogs += 1
  else:
    cats +=1

print("Number of Dogs are :", dogs)
print("Number of Cats are :", cats)
```

In this block of code, we first save the names of all the images present inside the `train` folder in the variable `train_fld` in the form of list. After that we create 2 counter variable `dog` and `cat` and initialize it to 0. Finally, we just iterate through the list and fetch the first 3 characters of the image name and check whether the `name` is equal to *dog* or *cat*, and then increment their corresponding counter variables.

### 9. Resizing 2000 Images
Since all the images are of different dimensions, we have to first resize it to same size for the neural network to find patterns between these images.

```python
os.mkdir('/content/resized_image/')

original_folder_path = '/content/train/'
resized_folder_path = '/content/resized_image/'

img_arr = os.listdir('/content/train/')

for i in range(2000):

  image_name = img_arr[i]
  image_path = original_folder_path + image_name

  img = Image.open(image_path)
  img = img.resize((224, 224))
  img = img.convert("RGB")

  new_image_path = resized_folder_path + image_name
  img.save(new_image_path)
```

First, we are creating a new folder called `resized_image` to store all our resized images.

Then, we create 2 path variables `original_folder_path` and `resized_folder_path`, to store the path of the `train` folder and the newly created `resized_image` folder.

Then (again, just for this block of code), we will create a list containing all the names of all the images present in the `train` folder.

Finally, we iterate through the first 2000 images and resize it all to `(224, 224)`. It's going to be super lengthy to explain every line of code, so I'm just gonna brief it  :))

First we create the original image path my adding the folder path and the image name. Then we open and load that particular image using `Image.open()` function into an `Image` object, `img`. Then we resize the image into *224x244* pixels, converts the image to the RGB color mode (if it's in another format like grayscale). After that, we create the new path for the image by adding the `resized_folder_path` and the image name together and  finally, we save the image (object) in that newly created path.
