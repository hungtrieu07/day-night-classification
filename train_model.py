# USAGE
# python train_model.py --dataset images --model model/day_night.h5

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.optimizers import Adam
from keras.src.optimizers.schedules import ExponentialDecay
from sklearn.model_selection import train_test_split
from keras.src.utils import img_to_array
from keras.src.utils import to_categorical
from utils.resnet50 import ResNet50  # Assuming you have ResNet50 implemented in utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
ap.add_argument("-e", "--epochs", type=int, default=140, required=False,
                help="number of epochs to train")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize training parameters
EPOCHS = args["epochs"]
INIT_LR = 1e-3
BS = 32

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (224, 224))  # ResNet50-compatible size
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the labels list
    label = imagePath.split(os.path.sep)[-2]
    label = 1 if label == "day" else 0
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% for training
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.25, random_state=42)

# convert the labels from integers to one-hot encoded vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# construct the image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# build the ResNet50 model for binary classification
print("[INFO] compiling model...")
model = ResNet50.build(width=224, height=224, depth=3, classes=2)

# Define the learning rate schedule
lr_schedule = ExponentialDecay(
    initial_learning_rate=INIT_LR,
    decay_steps=len(trainX) // BS,
    decay_rate=0.96,
    staircase=True
)

# compile the model
opt = Adam(learning_rate=lr_schedule)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS,
    verbose=1
)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Day/Night")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
