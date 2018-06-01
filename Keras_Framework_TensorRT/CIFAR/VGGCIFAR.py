
# import modules
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.datasets import cifar10
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPool2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils, plot_model

from matplotlib.ticker import MultipleLocator

from sklearn.model_selection import train_test_split

# consts
NUM_CLASSES = 10
BATCH_SIZE = 200


# plot function
def plot_history(history):
    # plot accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    plt.show()

    # plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(20))
    plt.show()


# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, NUM_CLASSES)
y_test = np_utils.to_categorical(y_test, NUM_CLASSES)

np.save("x_cifar_train.npy",x_train)
np.save("x_cifar_test.npy",x_test)
np.save("Y_cifar_train.npy",y_train)
np.save("y_cifar_test.npy",y_test)

# split data into training and valuation
x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    test_size = 0.1,
    train_size=0.9,
    random_state=42,
    shuffle=True
)

# data augmentation & normalization

#training data
trainGenerator = ImageDataGenerator(
    featurewise_center=True,
    #featurewise_std_normalization=True,
    zca_whitening=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

trainGenerator.fit(x_train)

# valuation data
valGenerator = ImageDataGenerator(
    featurewise_center=True,
    #featurewise_std_normalization=True,
    zca_whitening=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

valGenerator.fit(x_val)
# definition
model = Sequential([
    Conv2D(64, 3, padding='same', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(64, 3, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, 3, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(128, 3, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(256, 3, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(256, 3, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(256, 3, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(256, 3, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dropout(0.25),
    Dense(4 * 4 * 256),
    Activation('relu'),

    Dropout(0.25),
    Dense(1024),
    Activation('relu'),

    Dense(NUM_CLASSES),
    Activation('softmax')
])

# optimizer
optimizer = Adam()

# compile
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Epoch
EPOCHS = 100


# learning
history = model.fit_generator(
    trainGenerator.flow(x_train, y_train, BATCH_SIZE),
    epochs=EPOCHS,
    verbose=1,
    #callbacks=[earlyStopping],
    validation_data=valGenerator.flow(x_val, y_val, BATCH_SIZE)
)

# plot accuracy & loss
plot_history(history)

# visualize model
plot_model(model, to_file="./Resources/model.png", show_shapes=True)

# save model & weight
json_string = model.to_json()
open('./Resources/model.json', 'w').write(json_string)

model.save_weights('./Resources/model_weights.hdf5')