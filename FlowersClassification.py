#source: https://www.kaggle.com/olgabelitskaya/the-dataset-of-flower-images

import numpy as np
import pandas as pd

import cv2
import scipy
from skimage import io

from PIL import ImageFile
from tqdm import tqdm

import matplotlib.pylab as plt
from matplotlib import cm

from keras.preprocessing import image as keras_image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocess_input

from keras.callbacks import ModelCheckpoint

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, BatchNormalization
from keras.layers import Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

DATA_PATH = "input/flower_images/"
INCEPTIONV3_PATH = 'input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'


def loss_plot(fit_history):
    plt.figure(figsize=(18, 4))

    plt.plot(fit_history.history['loss'], label='train')
    plt.plot(fit_history.history['val_loss'], label='test')

    plt.legend()
    plt.title('Loss Function');
    plt.show()


def acc_plot(fit_history):
    plt.figure(figsize=(18, 4))

    plt.plot(fit_history.history['acc'], label='train')
    plt.plot(fit_history.history['val_acc'], label='test')

    plt.legend()
    plt.title('Accuracy');
    plt.show()


def path_to_tensor(img_path):
    img = keras_image.load_img(DATA_PATH + img_path,
                               target_size=(128, 128))
    x = keras_image.img_to_array(img)
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

ImageFile.LOAD_TRUNCATED_IMAGES = True

flowers = pd.read_csv("input/flower_images/flower_labels.csv")
flower_files = flowers['file']
flower_targets = flowers['label'].values

print('Label: ', flower_targets[168])
flower_image = cv2.imread("input/flower_images/"+flower_files[168])
rgb_flower_image = cv2.cvtColor(flower_image, cv2.COLOR_BGR2RGB)
#plt.figure(figsize=(3,3))
#plt.imshow(rgb_flower_image);
#plt.show()

flower_tensors = paths_to_tensor(flower_files);

x_train, x_test, y_train, y_test = train_test_split(flower_tensors, flower_targets,
                                                    test_size = 0.2, random_state = 1)

n = int(len(x_test)/2)
x_valid, y_valid = x_test[:n], y_test[:n]
x_test, y_test = x_test[n:], y_test[n:]

print('Label: ', y_train[1])
#plt.figure(figsize=(3,3))
#plt.imshow((x_train[1]/255).reshape(128,128,3));
#plt.show()

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
x_valid = x_valid.astype('float32')/255

c_y_train = to_categorical(y_train, 10)
c_y_test = to_categorical(y_test, 10)
c_y_valid = to_categorical(y_valid, 10)


# MLP
def mlp_mc_model():
    model = Sequential()

    model.add(Dense(128, activation='relu', input_shape=(128 * 128 * 3,)))
    model.add(BatchNormalization())

    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


mlp_mc_model = mlp_mc_model()

mlp_mc_history = mlp_mc_model.fit(x_train.reshape(-1, 128*128*3), c_y_train,
                                  validation_data=(x_valid.reshape(-1, 128*128*3), c_y_valid),
                                  epochs=50, batch_size=64, verbose=2)

loss_plot(mlp_mc_history)
acc_plot(mlp_mc_history)

mlp_mc_test_score = mlp_mc_model.evaluate(x_test.reshape(-1, 128*128*3), c_y_test)
print(mlp_mc_test_score)


# CNN
def cnn_mc_model():
    model = Sequential()

    model.add(Conv2D(32, (5, 5), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(96, (5, 5)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #     model.add(Flatten())
    model.add(GlobalAveragePooling2D())

    model.add(Dense(512, activation='tanh'))
    model.add(Dropout(0.25))

    #    model.add(Dense(256, activation='tanh'))
    #    model.add(Dropout(0.25))

    model.add(Dense(128, activation='tanh'))
    model.add(Dropout(0.25))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

    return model


cnn_mc_model = cnn_mc_model()

cnn_mc_history = cnn_mc_model.fit(x_train, c_y_train,
                                  epochs=50, batch_size=64, verbose=2,
                                  validation_data=(x_valid, c_y_valid))

loss_plot(cnn_mc_history)
acc_plot(cnn_mc_history)

cnn_mc_test_score = cnn_mc_model.evaluate(x_test, c_y_test)
print(cnn_mc_test_score)

data_generator = keras_image.ImageDataGenerator(shear_range=0.3,
                                                zoom_range=0.3,
                                                rotation_range=30,
                                                horizontal_flip=True)
cnn_mc_dg_history = cnn_mc_model.fit_generator(data_generator.flow(x_train, c_y_train, batch_size=64),
                                               steps_per_epoch=189, epochs=3, verbose=2,
                                               validation_data=(x_valid, c_y_valid))

cnn_mc_test_score = cnn_mc_model.evaluate(x_test, c_y_test)
print(cnn_mc_test_score)


# RNN
def rnn_mc_model():
    model = Sequential()

    model.add(LSTM(128, return_sequences=True, input_shape=(1, 128 * 128 * 3)))
    #    model.add(LSTM(128, return_sequences=True))

    model.add(LSTM(128))
    model.add(Dense(512, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


rnn_mc_model = rnn_mc_model()

rnn_mc_history = rnn_mc_model.fit(x_train.reshape(-1,1,128*128*3), c_y_train,
                                  epochs=16, batch_size=64, verbose=2,
                                  validation_data=(x_valid.reshape(-1,1,128*128*3), c_y_valid))

loss_plot(rnn_mc_history)
acc_plot(rnn_mc_history)

rnn_mc_test_score = rnn_mc_model.evaluate(x_test.reshape(-1,1,128*128*3), c_y_test)
print(rnn_mc_test_score)

# InceptionV3
iv3_base_model = InceptionV3(weights=INCEPTIONV3_PATH,
            include_top=False)
x = iv3_base_model.output

x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)

y = Dense(10, activation='softmax')(x)

iv3_model = Model(inputs=iv3_base_model.input, outputs=y)

# Freeze InceptionV3 convolutional layers
for layer in iv3_base_model.layers:
    layer.trainable = False

iv3_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

steps, epochs = 150, 7
data_generator = keras_image.ImageDataGenerator(shear_range=0.2,
                                                zoom_range=0.2,
                                                horizontal_flip=True)

iv3_history = iv3_model.fit_generator(data_generator.flow(x_train, c_y_train, batch_size=64),
                                      steps_per_epoch=steps, epochs=epochs, verbose=2,
                                      validation_data=(x_valid, c_y_valid))

# Unfreeze the layers [173:]
for layer in iv3_model.layers[:173]:
    layer.trainable = False
for layer in iv3_model.layers[173:]:
    layer.trainable = True

iv3_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

iv3_history_unfreeze = iv3_model.fit_generator(data_generator.flow(x_train, c_y_train, batch_size=64),
                                               steps_per_epoch=50, epochs=epochs, verbose=2,
                                               validation_data=(x_valid, c_y_valid))

#iv3_model.load_weights('weights.best.iv3.flowers.hdf5')
iv3_test_scores = iv3_model.evaluate(x_test, c_y_test)
print("Accuracy: %.2f%%" % (iv3_test_scores[1]*100))