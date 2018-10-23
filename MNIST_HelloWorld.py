from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

def printDataInformation(train_images, train_labels, test_images, test_labels):
    print("Training set shape: %s" % str(train_images.shape))
    print("Training set size: %s" % len(train_labels))
    print("Training labels: %s" % train_labels)
    print("")
    print("Test set shape: %s" % str(test_images.shape))
    print("Test set size: %s" % len(test_labels))
    print("Test labels: %s" % test_labels)
    print("")

def generateModel():
    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))

    network.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return network

def transformImageData(images, length):
    images = images.reshape((length, 28 * 28))
    images = images.astype('float32') / 255
    return images

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
printDataInformation(train_images, train_labels, test_images, test_labels)

network = generateModel()

train_images = transformImageData(train_images, len(train_labels))
test_images = transformImageData(test_images, len(test_labels))

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

print("Start learning")
network.fit(train_images, train_labels, epochs=5, batch_size=128)
print("Learning has finished")

test_loss, test_acc = network.evaluate(test_images, test_labels)
print("Test accuracy: %s" % test_acc)

print("Hello World!")