import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.models import Sequential

np.random.seed(42)

# Training parameters
batch_size = 256
input_shape = (28, 28, 1)
num_classes = 10  # total classes (0-9 digits)

# Prepare MNIST data
# http://yann.lecun.com/exdb/mnist/
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Training set : ", x_train.shape)
print("Testing set : ", x_test.shape)

# Convert to float32
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Flatten images to 1-D vector of 784 features (28*28)
x_train = np.reshape(x_train, [x_train.shape[0], x_train.shape[1], x_train.shape[2], 1])
x_test = np.reshape(x_test, [x_test.shape[0], x_test.shape[1], x_test.shape[2], 1])
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# Normalize images value from [0, 255] to [0, 1]
x_train, x_test = x_train / 255., x_test / 255.

# Use tf.data API to shuffle and batch data
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)


def digModel(input_shape=(28, 28, 1)):
    '''
    X_input = Input(input_shape)
    X = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(X_input)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(X)

    X = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(X)

    X = Flatten()(X)
    X = Dense(1024, activation='relu', kernel_initializer='TruncatedNormal', bias_initializer='zeros')(X)
    X = Dense(num_classes, activation='softmax', kernel_initializer='TruncatedNormal', bias_initializer='zeros')(X)

    model = Model(inputs=X_input, outputs=X, name='digModel')
    '''

    number_of_classes = 10

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu',  padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Conv2D(64, (5, 5), activation='relu',  padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(number_of_classes, activation='softmax'))

    return model


model = digModel(input_shape=(28, 28, 1))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

hist = model.fit(x_train, y_train,
                 batch_size=512,
                 epochs=10,
                 validation_data=(x_test, y_test),
                 )

model.save_weights("model/weights.h5")
model.save("model/model.h5")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(hist.history["loss"], label="Training loss")
ax[0].plot(hist.history["val_loss"], label="Validation loss")
ax[0].legend()

ax[1].plot(hist.history["accuracy"], label="Training accuracy")
ax[1].plot(hist.history["val_accuracy"], label="Validation accuracy")
ax[1].legend()

plt.show()

# Load Model
model.load_weights("model/weights.h5")

score = model.evaluate(x_test, y_test)
score2 = model.evaluate(x_train, y_train)
print("Train Accuracy: " + str(score2[1] * 100) + "%")
print("Test Accuracy: " + str(score[1] * 100) + "%")
