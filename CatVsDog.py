import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation,BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import tensorflow as tf
import random



path = 'C:\\Users\\duong\\Downloads\\archive\\training_set\\training_set'
train_datagen = ImageDataGenerator(rescale=1. / 255)
train = train_datagen.flow_from_directory(path, target_size=(50,50), batch_size= 32, class_mode='binary')

def CNN(input_shape):
    X_input = Input(input_shape)

    X = Conv2D(64, (3, 3), strides=4)(X_input)
    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=2)(X)

    X = Conv2D(64, (5, 5), padding='same')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=2, name='max1')(X)

    X = Conv2D(384, (3, 3), padding='same', name='conv2')(X)
    X = Activation('relu')(X)

    # X = Conv2D(384, (3, 3), padding='same', name='conv3')(X)
    # X = BatchNormalization()(X)#axis=3, name='bn3')(X)
    # X = Activation('relu')(X)

    # X = Conv2D(256, (3, 3), padding='same', name='conv4')(X)
    # X = BatchNormalization()(X)#axis=3, name='bn4')(X)
    # X = Activation('relu')(X)
    #
    # X = MaxPooling2D((3, 3), strides=2, name='max2')(X)

    X = Flatten()(X)

    X = Dense(512, activation='relu')(X)

    # X = Dense(4096, activation='relu', name='fc1')(X)

    X = Dense(1, activation='sigmoid', name='fc2')(X)

    model = Model(inputs=X_input, outputs=X, name='AlexNet')

    return model

model = CNN(train[0][0].shape[1:])
model.compile(optimizer = tf.keras.optimizers.Adam(0.001) , loss = 'binary_crossentropy' , metrics=['accuracy'])
history = model.fit_generator(train,epochs=20)

path_test = 'C:\\Users\\duong\\Downloads\\archive\\test_set\\test_set'
test_datagen = ImageDataGenerator(rescale=1. / 255)
test = test_datagen.flow_from_directory(path_test, target_size=(50,50), class_mode='binary')

preds = model.evaluate_generator(test)

print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

predictions = model.predict_generator(test)

import os
def get_category(predicted_output):
    path  = "C:\\Users\\duong\\Downloads\\archive\\test_set\\test_set"
    return os.listdir(path)[np.argmax(predicted_output)]


plt.plot(history.history['loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

fig, axs = plt.subplots(2,2 ,figsize = (10,10))

axs[0][0].imshow(test[10][0][0])
axs[0][0].set_title(get_category(predictions[10]))

axs[0][1].imshow(test[22][0][0])
axs[0][1].set_title(get_category(predictions[22]))

axs[1][0].imshow(test[44][0][0])
axs[1][0].set_title(get_category(predictions[44]))

axs[1][1].imshow(test[52][0][0])
axs[1][1].set_title(get_category(predictions[52]))




plt.show()














