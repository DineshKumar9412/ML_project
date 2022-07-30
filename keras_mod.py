import keras.losses
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, GlobalMaxPool2D
from keras import Sequential
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
import numpy as np

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(224, 224, 3) ))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=36, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(rate= 0.25))

model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(rate= 0.25))
model.add(Dense(units= 1, activation='sigmoid'))

model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

def pre(path):
    img_data = ImageDataGenerator(zoom_range= 0.2, shear_range=0.2, rescale=1/255, horizontal_flip=True)
    img = img_data.flow_from_directory(directory=path, target_size= (224,224), batch_size= 32, class_mode= 'binary')
    return img
train_data = pre("/Users/harikumar/PycharmProjects/Deep_learning/venv/train")


def pres(path):
    img_data = ImageDataGenerator(rescale=1/255)
    img = img_data.flow_from_directory(directory=path, target_size= (224,224), batch_size= 32, class_mode= 'binary')
    return img
test_data = pres("/Users/harikumar/PycharmProjects/Deep_learning/venv/test")
val_data = pres("/Users/harikumar/PycharmProjects/Deep_learning/venv/val")

es = EarlyStopping(monitor="val_accuracy", min_delta= 0.01, patience= 3, verbose=1, mode='auto')
mc = ModelCheckpoint(monitor="val_accaracy", filepath="best.h5", verbose= 1, save_best_only=True, mode='auto')

cd = [es, mc]

hs = model.fit_generator(generator=train_data,
                         steps_per_epoch=8,
                         epochs= 10,
                         verbose=1,
                         validation_data=val_data,
                         validation_steps= 16)

model.save("name.h5")

# model = load_model("name.h5")

# path = "/Users/harikumar/PycharmProjects/Deep_learning/venv/train/yes/y0.jpg"
#
# img = load_img(path, target_size=(224,224))
# imf = img_to_array(img)/255
#
# # print(imf.shape)/
#
# imgh = np.expand_dims(imf, axis=0)
# prd = model.predict(imgh)
# print(prd)
# classes_x=np.argmax(prd,axis=1)
# print(classes_x)
