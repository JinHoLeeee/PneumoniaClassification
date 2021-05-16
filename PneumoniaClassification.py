# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 16:59:05 2020

@author: USER
"""

import os
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator


base='E:\\dl\\chest-xray-pneumonia\\chest_xray\\'

train_dir=os.path.join(base,'train')
test_dir=os.path.join(base,'test')
val_dir=os.path.join(base,'val')

train_normal_dir=os.path.join(train_dir,'NORMAL')
train_pn_dir=os.path.join(train_dir,'PNEUMONIA')

test_normal_dir=os.path.join(test_dir,'NORMAL')
test_pn_dir=os.path.join(test_dir,'PNEUMONIA')

val_normal_dir=os.path.join(val_dir,'NORMAL')
val_pn_dir=os.path.join(val_dir,'PNEUMONIA')

print("Num of train_normal : ",len(os.listdir(train_normal_dir)))
print("Num of train_pn : ",len(os.listdir(train_pn_dir)))
print("Num of test_normal : ",len(os.listdir(test_normal_dir)))
print("Num of test_pn : ",len(os.listdir(test_pn_dir)))
print("Num of val_normal : ",len(os.listdir(val_normal_dir)))
print("Num of val_pn : ",len(os.listdir(val_pn_dir)))

train_datagen=ImageDataGenerator(rescale=1.0/255.)
test_datagen=ImageDataGenerator(rescale=1.0/255.)
val_datagen=ImageDataGenerator(rescale=1.0/255.)

train_generator=train_datagen.flow_from_directory(train_dir,
                                                  batch_size=20,
                                                  class_mode='binary',
                                                  target_size=(150,150))
test_generator=test_datagen.flow_from_directory(test_dir,
                                                batch_size=20,
                                                class_mode='binary',
                                                target_size=(150,150))
val_generator=val_datagen.flow_from_directory(val_dir,
                                              batch_size=20,
                                              class_mode='binary',
                                              target_size=(150,150))




model=keras.models.Sequential([
    keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(32,(3,3),activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64,(3,3),activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')])

model.summary()

model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

#train_datagen=ImageDataGenerator(rescale=1.0/255.)


train_datagen=ImageDataGenerator(rescale=1.0/255.,
                                 rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest')

test_datagen=ImageDataGenerator(rescale=1.0/255.,
                                rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest')

val_datagen=ImageDataGenerator(rescale=1.0/255.,
                               rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest')

#test_datagen=ImageDataGenerator(rescale=1.0/255.)
#val_datagen=ImageDataGenerator(rescale=1.0/255.)

train_generator=train_datagen.flow_from_directory(train_dir,
                                                  batch_size=20,
                                                  class_mode='binary',
                                                  target_size=(150,150))
test_generator=test_datagen.flow_from_directory(test_dir,
                                                batch_size=20,
                                                class_mode='binary',
                                                target_size=(150,150))
val_generator=val_datagen.flow_from_directory(val_dir,
                                              batch_size=20,
                                              class_mode='binary',
                                              target_size=(150,150))


history=model.fit(train_generator,
                  validation_data=val_generator,
                  steps_per_epoch=100,
                  epochs=100,
                  validation_steps=50,
                  verbose=2)
 
scores=model.evaluate(test_generator)
print("loss: %.2f" %scores[0])
print("test acc: %.2f" % scores[1]) 

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc))

plt.plot(epochs,acc,'bo',label='Training accuracy')
plt.plot(epochs,val_acc,'b',label='validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'go', label="training loss")
plt.plot(epochs, val_loss,'g',label='validation loss')
plt.title("Training and validation loss")
plt.legend()

plt.show()

model.summary()

model.save('C:\\Users\\USER\\Desktop\\2020-2\\deeplearning\\32183631_jinholee.h5')

