import keras
import sys
import time
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Conv2D, Activation, Reshape, Flatten
from keras import optimizers
# from xception import XceptionFix

# xception_spatical.py train 32 1 101
if sys.argv[1] == 'train':
    train = True
else:
    train = False

train_path = 'home/oanhnt/thainh/data/rgb/train'
test_path = 'home/oanhnt/thainh/data/rgb/test'
epochs = int(sys.argv[3])
batch_size = int(sys.argv[2])
classes = int(sys.argv[4])

if train:
    train_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(
        train_path,
        target_size=(224, 224),
        batch_size=batch_size
    )
else:
    test_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(
        test_path,
        target_size=(224, 224),
        batch_size=batch_size
    )

# MobileNet model
if train:
    model = keras.applications.xception.Xception(
        include_top=True,  
        weights='imagenet'
    )
else:
    model = keras.applications.xception.Xception(
        include_top=True,  
    )
# model.summary()

# Non-trainable layers, only train last layer
# for layer in model.layers: layer.trainable = False

# Modify network some last layer

# Non-trainable layers, only train last layer
# for layer in model.layers: layer.trainable = False

# Add a layer where input is the output of the  second last layer 
x = Dense(classes, activation='softmax', name='predictions')(model.layers[-2].output)

#Then create the corresponding model 
my_model = Model(input=model.input, output=x)
# my_model.summary()

my_model.compile(loss='mean_squared_error',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

if train:
    my_model.fit_generator(
        train_batches,
        epochs=epochs
    )
    my_model.save_weights('xception_spatial_{}e.h5'.format(epochs))
else:
    my_model.load_weights('xception_spatial_{}e.h5'.format(epochs))
    score = my_model.evaluate_generator(test_batches)
    # with open('spatial_result.txt', 'a') as the_file:
    #     the_file.write('Test loss:', score[0])
    #     the_file.write('Test accuracy:', score[1])
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
