import keras
import sys
from keras.models import Model
from keras.layers import Dense, Conv2D, Activation, Reshape, Flatten
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import pickle
import random
import config

# from matplotlib import pyplot as plt
# from IPython.display import clear_output

# train: python mobilenet_spatial.py train 32 1 101
# test: python mobilenet_spatial.py test 32 1 101
# retrain: python mobilenet_spatial.py retrain 32 1 101 1
class PlotLearning(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        # self.i = 0
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        # self.i+=1
        with open('data/trainHistorySpatial', 'wb') as file_pi:
            pickle.dump(self.logs, file_pi)

plot_losses = PlotLearning()

if sys.argv[1] == 'train':
    train = True
    retrain = False
    old_epochs = 0
elif sys.argv[1] == 'retrain':
    train = True
    retrain = True
    old_epochs = int(sys.argv[5])
else:
    train = False
    retrain = False

batch_size = int(sys.argv[2])
epochs = int(sys.argv[3])
classes = int(sys.argv[4])
len_samples = 430288
server = config.server()
if server:
    train_path = '/home/oanhnt/thainh/data/rgb/train'
    test_path = '/home/oanhnt/thainh/data/rgb/test'
else:
    train_path = '/mnt/data11/rgb/train'
    test_path = '/mnt/data11/rgb/test'

if train:
    train_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(
        train_path,
        batch_size=batch_size,
        target_size=(224, 224),
    )
else:
    test_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(
        test_path,
        batch_size=batch_size,
        target_size=(224, 224),
    )

# MobileNet model
if train & (not retrain):
    model = keras.applications.mobilenet.MobileNet(
        include_top=True,  
        weights='imagenet'
    )
else:
    model = keras.applications.mobilenet.MobileNet(
        include_top=True,  
    )

# Modify network some last layer
x = Conv2D(classes, (1, 1),
                   padding='same', name='conv_preds')(model.layers[-4].output)
x = Activation('softmax', name='act_softmax')(x)
x = Reshape((classes,), name='reshape_2')(x)

#Then create the corresponding model 
result_model = Model(input=model.input, output=x)
# result_model.summary()
result_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])


if train:
    if retrain:
        result_model.load_weights('weights/mobilenet_spatial_{}e.h5'.format(old_epochs))

    
    print('-'*40)
    print('MobileNet RGB stream only: Training')
    print('-'*40)
    
    result_model.fit_generator(train_batches, verbose=1, callbacks=[plot_losses], max_queue_size=2, steps_per_epoch=len_samples/batch_size, epochs=1)
    result_model.save_weights('weights/mobilenet_spatial_{}e.h5'.format(old_epochs+1+e))
else:
    result_model.load_weights('weights/mobilenet_spatial_{}e.h5'.format(epochs))

    print('-'*40)
    print('MobileNet RGB stream only: Testing')
    print('-'*40)

    # random.shuffle(keys)
    score = result_model.evaluate_generator(test_batches, max_queue_size=3, steps=len_samples/batch_size)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])