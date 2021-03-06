import keras
import sys
from keras.models import Model
from keras.layers import Dense, Conv2D, Activation, Reshape, Flatten
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback
import pickle
import random
import config

# train: python mobilenet_spatial.py train 32 1 101
# test: python mobilenet_spatial.py test 32 1 101
# retrain: python mobilenet_spatial.py retrain 32 1 101 1
class PlotLearning(Callback):
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

server = config.server()
if server:
    train_path = '/home/oanhnt/thainh/data/rgb/train'
    valid_path = '/home/oanhnt/thainh/data/rgb/valid'
    test_path = '/home/oanhnt/thainh/data/rgb/test'
else:
    train_path = '/mnt/data11/rgb/train'
    valid_path = '/mnt/data11/rgb/valid'
    test_path = '/mnt/data11/rgb/test'

if train:
    len_samples = 71090
    train_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(
        train_path,
        batch_size=batch_size,
        target_size=(224, 224),
    )
    len_valid = 26392
    valid_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(
        valid_path,
        batch_size=batch_size,
        target_size=(224, 224),
    )
else:
    len_samples = 26114
    test_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(
        test_path,
        batch_size=batch_size,
        target_size=(224, 224),
    )

# MobileNet model
if train & (not retrain):
    model = keras.applications.mobilenet.MobileNet(
        include_top=True,  
        weights='imagenet',
        dropout=0.5
    )
else:
    model = keras.applications.mobilenet.MobileNet(
        include_top=True,  
        dropout=0.5
    )

# Modify network some last layer
x = Conv2D(classes, (1, 1),
                   padding='same', name='conv_preds')(model.layers[-4].output)
x = Activation('softmax', name='act_softmax')(x)
x = Reshape((classes,), name='reshape_2')(x)

#Then create the corresponding model 
result_model = Model(inputs=model.input, outputs=x)
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
    check_point = ModelCheckpoint('weights/mobilenet_spatial_{epoch}e.h5', verbose=1, save_weights_only=True)
    result_model.fit_generator(
        train_batches, 
        verbose=1, 
        callbacks=[plot_losses,check_point], 
        max_queue_size=2, 
        steps_per_epoch=len_samples/batch_size, 
        epochs=epochs,
        validation_data=valid_batches,
        validation_steps=len_valid/batch_size)

else:
    result_model.load_weights('weights/mobilenet_spatial_{}e.h5'.format(epochs))

    print('-'*40)
    print('MobileNet RGB stream only: Testing')
    print('-'*40)

    # random.shuffle(keys)
    score = result_model.evaluate_generator(test_batches, max_queue_size=3, steps=len_samples/batch_size)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])