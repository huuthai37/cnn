import keras
import sys
from keras.models import Model
from keras.layers import Dense, Conv2D, Activation, Reshape, Flatten
from keras import optimizers
import get_data as gd
import pickle
import random

# train: python mobilenet_spatial.py train 32 1 101
# test: python mobilenet_spatial.py test 32 1 101
# retrain: python mobilenet_spatial.py retrain 32 1 101 1
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

server = False
if server:
    if train:
        out_file = '/home/oanhnt/thainh/data/database/train-opt.pickle'
    else:
        out_file = '/home/oanhnt/thainh/data/database/test-opt.pickle'
else:
    if train:
        out_file = '/mnt/smalldata/database/train-opt.pickle'
    else:
        out_file = '/mnt/smalldata/database/test-opt.pickle'

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
x = Flatten()(model.layers[-4].output)
x = Dense(classes, activation='softmax', name='predictions')(x)

#Then create the corresponding model 
result_model = Model(input=model.input, output=x)

result_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

if train:
    if retrain:
        result_model.load_weights('weights/mobilenet_spatial_{}e.h5'.format(old_epochs))

    with open(out_file,'rb') as f1:
        keys = pickle.load(f1)
    len_samples = len(keys)
    print('-'*40)
    print('MobileNet RGB stream only: Training')
    print('-'*40)
    print 'Number samples: {}'.format(len_samples)
    
    for e in range(epochs):
        print('-'*40)
        print('Epoch', e+1)
        print('-'*40)

        random.shuffle(keys)

        result_model.fit_generator(gd.getTrainData(keys,batch_size,classes,1,train), verbose=1, max_queue_size=2, steps_per_epoch=len_samples/batch_size, epochs=1)
        result_model.save_weights('weights/mobilenet_spatial_{}e.h5'.format(old_epochs+1+e))
else:
    result_model.load_weights('weights/mobilenet_spatial_{}e.h5'.format(epochs))
    
    with open(out_file,'rb') as f2:
        keys = pickle.load(f2)
    len_samples = len(keys)
    print('-'*40)
    print('MobileNet RGB stream only: Testing')
    print('-'*40)
    print 'Number samples: {}'.format(len_samples)

    # random.shuffle(keys)
    score = result_model.evaluate_generator(gd.getTrainData(keys,batch_size,classes,1,train), max_queue_size=3, steps=len_samples/batch_size)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])