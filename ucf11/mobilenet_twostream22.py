import keras
import sys
from keras.models import Model
from keras.layers import Dense, Conv2D, Activation, Reshape, Flatten, Input, ZeroPadding2D, Average
import get_data as gd
from keras import optimizers
import pickle
import random
import numpy as np
import config

# train: python mobilenet_two_stream.py train 32 1 101 0 0
# test: python mobilenet_two_stream.py test 32 1 101
# retrain: python mobilenet_two_stream.py retrain 32 1 101 1
if sys.argv[1] == 'train':
    train = True
    retrain = False
    old_epochs = 0
    spa_epochs = int(sys.argv[5])
    tem_epochs = int(sys.argv[6])
elif sys.argv[1] == 'retrain':
    train = True
    retrain = True
    old_epochs = int(sys.argv[5])
else:
    train = False
    retrain = False

opt_size = 2
batch_size = int(sys.argv[2])
epochs = int(sys.argv[3])
classes = int(sys.argv[4])

depth = 20
input_shape = (224,224,depth)

server = config.server()
if server:
    if train:
        out_file = '/home/oanhnt/thainh/data/database/train-opt2.pickle'
    else:
        out_file = '/home/oanhnt/thainh/data/database/test-opt2.pickle'
    valid_file = r'/home/oanhnt/thainh/data/database/valid-opt2.pickle'
else:
    if train:
        out_file = '/mnt/smalldata/database/train-opt2.pickle'
    else:
        out_file = '/mnt/smalldata/database/test-opt2.pickle'

# two_stream
model = keras.applications.mobilenet.MobileNet(
    include_top=True,
    dropout=0.5
)

# Disassemble layers
layers = [l for l in model.layers]

input_opt = Input(shape=input_shape)
x = ZeroPadding2D(padding=(1, 1), name='conv1_padx')(input_opt)
x = Conv2D(filters=32, 
          kernel_size=(3, 3),
          padding='valid',
          use_bias=False,
          strides=(2,2),
          name='conv_new')(x)

for i in range(3, len(layers)-3):
    layers[i].name = str(i)
    x = layers[i](x)

x = Flatten()(x)
x = Dense(classes, activation='softmax')(x)
temporal_model = Model(inputs=input_opt, outputs=x)
if train & (not retrain):
    temporal_model.load_weights('weights/mobilenet_temporal22_{}e.h5'.format(tem_epochs))

# Spatial
model2 = keras.applications.mobilenet.MobileNet(
    include_top=True,
    input_shape=(224,224,3),
    dropout=0.5
)

y = Flatten()(model.layers[-4].output)
y = Dense(classes, activation='softmax')(y)
spatial_model = Model(inputs=model2.input, outputs=y)
if train & (not retrain):
    spatial_model.load_weights('weights/mobilenet_spatial2_{}e.h5'.format(spa_epochs))

# Fusion
z = Average()([y, x])

# Final touch
result_model = Model(inputs=[model2.input,input_opt], outputs=z)

# Run
result_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

if train:
    if retrain:
        result_model.load_weights('weights/mobilenet_twostream22_{}e.h5'.format(old_epochs))

    with open(out_file,'rb') as f1:
        keys = pickle.load(f1)
    len_samples = len(keys)
    if server:
        with open(valid_file,'rb') as f2:
            keys_valid = pickle.load(f2)
        len_valid = len(keys_valid)

    print('-'*40)
    print 'MobileNet Optical #{} stream only: Training'.format(opt_size)
    print('-'*40)
    print 'Number samples: {}'.format(len_samples)
    if server:
        print 'Number valid: {}'.format(len_valid)
    histories = []
    
    for e in range(epochs):
        print('-'*40)
        print('Epoch', e+1)
        print('-'*40)

        random.shuffle(keys)
        if server:
            history = result_model.fit_generator(
                gd.getTrainData(keys,batch_size,classes,5,'train'), 
                verbose=1, 
                max_queue_size=2, 
                steps_per_epoch=len_samples/batch_size, 
                epochs=1,
                validation_data=gd.getTrainData(keys_valid,batch_size,classes,5,'valid'),
                validation_steps=len_valid/batch_size
            )
            histories.append([
                history.history['acc'],
                history.history['val_acc'],
                history.history['loss'],
                history.history['val_loss']
            ])
        else:
            history = result_model.fit_generator(
                gd.getTrainData(keys,batch_size,classes,5,'train'), 
                verbose=1, 
                max_queue_size=2, 
                steps_per_epoch=3, 
                epochs=1
            )

            histories.append([
                history.history['acc'],
                history.history['loss']
            ])
        result_model.save_weights('weights/mobilenet_twostream22_{}e.h5'.format(old_epochs+1+e))
    print histories
    with open('data/trainHistoryTwoStream2{}_{}_{}e'.format(2, old_epochs, epochs), 'wb') as file_pi:
        pickle.dump(histories, file_pi)

else:
    result_model.load_weights('weights/mobilenet_twostream22_{}e.h5'.format(epochs))

    with open(out_file,'rb') as f2:
        keys = pickle.load(f2)
    len_samples = len(keys)
    print('-'*40)
    print('MobileNet Optical+RGB stream: Testing')
    print('-'*40)
    print 'Number samples: {}'.format(len_samples)

    score = result_model.evaluate_generator(gd.getTrainData(keys,batch_size,classes,5,'test'), max_queue_size=3, steps=len_samples/batch_size)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
