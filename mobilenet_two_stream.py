import keras
import sys
from keras.models import Model
from keras.layers import Dense, Conv2D, Activation, Reshape, Flatten, Input, ZeroPadding2D
import get_data as gd
from keras import optimizers
import pickle
import random
import numpy as np

# train: python mobilenet_temporal.py train 32 1 101
# test: python mobilenet_temporal.py test 32 1 101
# retrain: python mobilenet_temporal.py retrain 32 1 101 1
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

depth = 20
input_shape = (224,224,depth)

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

# Temporal
model = keras.applications.mobilenet.MobileNet(
    include_top=True,
)

# Disassemble layers
layers = [l for l in model.layers]

input_opt = Input(shape=input_shape)
x = ZeroPadding2D(padding=(1, 1), name='conv1_pad')(input_opt)
x = Conv2D(filters=32, 
          kernel_size=(3, 3),
          padding='valid',
          use_bias=False,
          strides=(2,2),
          name='conv_new')(x)

for i in range(2, len(layers)-3):
    layers[i].name = str(i)
    x = layers[i](x)

x = Flatten()(x)
x = Dense(classes, activation='softmax', name='predictions_x')(x)
temporal_model = Model(inputs=input_opt, outputs=x)
if train & (not retrain):
    temporal_model.load_weights('weights/mobilenet_temporal1_{}e.h5'.format(tem_epochs))

# Spatial
model2 = keras.applications.mobilenet.MobileNet(
    include_top=True,
    input_shape=(224,224,3)
)
y = Flatten()(model2.layers[-4].output)
y = Dense(classes, activation='softmax', name='predictions_y')(y)
spatial_model = Model(inputs=model2.input, outputs=y)
if train & (not retrain):
    spatial_model.load_weights('weights/mobilenet_spatial_{}e.h5'.format(spa_epochs))

# Fusion
z = Average()([x, y])

# Final touch
result_model = Model(inputs=[model2.input,input_opt], outputs=z)

# Run
result_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

if train:
    if retrain:
        result_model.load_weights('weights/mobilenet_twostream1_{}e.h5'.format(old_epochs))

    with open(out_file,'rb') as f1:
        keys = pickle.load(f1)
    len_samples = len(keys)
    print('-'*40)
    print('MobileNet Optical stream only: Training')
    print('-'*40)
    print 'Number samples: {}'.format(len_samples)
    
    for e in range(epochs):
        print('-'*40)
        print('Epoch', e+1)
        print('-'*40)

        random.shuffle(keys)
        result_model.fit_generator(gd.getTrainData(keys,batch_size,classes,2,train), verbose=1, max_queue_size=2, steps_per_epoch=len_samples/batch_size, epochs=1)
        result_model.save_weights('weights/mobilenet_twostream1_{}e.h5'.format(old_epochs+1+e))

else:
    result_model.load_weights('weights/mobilenet_twostream1_{}e.h5'.format(epochs))

    with open(out_file,'rb') as f2:
        keys = pickle.load(f2)
    len_samples = len(keys)
    print('-'*40)
    print('MobileNet Optical stream only: Testing')
    print('-'*40)
    print 'Number samples: {}'.format(len_samples)

    score = result_model.evaluate_generator(gd.getTrainData(keys,batch_size,classes,2,train), max_queue_size=3, steps=len_samples/batch_size)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])