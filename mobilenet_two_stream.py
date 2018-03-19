import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, Activation, Reshape, Flatten, Input, ZeroPadding2D, Average
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import optimizers
from keras.utils import np_utils
import pickle
import random
from PIL import Image
import cv2

hdf5_path = '/mnt/dataset_opt_stack.hdf5'
train = False
depth = 20
input_shape = (224,224,depth)
classes = 3
batch_size=8
if train:
    out_file = '/mnt/smalldata/database/train-opt.pickle'
    data_folder = '/mnt/smalldata/opt/train/'
    data_folder_rgb = '/mnt/smalldata/rgb/train/'
else:
    out_file = '/mnt/smalldata/database/test-opt.pickle'
    data_folder = '/mnt/smalldata/opt/test/'
    data_folder_rgb = '/mnt/smalldata/rgb/test/'

def convert_weights(weights):
	mat = weights[0]
	mat2 = np.empty([3,3,depth,32])
	for i in range(32):
		x=(mat[:,:,0,i] + mat[:,:,1,i] + mat[:,:,2,i])/3
		for j in range(depth):
			mat2[:,:,j,i] = x
	return [mat2]

def getTrainData(keys,batch_size,classes):
    
    while 1:
        for i in range(0, len(keys), batch_size):
            X_train,Y_train=stackOF(keys[i:i+batch_size])
            if (X_train and Y_train):
                # X_train/=255
                # X_train=X_train-np.average(X_train)
                Y_train=np_utils.to_categorical(Y_train,classes)
            print('Add batch', i/batch_size+1)
            yield X_train, np.array(Y_train)

def stackOF(chunk):
    labels = []
    stack_opt = []
    stack_rgb = []
    for opt in chunk:
        folder_opt = opt[0]
        start_opt = opt[1]
        labels.append(opt[2])
        arrays = []

        # RGB Frame
        if (start_opt%20>0):
            start_rgb = (int(np.floor(start_opt/20)) + 1 ) * 10
        else:
            start_rgb = int(start_opt/2)
        rgb = cv2.imread(data_folder_rgb + folder_opt + '-' + str(start_rgb) + '.jpg')
        resize_rgb = cv2.resize(rgb, (224, 224))
        resize_rgb = resize_rgb.astype('float16',copy=False)
        resize_rgb/=255
        # Stack optical flow
        for i in range(start_opt, start_opt + 20):
            img=Image.open(data_folder + folder_opt + '/' + str(i) + '.jpg')
            arrays.append(img)

        stack = np.dstack(arrays)
        ax = random.randint(0,96)
        ay = random.randint(0,16)
        nstack = stack[ay:ay+224,ax:ax+224,:]
        nstack = nstack.astype('float16',copy=False)
        nstack/=255
        # print nstack.shape
        stack_rgb.append(resize_rgb)
        stack_opt.append(nstack)

    return [np.array(stack_rgb), np.array(stack_opt)], labels

# Temporal
model = keras.applications.mobilenet.MobileNet(
	include_top=True,  
	# weights='imagenet'
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
# temporal_model.load_weights('mobilenet_opt_weights_63e.h5')

# Spatial
model2 = keras.applications.mobilenet.MobileNet(
    include_top=True,  
    # weights='imagenet'
    input_shape=(224,224,3)
)
y = Flatten()(model2.layers[-4].output)
y = Dense(classes, activation='softmax', name='predictions_y')(y)
spatial_model = Model(inputs=model2.input, outputs=y)
# spatial_model.load_weights('mobilenet_nor_weights_v3.h5')

# Fusion
z = Average()([x, y])

# Final touch
result_model = Model(inputs=[model2.input,input_opt], outputs=z)

# Run
result_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

# result_model.summary()

if train:
    epoch = 1

    result_model.load_weights('mobilenet_two_weights_1e.h5')
    with open(out_file,'rb') as f2:
        keys = pickle.load(f2)
    len_samples = len(keys)
    print len_samples
    
    for e in range(epoch):
        print('-'*40)
        print('Epoch', e+1)
        print('-'*40)

        random.shuffle(keys)
        result_model.fit_generator(getTrainData(keys,batch_size,classes), verbose=1, max_queue_size=1, steps_per_epoch=len_samples/batch_size, epochs=1)
        result_model.save_weights('mobilenet_two_weights_2e.h5')

    # result_model.save_weights('mobilenet_opt_weights_66e.h5')
else:
    result_model.load_weights('mobilenet_two_weights_2e.h5')
    with open(out_file,'rb') as f2:
        keys = pickle.load(f2)
    len_samples = len(keys)
    print len_samples

    random.shuffle(keys)

    score = result_model.evaluate_generator(getTrainData(keys,batch_size,classes), steps=len_samples/batch_size)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

# mobilenet_nor_weights_v2 for regular MobileNet with RGB non-stack with change last layer
# mobilenet_stack_weights for regular MobileNet with RGB stack with change last layer, frist layer