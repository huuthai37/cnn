import numpy as np
import pickle
import random
from PIL import Image
import cv2
from keras.utils import np_utils
import config

server = config.server()

def chunks(l, n):
    """Yield successive n-sized chunks from l"""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def getTrainData(keys,batch_size,classes,mode,train): 
    """
    mode 1: RGB Stream
    mode 2: Optical Stream
    mode 3: Sampled Optical Stream
    mode 4: RGB + Optical Stream
    mode 5: RGB + Sampled Optical Stream
    mode 6: Three stream
    """
    if train == 'train':
        data = True
    else: 
        data = False
    print train
    if server:
        data_folder = r'/home/oanhnt/thainh/data/opt1/{}/'.format(train)
        data_folder_rgb = r'/home/oanhnt/thainh/data/rgb/{}/'.format(train)
        data_folder_opt2 = r'/home/oanhnt/thainh/data/opt2/{}/'.format(train)
    else:
        data_folder = r'/mnt/smalldata/opt/{}/'.format(train)
        data_folder_rgb = r'/mnt/smalldata/rgb/{}/'.format(train)
        data_folder_opt2 = r'/mnt/smalldata/opt2/{}/'.format(train)

    while 1:
        for i in range(0, len(keys), batch_size):
            if mode == 1:
                X_train,Y_train=stackRGB(keys[i:i+batch_size],data_folder_rgb)
                print('Mode', mode)
            elif mode == 2:
                X_train,Y_train=stackOpticalFlow(keys[i:i+batch_size],data_folder,data)
                print('Mode', mode)
                print keys[i]
            elif mode == 3:
                X_train,Y_train=stackOpticalFlow(keys[i:i+batch_size],data_folder_opt2,data)
                print('Mode', mode)
            elif mode == 4:
                X_train,Y_train=stackOpticalFlowRGB(keys[i:i+batch_size],data_folder,data_folder_rgb,data)
                print('Mode', mode)
            elif mode == 5:
                X_train,Y_train=stackSparseOpticalFlowRGB(keys[i:i+batch_size],data_folder_opt2,data_folder_rgb,data)
                print('Mode', mode)
            else:
                X_train,Y_train=stackThreeStream(keys[i:i+batch_size],data_folder,data_folder_rgb,data_folder_opt2,data)
                print('Mode', mode)

            Y_train=np_utils.to_categorical(Y_train,classes)
            if not data:
                print 'Test batch {}'.format(i/batch_size+1)
            yield X_train, np.array(Y_train)

def stackOpticalFlow(chunk,data_folder,train):
    labels = []
    stack_opt = []
    for opt in chunk:
        folder_opt = opt[0] + '/'
        start_opt = opt[1]
        labels.append(opt[2])
        arrays = []
        # print(opt[0], opt[1])

        for i in range(start_opt, start_opt + 20):
            img=Image.open(data_folder + folder_opt + str(i) + '.jpg')
            arrays.append(img)

        stack = np.dstack(arrays)
        if train:
            ax = random.randint(0,96)
            ay = random.randint(0,16)
        else:
            ax = 48
            ay = 8
        nstack = stack[ay:ay+224,ax:ax+224,:]
        nstack = nstack.astype('float16',copy=False)
        nstack/=255
        print nstack.shape
        stack_opt.append(nstack)

    return (np.array(stack_opt), labels)

def stackRGB(chunk,data_folder_rgb):
    labels = []
    stack_rgb = []
    for opt in chunk:
        folder_opt = opt[0]
        start_opt = opt[1]
        labels.append(opt[2])

        if (start_opt%20>0):
            start_rgb = (int(np.floor(start_opt/20)) + 1 ) * 10
        else:
            start_rgb = int(start_opt/2)
        rgb = cv2.imread(data_folder_rgb + folder_opt + '-' + str(start_rgb) + '.jpg')
        resize_rgb = cv2.resize(rgb, (224, 224))
        resize_rgb = resize_rgb.astype('float16',copy=False)
        resize_rgb/=255

        stack_rgb.append(resize_rgb)

    return (np.array(stack_rgb), labels)

def stackOpticalFlowRGB(chunk,data_folder,data_folder_rgb,train):
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
        if train:
            ax = random.randint(0,96)
            ay = random.randint(0,16)
        else:
            ax = 48
            ay = 8
        nstack = stack[ay:ay+224,ax:ax+224,:]
        nstack = nstack.astype('float16',copy=False)
        nstack/=255
        # print nstack.shape
        stack_rgb.append(resize_rgb)
        stack_opt.append(nstack)

    return [np.array(stack_rgb), np.array(stack_opt)], labels

def stackSparseOpticalFlowRGB(chunk,data_folder,data_folder_rgb,train):
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
            start_rgb = (int(np.floor(start_opt/20))) * 20
        else:
            start_rgb = start_opt
        rgb = cv2.imread(data_folder_rgb + folder_opt + '-' + str(start_rgb) + '.jpg')
        resize_rgb = cv2.resize(rgb, (224, 224))
        resize_rgb = resize_rgb.astype('float16',copy=False)
        resize_rgb/=255
        # Stack optical flow
        for i in range(start_opt, start_opt + 20):
            img=Image.open(data_folder + folder_opt + '/' + str(i) + '.jpg')
            arrays.append(img)

        stack = np.dstack(arrays)
        if train:
            ax = random.randint(0,96)
            ay = random.randint(0,16)
        else:
            ax = 48
            ay = 8
        nstack = stack[ay:ay+224,ax:ax+224,:]
        nstack = nstack.astype('float16',copy=False)
        nstack/=255
        # print nstack.shape
        stack_rgb.append(resize_rgb)
        stack_opt.append(nstack)

    return [np.array(stack_rgb), np.array(stack_opt)], labels

def stackThreeStream(chunk,data_folder,data_folder_rgb,data_folder_opt2,train):
    labels = []
    stack_opt = []
    stack_opt2 = []
    stack_rgb = []
    for opt in chunk:
        folder_opt = opt[0]
        start_opt = opt[1]
        labels.append(opt[2])
        start_opt2 = opt[3]
        arrays = []
        arrays2 = []

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
        if train:
            ax = random.randint(0,96)
            ay = random.randint(0,16)
        else:
            ax = 48
            ay = 8
        nstack = stack[ay:ay+224,ax:ax+224,:]
        nstack = nstack.astype('float16',copy=False)
        nstack/=255

        # Stack optical flow sparse
        for i in range(start_opt2, start_opt2 + 20):
            img=Image.open(data_folder_opt2 + folder_opt + '/' + str(i) + '.jpg')
            arrays2.append(img)

        stack2 = np.dstack(arrays2)
        if train:
            ax = random.randint(0,96)
            ay = random.randint(0,16)
        else:
            ax = 48
            ay = 8
        nstack2 = stack2[ay:ay+224,ax:ax+224,:]
        nstack2 = nstack2.astype('float16',copy=False)
        nstack2/=255
        
        # Add stack to list
        stack_rgb.append(resize_rgb)
        stack_opt.append(nstack)
        stack_opt2.append(nstack2)

    return [np.array(stack_rgb), np.array(stack_opt), np.array(stack_opt2)], labels

def convert_weights(weights, depth):
    mat = weights[0]
    mat2 = np.empty([3,3,depth,32])
    for i in range(32):
        x=(mat[:,:,0,i] + mat[:,:,1,i] + mat[:,:,2,i])/3
        for j in range(depth):
            mat2[:,:,j,i] = x
    return [mat2]
