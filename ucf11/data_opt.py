import cv2
import os
import sys
import random
import numpy as np
import pickle
import config

# data_opt.py train 1
train = sys.argv[1]
sample_rate = int(sys.argv[2])
opt_rate = int(10/sample_rate)
server = config.server()
debug = False

if server:
    data_folder = r'/home/oanhnt/thainh/data/opt{}/{}/'.format(sample_rate,train)
    text_file = r'data/{}list.txt'.format(train)
    class_file = r'data/classInd.txt'
    data_video_folder = '/home/oanhnt/thainh/UCF-11/'
    out_file = r'/home/oanhnt/thainh/data/database/{}-opt{}.pickle'.format(train,sample_rate)
else:
    data_folder = r'/mnt/data11/opt{}/{}/'.format(sample_rate,train)
    out_file = r'/mnt/data11/database/{}-opt{}.pickle'.format(train,sample_rate)
    text_file = r'data/{}list.txt'.format(train)
    class_file = r'data/classInd.txt'
    data_video_folder = '/mnt/UCF-11/'

data=[]
classInd=[]

with open(class_file) as f0:
    for line in f0:
        class_name = line.rstrip()
        if class_name:
            classInd.append(class_name)
c = 0
with open(text_file) as f1:
    for line in f1:
        # create image name and folder
        if train == 'test':
            arr_line = line.rstrip()
        else:
            arr_line = line.split(' ')[0]
        path_video = arr_line.split('/')
        num_name = len(path_video)
        name_video = path_video[num_name - 1].split('.')[0]
        folder_video = path_video[0]
        path = data_folder + folder_video + '/'
        video_class = classInd.index(folder_video)

        if not os.path.isdir(path):
            os.makedirs(path)
            print 'make dir ' + path

        if not os.path.isdir(path + name_video):
            os.makedirs(path + name_video)
            print 'make dir ' + path + name_video

        cap = cv2.VideoCapture(data_video_folder + arr_line)
        ret, frame1 = cap.read()
        if not ret:
            continue
            print 'out'
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        k = 0
        m = 0
        os.chdir(path + name_video)
        while(True):  
            # Capture frame-by-frame
            ret, frame2 = cap.read()
            if not ret:
                break;

            if m%sample_rate == 0:
                if (k%opt_rate == 0) & (k > 9):
                    data.append([folder_video + '/' + name_video, 2*(k-10), video_class])
                    c+=1
                if not debug:
                    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
                    
                    # flow = optical_flow.calc(prvs, next, None)
                    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    prvs = next

                    horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
                    vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)

                    horz = horz.astype('uint8')
                    vert = vert.astype('uint8')

                    cv2.imwrite(str(2*k)+'.jpg',horz,[int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    cv2.imwrite(str(2*k+1)+'.jpg',vert,[int(cv2.IMWRITE_JPEG_QUALITY), 90])

                k+=1
            m+=1

        if (k%opt_rate > int(opt_rate/2)) | (k%opt_rate == 0):
            data.append([folder_video + '/' + name_video, 2*(k-10), video_class])
            c+=1

        print name_video
        # When everything done, release the capture
        cap.release()
print c
with open(out_file,'wb') as f2:
    pickle.dump(data,f2)

# with open(out_file,'rb') as f2:
#     b = pickle.load(f2)

# print b[100]

