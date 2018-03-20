import cv2
import os
import sys
import random
import numpy as np
import pickle
import config

# data_opt.py train
train = sys.argv[1]
server = config.server()
debug = True

if server:
    data_folder = r'/home/oanhnt/thainh/data/opt/{}/'.format(train)
    text_file = r'/home/oanhnt/thainh/ucfTrainTestlist/{}list01.txt'.format(train)
    class_file = r'/home/oanhnt/thainh/ucfTrainTestlist/classInd.txt'
    data_video_folder = '/home/oanhnt/thainh/UCF-101/'
    out_file = r'/home/oanhnt/thainh/data/database/{}-opt.pickle'.format(train)
else:
    data_folder = r'/mnt/smalldata/opt/{}/'.format(train)
    text_file = r'/mnt/ucf101/ucfTrainTestlist/{}b.txt'.format(train)
    class_file = r'/mnt/ucf101/ucfTrainTestlist/class.txt'
    data_video_folder = '/mnt/ucf101/UCF-101/'
    out_file = r'/mnt/smalldata/database/{}-opt.pickle'.format(train)

data=[]
classInd=[]

with open(class_file) as f0:
    for line in f0:
        class_name = line.split(' ')[1]
        class_name = class_name.rstrip()
        if class_name:
            classInd.append(class_name)
c = 0
with open(text_file) as f1:
    for line in f1:
        # create image name and folder
        if train == 'train':
            arr_line = line.split(' ')[0]
        else:
            arr_line = line.rstrip()
        path_video = arr_line.split('/')
        num_name = len(path_video)
        name_video = path_video[num_name - 1].split('.')[0]
        folder_video = path_video[num_name - 2]
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
        os.chdir(path + name_video)
        while(True):
            # After 10 frames push into data
            if (k%10 == 0) & (0 != k):
                data.append([folder_video + '/' + name_video, 2*(k-10), video_class])
                c+=1
            # Capture frame-by-frame
            ret, frame2 = cap.read()
            if not ret:
                break;
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

        if (k%10 > 4):
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

