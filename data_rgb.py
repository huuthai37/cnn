import cv2
import os
import sys
import random
import numpy as np

# data_rgb.py train 10 noaug 

train = sys.argv[1]
sample_rate = int(sys.argv[2])
if sys.argv[3] == 'aug':
    gen_aug = True
    aug_size = sys.argv[4]
else:
    gen_aug = False

data_folder = r'/home/oanhnt/thainh/data/rgb/{}/'.format(train)
text_file = r'/home/oanhnt/thainh/ucfTrainTestlist/{}list01.txt'.format(train)
data_video_folder = '/home/oanhnt/thainh/UCF-101/'

def crop_image(frame, name_video, i, y, x):
    crop = frame[y:y+224, x:x+224].copy()
    cv2.imwrite(r'{}-{}-{}{}.png'.format(name_video, i, y, x), crop)
    crop_flip = cv2.flip(crop, 1)
    cv2.imwrite(r'{}-{}-{}{}-flh.png'.format(name_video, i, y, x), crop_flip)

with open(text_file) as f:
    for line in f:
        # create image name and folder
        if train == 'train':
            arr_line = line.split(' ')[0]
        else:
            arr_line = line.split('\n')[0]
        path_video = arr_line.split('/')
        num_name = len(path_video)
        name_video = path_video[num_name - 1].split('.')[0]
        folder_video = path_video[num_name - 2]
        path = data_folder + folder_video + '/'

        if not os.path.isdir(path):
            os.makedirs(path)
            print 'make dir ' + path

        cap = cv2.VideoCapture(data_video_folder + arr_line)
        i = -1
        os.chdir(path)
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break
                cap.release()
                sys.exit()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            i = i + 1
            if (i%sample_rate != 0):
                continue
            
            # Our operations on the frame come here
            # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the resulting frame
            # cv2.imshow('frame',frame)
            # print r'{}{}-{}.png'.format(path, name_video, i)

            # random crop to 224x224
            x = random.randint(0,96)
            y = random.randint(0,16)
            
            crop = frame[y:y+224, x:x+224].copy()

            cv2.imwrite(r'{}-{}.png'.format(name_video, i),crop)
            if train == 'train':
                if not gen_aug:
                    crop_flip = crop.copy()
                    crop_flip = cv2.flip(crop_flip, 1)
                    cv2.imwrite(r'{}-{}-flip.png'.format(name_video, i),crop_flip)
                else:
                    #make augmentation data image
                    for k in range(1, aug_size + 1):
                        flip = random.randint(0,1)
                        # random horizontal flipping
                        crop_flip = crop.copy()
                        if (flip==1):
                            crop_flip = cv2.flip(crop_flip, flip)
                        # random rgb jittering
                        B = crop_flip[:,:,0]
                        G = crop_flip[:,:,1]
                        R = crop_flip[:,:,2]
                        crop_flip = np.dstack( (
                            np.roll(B, random.randint(1,5) - 3, axis=random.randint(0,1)), 
                            np.roll(G, random.randint(1,5) - 3, axis=random.randint(0,1)), 
                            np.roll(R, random.randint(1,5) - 3, axis=random.randint(0,1))
                        ))
                        cv2.imwrite(r'{}-{}-{}.png'.format(name_video, i, k),crop_flip)
            else:
                # crop center and 4 corners + flip => 10 images
                crop_image(frame, name_video, i, 0, 0)
                crop_image(frame, name_video, i, 8, 48)
                crop_image(frame, name_video, i, 16, 96)
                crop_image(frame, name_video, i, 16, 0)
                crop_image(frame, name_video, i, 0, 96)

        print name_video

    

# When everything done, release the capture
cap.release()
# print("--- %s seconds ---" % (time.time() - start_time))
