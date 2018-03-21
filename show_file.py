import numpy as np 
import pickle
import sys
import cv2
import config

train = sys.argv[1]
file = sys.argv[2]
num = int(sys.argv[3])
server = config.server()

if server:
	dirr = '/home/oanhnt/thainh/data/database/'
	dir_data = '/home/oanhnt/thainh/data/{}/{}/'.format(file, train)
else:
	dirr = '/mnt/smalldata/database/'
	dir_data = '/mnt/smalldata/{}/{}/'.format(file, train)

with open(dirr + train + '-' + file + '.pickle','rb') as f1:
    data = pickle.load(f1)
length = len(data)
print length
for i in range(length):
	fileimg = data[i][0]
	start_opt = data[i][1]
	for j in range(start_opt, start_opt + 20):
		img = cv2.imread(dir_data + fileimg + '/' + str(j) + '.jpg')
		if img is None:
			print data[i]
	if i%1000 == 0:
		print 'Checked {}/{}'.format(i, length)
for i in range(num):
	print data[i]