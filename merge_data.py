import numpy as np 
import pickle
import sys

train = sys.argv[1]
server = True

if server:
    opt_file = r'/home/oanhnt/thainh/data/database/{}-opt.pickle'.format(train)
    opt2_file = r'/home/oanhnt/thainh/data/database/{}-opt2.pickle'.format(train)
    out_file = r'/home/oanhnt/thainh/data/database/{}-all.pickle'.format(train)
else:
    opt_file = r'/mnt/smalldata/database/{}-opt.pickle'.format(train)
    opt2_file = r'/mnt/smalldata/database/{}-opt2.pickle'.format(train)
    out_file = r'/mnt/smalldata/database/{}-all.pickle'.format(train)

with open(opt_file,'rb') as f1:
    opt1 = pickle.load(f1)

with open(opt2_file,'rb') as f2:
    opt2 = pickle.load(f2)
l = len(opt2)
print (len(opt1), len(opt2))

k = 0
for i in range(len(opt1)):
	# print (i, k)
	if (i - k >= l):
		k+=1
		opt1[i].append(opt2[i-k][1])
	else:
		if(opt1[i][0] == opt2[i-k][0]):
			opt1[i].append(opt2[i-k][1])
		else:
			k+=1
			opt1[i].append(opt2[i-k][1])

with open(out_file,'wb') as f3:
    pickle.dump(opt1,f3)