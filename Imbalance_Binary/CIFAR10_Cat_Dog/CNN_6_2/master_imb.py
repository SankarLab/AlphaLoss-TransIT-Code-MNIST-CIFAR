import glob
import os
import sys

folders = glob.glob('*/')

lis = []
for folder in folders:
    a,b = folder.split('-')[2:4]
    a,b = int(a),int(b)
    ratio = None
    if a < b:
        ratio = 1 - a/b
    else:
        ratio = 1 - b/a
    lis.append((ratio,folder))

folders = []
lis.sort(key= lambda tup: tup[0], reverse=True)
for l in lis:
    folders.append(l[1])

class_a = sys.argv[1]
class_b = sys.argv[2]
print('Imbalance, Minority, Majority, LL Minotiy Acc, LL Majority Acc,Alpha Minority Acc, Alpha Majority Acc, Alpha*')
for folder in folders:
    os.system('python calculate_imb.py ' + \
                folder + \
                ' 0.95 ' + \
                class_a + ' ' +  \
                class_b)
