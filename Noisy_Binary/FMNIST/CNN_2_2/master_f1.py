import glob
import os
import sys

folders = glob.glob('*/')
#print(folders)

class_a = sys.argv[1]
class_b = sys.argv[2]
print('Imbalance, Minority, Majority, LL F1, Alpha F1, Alpha*')
for folder in folders:
    os.system('python calculate_f1.py ' + \
                folder + \
                ' 0.95 ' + \
                class_a + ' ' +  \
                class_b)
