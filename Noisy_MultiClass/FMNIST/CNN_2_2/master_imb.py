import glob
import os
import sys

folders = glob.glob('*/')
#print(folders)

class_a = sys.argv[1]
class_b = sys.argv[2]
print('Imbalance, Minority, Majority, LL Minotiy Acc, LL Majority Acc,Alpha Minority Acc, Alpha Majority Acc, Alpha*')
for folder in folders:
    os.system('python calculate_imb.py ' + \
                folder + \
                ' 0.95 ' + \
                class_a + ' ' +  \
                class_b)
