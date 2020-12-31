import os
import glob
import sys

binary = sys.argv[1]
sentinel = sys.argv[2]

if sentinel == 'left':
    for lr in ['1e-3','5e-4','1e-4']:
        print('-------',lr,'-------')
        for pair in [(5000,5000)]:#,(2500,2500),(500, 4500),(250,4750),(50,4950)]:
            a,b = str(pair[0]),str(pair[1])
            #print('Alpha,',a,',',b)
            print(' ')
            folder = 'mnist-imb-'+a+'-'+b+'-binary-'+binary+'-noise-p0-lr_'+lr+'/'
            os.system('python calculate_confidence.py ' + folder + ' 0.95')
            print('--------------------')
elif sentinel == 'right':
    for lr in ['1e-3','5e-4','1e-4']:
        print('-------',lr,'-------')
        for pair in [(5000,5000),(2500,2500),(4500, 500),(4750,250),(4950,50)]:
            a,b = str(pair[0]),str(pair[1])
            print('Alpha,',a,',',b)
            folder = 'mnist-imb-'+a+'-'+b+'-binary-'+binary+'-noise-p0-lr_'+lr+'/'
            os.system('python calculate_confidence.py ' + folder + ' 0.95')
            print('--------------------')
