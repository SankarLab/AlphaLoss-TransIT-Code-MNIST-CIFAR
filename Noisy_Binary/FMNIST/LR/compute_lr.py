import numpy as np
import scipy.stats
import sys
import glob

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

import os
import glob

files = glob.glob('./*/')
for fi in files:
    lr = fi.split('lr_')[-1]
    test_acc = []
    fil = glob.glob(fi+'*alpha-1.0.out')[0]
    with open(fil,'r') as f:
        for line in f:
            line = line.strip('\n')
            if 'Test Accuracy:\t' in line:
                line = line.split('Test Accuracy:\t')[-1]
                test_acc.append(float(line))

    mean3, se3 = mean_confidence_interval(test_acc)
    #file_name = file_.split('.out')[0]
    '''
    print('Alpha ' + str(alpha) + ', ' + str(round(mean1,3)) + " \u00B1 " + str(round(se1,3)) + 
            ',' + str(round(mean2,3)) + " \u00B1 " + str(round(se2,3)) +
            ',' + str(round(mean3,3)) + " \u00B1 " + str(round(se2,3)))
    '''
    print('LR (' + str(lr).replace('/','') + ')' + ' Alpha 1.0, ' + str(round(mean3,3)))
    #print(str(round(mean,2)) + " \u00B1 " + str(round(se,2)))
