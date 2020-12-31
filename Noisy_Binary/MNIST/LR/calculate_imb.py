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

prefix = sys.argv[1]
con = sys.argv[2]
class_a = sys.argv[3]
class_b = sys.argv[4]
#test_acc = []
files = glob.glob(prefix+'*.out')
files = [f for f in files if 'alpha' in f]
#print(files[0].split('.out')[0].split('-')[-1])
files = [(float(f.split('.out')[0].split('-')[-1]),f) for f in files]
files.sort(key = lambda x: x[0])

# Capture Imbalance 
prefix_ = prefix.strip('/')
imb = prefix_.split('imb-')[-1].split('-binary')[0]
imb_a,imb_b = imb.split('-')[0], imb.split('-')[-1]
#print(imb_a,imb_b)

max_f1 = []
ll_f1 = None
for tup in files:
    alpha ,file_ = tup
    test_acc = []
    test_a, test_b = [],[]
    with open(file_,'r') as f:
        for line in f:
            line = line.strip('\n')
            if 'Test Class Accuracies:\t' not in line:
                continue
            line = line.split('Test Class Accuracies:\t')[-1]
            line = line.split(',')
            #acc = float(line)
            #test_acc.append(acc)
            test_a.append(float(line[0]))
            test_b.append(float(line[1]))

    mean1, se1 = mean_confidence_interval(test_a,float(con))
    mean2, se2 = mean_confidence_interval(test_b,float(con))
    file_name = file_.split('.out')[0]
    #print('Alpha ' + str(alpha) + ', ' + str(round(mean1,3)) + " \u00B1 " + str(round(se1,3)))
    # Log Loss
    if alpha == 1.0:
        ll_f1 = (str(round(mean1,3)), str(round(se1,3)),
                 str(round(mean2,3)), str(round(se2,3)))
    # Add alpha, mean, se
    max_f1.append((round(mean1,3),round(se1,3),
                   round(mean2,3),round(se2,3),alpha))

#max_f1.sort(key = lambda x: x[0])

minority, majority = '',''
min_, maj_ = 0,0
maxs_f1 = []
if float(imb_a) < float(imb_b):
    minority, majority = class_a, class_b
    min_, maj_ = float(imb_a), float(imb_b)
    max_f1.sort(key = lambda x: x[0])
    max_f1_mean = max_f1[-1][0]
    for i in max_f1:
        if max_f1_mean == i[0]:                                                             maxs_f1.append((str(i[0]),str(i[1]),
                str(i[2]),str(i[3]), str(i[4])))
else:
    minority, majority = class_b, class_a
    min_, maj_ = float(imb_b), float(imb_a)
    max_f1.sort(key = lambda x: x[2])
    max_f1_mean = max_f1[-1][2]
    new_ll = (ll_f1[2],ll_f1[3],ll_f1[0],ll_f1[1])
    ll_f1 = new_ll
    for i in max_f1:
        if max_f1_mean == i[2]:                                                             maxs_f1.append((str(i[2]),str(i[3]),
                str(i[0]),str(i[1]), str(i[4])))

imb_ratio = str(maj_/ (min_ + maj_))
#print('Imbalance, Minority, Majority, LL Minority Accuracy, LL Majority Accuracy, Alpha Minority Accuracy, Alpha Majority Accuracy, Alpha*')
'''
print(imb_ratio + ',' + \
     minority + ',' + \
     majority + ',' + \
     ll_f1[0] + " \u00B1 " + ll_f1[1] + ',' + \
     ll_f1[2] + " \u00B1 " + ll_f1[3] + ',' + \
     str(maxs_f1[-1][0]) + " \u00B1 " + str(maxs_f1[-1][1]) + ',' + \
     str(maxs_f1[-1][2]) + " \u00B1 " + str(maxs_f1[-1][3]) + ',' + \
    ';'.join([l[-1] for l in maxs_f1]))
'''
print(imb_ratio + ',' + \
    minority + ',' + \
    majority + ',' + \
    ll_f1[0] + ',' + \
    ll_f1[2] + ',' + \
    str(maxs_f1[-1][0]) + ',' + \
    str(maxs_f1[-1][2]) + ',' + \
    ';'.join([l[-1] for l in maxs_f1]))
