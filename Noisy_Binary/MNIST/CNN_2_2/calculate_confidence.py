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
#test_acc = []
files = glob.glob(prefix+'*.out')
files = [f for f in files if 'alpha' in f]
#print(files[0].split('.out')[0].split('-')[-1])
files = [(float(f.split('.out')[0].split('-')[-1]),f) for f in files]
files.sort(key = lambda x: x[0])

for tup in files:
    alpha ,file_ = tup
    test_acc = []
    test_a, test_b = [],[]
    with open(file_,'r') as f:
        for line in f:
            line = line.strip('\n')
            if 'Test Accuracy:\t' not in line:
                continue
            line = line.split('Test Accuracy:\t')[-1]
            #line = line.split(',')
            acc = float(line)
            test_acc.append(acc)
            #test_a.append(float(line[0]))
            #test_b.append(float(line[1]))

    mean,se = mean_confidence_interval(test_acc,float(con))
    #mean1, se1 = mean_confidence_interval(test_a,float(con))
    #mean2, se2 = mean_confidence_interval(test_b,float(con))
    file_name = file_.split('.out')[0]
    #print('Alpha ' + str(alpha) + ', ' + str(round(mean1,3)) + " \u00B1 " + str(round(se1,3)) + ',' + str(round(mean2,3)) + " \u00B1 " + str(round(se2,3)))
    print('Alpha ' + str(alpha) + ', ' + str(round(mean,2)) + " \u00B1 " + str(round(se,2)))
