import sys
import glob

prefix = sys.argv[1]
#test_acc = []
files = glob.glob(prefix+'*.out')
files = [f for f in files if 'alpha' in f]
files = [(float(f.split('.out')[0].split('-')[-1]),f) for f in files]
files.sort(key = lambda x: x[0])

for tup in files:
    alpha ,file_ = tup
    x = None
    with open(file_,'r') as f:
        content = f.read()
        x = content.split('=')[-1].strip('\n').split('True Negative')[0].strip('\n')
    print('Alpha ' + str(alpha) + ':')
    #print(','.join([y for y in x.split('\n')[0].replace('[','').replace(']','').split(' ') if len(y) > 1]))
    #print(','.join([y for y in x.split('\n')[1].replace('[','').replace(']','').split(' ') if len(y) > 1]))
    print(','.join(x.split('\n')[0].split('\t')))#[-1:]))
    print(','.join(x.split('\n')[1].split('\t')))#[-1:]))
    print('------')
    #print(str(round(mean,2)) + " \u00B1 " + str(round(se,2)))
