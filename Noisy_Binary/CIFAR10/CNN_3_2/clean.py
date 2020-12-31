import os
import glob

files = glob.glob('./*/')
for f in files:
    os.system('rm -rf ' + f)
