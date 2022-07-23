#!/usr/bin/env python3

import argparse
import numpy as np
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('input',  metavar='FILENAME', type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args.input)
    
with open(args) as f:
    text = f.readlines()
    data = np.array(text, dtype=float)

code = np.array([+1, +1, +1, -1, -1, -1, +1, -1, -1, +1, -1], dtype=np.int8)
code_ = np.repeat(code, 5)
decoted = np.convolve(data, code_[::-1], mode='full')


std = np.std(decoted)
mean = np.mean(decoted)


b = []
k = 0
for i in range(len(decoted)):
    
    if decoted[i] > mean +  2*std:
        if decoted[i - 1] > mean +  2*std:
            k = 1
        else:
            b.append(1)

    elif decoted[i] < mean - 2*std:
        if decoted[i - 1] < mean -  2*std:
            k = 0
        else:
             b.append(0)
        
b = np.asarray(b, dtype=np.int8) 
c = np.packbits(b)
d = c.tobytes()


file = { "message":  d.decode('ascii')}
with open('wifi.json', 'w+') as f:
    json.dump(file, f)
