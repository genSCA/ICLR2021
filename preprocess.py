import numpy as np
from numpy import savez_compressed

'''
Use this tool to preprocess the raw data get by pintools `get_trace.cpp`
'''

pad_size = 256 * 256 * 6

CACHE_MASK = 0xFC0

input_file = open('get_trace.out', 'r')
lines = input_file.readlines()
length = len(lines) - 1
print(length)

res = []
for line in lines[:-1]:
    content = line.split(' ')
    RorW = content[0]
    address = (int(content[1], 16) & CACHE_MASK) >> 6
    if RorW == 'R':
        res.append(address)
    elif RorW == 'W':
        res.append(-address)
    else:
        sys.exit('Error!')
if length < pad_size:
    for i in range(length, pad_size):
        res.append(0)
else:
    res = res[:pad_size]

matrix = np.array(list(res))
savez_compressed('trace.npz', matrix)

'''
For OS Page Table
'''
#input_file = open('get_trace.out', 'r')
#lines = input_file.readlines()
#length = len(lines) - 1
#print(length)
#
#res = []
#for line in lines[:-1]:
#    content = line.split(' ')
#    RorW = content[0]
#    address = (int(content[1], 16)) >> 12
#    if RorW == 'R':
#        res.append(address)
#    elif RorW == 'W':
#        res.append(-address)
#    else:
#        sys.exit('Error!')
#if length < pad_size:
#    for i in range(length, pad_size):
#        res.append(0)
#else:
#    res = res[:pad_size]
#
#matrix = np.array(list(res))
#savez_compressed('trace.npz', matrix)

'''
For Read/Write Access
'''
#input_file = open('get_trace.out', 'r')
#lines = input_file.readlines()
#length = len(lines) - 1
#print(length)
#
#res = []
#for line in lines[:-1]:
#    content = line.split(' ')
#    RorW = content[0]
#    if RorW == 'R':
#        res.append(1)
#    elif RorW == 'W':
#        res.append(-1)
#    else:
#        sys.exit('Error!')
#if length < pad_size:
#    for i in range(length, pad_size):
#        res.append(0)
#else:
#    res = res[:pad_size]
#
#matrix = np.array(list(res))
#savez_compressed('trace.npz', matrix)