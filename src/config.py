import torch
import os
from datetime import datetime

PATH = './data/'
MAXINT = 10000
SEQ_LENGTH = 21 # x: 'START' + tokens; y: tokens + 'END'
EMB_SIZE = 32
GENERATE_NUM = 100
FILTER_SIZE = list(range(1,SEQ_LENGTH))
NUM_FILTER =  ([100] + [200] * 9 + [160] * SEQ_LENGTH)[0:SEQ_LENGTH-1]
DIS_NUM_EPOCH = 30
DIS_NUM_EPOCH_PRETRAIN = 10
GEN_NUM_EPOCH = 30
GEN_NUM_EPOCH_PRETRAIN = 10
GEN_HIDDEN_DIM = 48
ROLLOUT_ITER = 16
TOTAL_BATCH = 50

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NrGPU = 0

def openLog(filename='record.txt'):
    if os.path.exists(PATH+filename):
        append_write = 'a'
    else:
        append_write = 'w'
    log = open(PATH+filename, append_write)
    return log

if DEVICE.type == 'cuda':
    NrGPU = torch.cuda.device_count()
    print('number of GPUs available:{}'.format(NrGPU))
    log = openLog('gpu.txt')
    log.write('datetime:{}, device name:{}\n'.format(datetime.now(),
                                          torch.cuda.get_device_name(0)))
    log.write('Memory Usage:')
    log.write('\nAllocated:'+str(round(torch.cuda.memory_allocated(0)/1024**3,1))+'GB')
    log.write('\nCached:   '+str(round(torch.cuda.memory_reserved(0)/1024**3,1))+'GB')
    log.close()
