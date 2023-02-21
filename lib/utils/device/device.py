from math import ceil
import multiprocessing

import torch
import torch.cuda as Cuda


Mode = 1 # 0, 1, 2
NUM_GPU = Cuda.device_count()
if Mode == 0:
    CUDA_AVAI = False
    DataParallel = False
    Device = torch.device('cpu')
elif Mode == 1:
    # GPU
    CUDA_AVAI = Cuda.is_available()
    DataParallel = NUM_GPU > 1
    Device = torch.device('cuda:0' if CUDA_AVAI else 'cpu')


def workerManager(BatchSize):
    NUM_PROC = multiprocessing.cpu_count()
    if CUDA_AVAI:
        if BatchSize <= 32:
            WorkerKanban = BatchSize // 4
        elif BatchSize <= 64:
            WorkerKanban = BatchSize // 8
        else:
            WorkerKanban = BatchSize // 16
        NumWorkers = ceil(min(WorkerKanban, NUM_PROC - 2) / 2.) * 2
        PinMemory = True
    else:
        NumWorkers = ceil(min(4 * round(NUM_PROC / 8), NUM_PROC - 2) / 2.) * 2
        PinMemory = False
    print("We use [%d/%d] workers, and pin memory is %s" %(NumWorkers, NUM_PROC, str(PinMemory)))
    return PinMemory, NumWorkers