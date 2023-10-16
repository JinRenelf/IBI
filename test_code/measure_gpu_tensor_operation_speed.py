# -*- coding: utf-8 -*-

# @File    : measure_gpu_tensor_operation_speed.py
# @Date    : 2023-10-13
# @Author  : ${ RenJin}
import time
import torch

x = torch.cuda.FloatTensor(10000, 500).normal_()
w = torch.cuda.FloatTensor(200, 500).normal_()

# ensure that context initialization and normal_() operations
# finish before you start measuring time
torch.cuda.synchronize()
torch.cuda.synchronize()

a = time.perf_counter()
y = x.mm(w.t())
torch.cuda.synchronize() # wait for mm to finish
b = time.perf_counter()
print('batch GPU {:.02e}s'.format(b - a))

a = time.perf_counter()
y = x.mm(w.t())
torch.cuda.synchronize() # wait for mm to finish
b = time.perf_counter()
print('batch GPU {:.02e}s'.format(b - a))

a = time.perf_counter()
y = x.mm(w.t())
torch.cuda.synchronize() # wait for mm to finish
b = time.perf_counter()
print('batch GPU {:.02e}s'.format(b - a))

a = time.perf_counter()
y = x.mm(w.t())
torch.cuda.synchronize() # wait for mm to finish
b = time.perf_counter()
print('batch GPU {:.02e}s'.format(b - a))
