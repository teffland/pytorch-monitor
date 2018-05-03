from pytorch_monitor import *
import torch
m = torch.nn.Linear(5,1)
writer, config = init_experiment({'log_dir':'test'})
monitor_module(m, writer)
x = torch.ones(3,5)
y = m(x)
z = y.sum()
z.backward()
