import numpy as np 
import torch 

data_dir = 'data/'
V = np.load(f'{data_dir}/V.npy')
C = np.load(f'{data_dir}/C.npy')

V = torch.tensor(V, dtype=torch.float32)
C = torch.tensor(C, dtype=torch.float32)
V = V.unsqueeze(1).repeat(1,C.shape[1],1,1)
C0 = C[:,:-1,:]
C1 = C[:,1:,:]
V = V[:,1:,:]
C0 = C0.reshape(-1, C0.shape[2])
C1 = C1.reshape(-1, C1.shape[2])
V = V.reshape(-1, V.shape[2], 2)