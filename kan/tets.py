# from spline import *

# num_spline = 5
# num_sample = 100
# num_grid_interval = 10
# k = 3
# x = torch.normal(0, 1, size=(num_spline,num_sample))
# grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
# print(grids)
# splines_batch = B_batch(x, grids, k=k).shape
# print(splines_batch)
# for spl in splines_batch:
    
from KAN import *
import torch
# model = KAN(width=[2,3,2,1])
# x = torch.normal(0,1,size=(100,2))
# model(x);
# beta = 100
# model.plot(beta=beta)
model = KANLayer(in_dim=3, out_dim=5)
print(model.in_dim, model.out_dim)