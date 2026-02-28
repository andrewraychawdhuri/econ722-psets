# Basic Commands
import numpy as np
a = np.arange(15).reshape(3,5)
print(a)
A = np.array([[1,1], [0,1]])
B = np.array([[2,0],[3,4]])
C = A*B # element wise product
D = A @ B # matrix multiplication
E = A.dot(B) # matrix product but different notation

# Import a Package: Running a Linear Regression Model

from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit([[0,0],[1,1],[2,2]],[0,1,2])
reg.coef_

# PyTorch - Vectors and Matrices are called tensors in PyTorch

import torch
import math 
x = torch.empty(3,4)
print((type(x)))
print(x)
zeros = torch.zeros(2,3)
ones = torch.ones(2,3)
torch.manual_seed(1729)
random = torch.rand(2,3)
