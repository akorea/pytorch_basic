import torch
import numpy as np

#########################################
#인공지능 학습을 위한 텐서 기본 조작법

############################################
# Tensor 나눗셈 
############################################


a = torch.tensor([[1, 2], [3, 4.]])
b = torch.tensor([2, 1])
c= a/b
print(f"a/b={c}")

# tensor([[0.5000, 2.0000],
#         [1.5000, 4.0000]])

c= a/b.unsqueeze(-1)
print(f"a/b.unsqueeze(-1)={c}")
# tensor([[0.5000, 1.0000],
#         [3.0000, 4.0000]])

print(f"b.unsqueeze(0) {b.unsqueeze(0)}")

print(a[0]) #a[0] = a[0,:]
print(a[0,:])
print(a.T[0]) #a.T[0] = a[:,0]
print(a[:,0])

a = torch.tensor([[1, 2], [3, 4.]])
b = torch.tensor([[2, 1]])

c= a/b.T
print(f"a/b.T={c}")

a = torch.tensor([[1, 2], [3, 4.]])
b = torch.tensor([2, 1])

c= a/b.T
print(f"a/b.T={c}")

# tensor([[0.5000, 2.0000],
#         [1.5000, 4.0000]])

############################################
# Tensor 합
############################################

c= torch.sum(a, axis=0)
print(c)
# tensor([4., 6.])

c = torch.sum(a)
print(c)
# tensor(10.)

c= torch.sum(a, axis=1)
print(c)
#tensor([3., 7.])

############################################
# Tensor 원소, 리스트로 가져오기
############################################
c= torch.tensor([3])
print(c.item())
#3
print(c.tolist())
#[3]


############################################
# Tensor 합치기 concat 
############################################
x1_train = torch.FloatTensor([[1], [2], [3]])
x2_train = torch.FloatTensor([[1], [2], [3]])
x3_train = torch.FloatTensor([[1], [2], [3]])


c= torch.cat((x1_train.T, x2_train.T, x3_train.T) ,dim=0)
print(c)

# tensor([[1., 2., 3.],
#         [1., 2., 3.],
#         [1., 2., 3.]])

c= torch.cat((x1_train, x2_train, x3_train) ,dim=0)
print(c.view(-1,3))
# tensor([[1., 2., 3.],
#         [1., 2., 3.],
#         [1., 2., 3.]])

############################################
# Tensor 자동 미분 
############################################

w = torch.tensor(4.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

#연쇄 편 미분
y = w**2 +b
z = 3*y +1

#미분
z.backward()

# w가 속한 수식을 w로 미분한 값이 저장
# w(4) * 2 * 3 = 24
print(w.grad)

# b가 속한 수식을 b로 미분한 값이 저장
# b(1) * 3 = 3
print(b.grad)



