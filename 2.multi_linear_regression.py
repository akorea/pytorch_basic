import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


torch.manual_seed(1)

#H(x)=w1x1+w2x2+w3x3+b 구현하기
# x1, x2, x3   y
# 1,  1,  1    12
# 2,  2,  2    24
# 3,  3,  3    36


x1_train = torch.FloatTensor([[1], [2], [3]])
x2_train = torch.FloatTensor([[1], [2], [3]])
x3_train = torch.FloatTensor([[1], [2], [3]])

y_train = torch.FloatTensor([[12], [24], [36]])

w1= torch.zeros(1,requires_grad=True)
w2= torch.zeros(1,requires_grad=True)
w3= torch.zeros(1,requires_grad=True)

b=  torch.zeros(1,requires_grad=True)

optimizer = optim.Adam([w1, w2, w3, b], lr=0.01)

epochs = 5000

for i in range(epochs+1):
    y_pred = w1 * x1_train + w2 * x2_train + w3 * x3_train +b 

    loss = torch.mean((h-y_train)**2) 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} loss: {:.6f}'.format(
            i, epochs, w1.item(), w2.item(), w3.item(), b.item(), loss.item()
        ))

#H(x)= Wx+b 구현하기
#      X       Y
# x1, x2, x3   y
# 1,  1,  1    12
# 2,  2,  2    24
# 3,  3,  3    36

x_train= torch.FloatTensor([[1,1,1],[2,2,2],[3,3,3]])
y_train = torch.FloatTensor([[12], [24], [36]])

W = torch.zeros((3,1) , requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.Adam([W,b], lr = 0.01)
epochs = 5000

for i in range(epochs+1):
    y_pred = x_train.matmul(W) + b     # 행렬 곱 실행
    loss = torch.mean((h-y_train)**2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 100 ==0:
        print(f'{i}/{epochs} W={W.squeeze().detach()} loss = {loss.item()}')
    

 print(f' {x_train[1]} 의 실제값 : {y_train[1]}예측값 { x_train[1] @  W.detach() +b.detach().item()} ')