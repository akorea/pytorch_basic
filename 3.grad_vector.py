import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#H(x)= wx+b 벡터 미분 함수 직접 구현하여, 매개 변수 (w,b) 구하기
# x  [0,1,2,3,4,5..19]  
# y  [0.0, 2.5, 5.0,.. 47.5]


x_train= torch.FloatTensor([i for i in range(20)])
y_train = 2.5 * x_train

print(y_train.tolist())

w = torch.FloatTensor([2.3])
b = torch.randn(1)


epochs = 1000
learning_rate =  1e-6

for i in range(epochs+1):
    y_pred = x_train * w + b     # 벡터 곱 실행
    loss = torch.mean((y_pred-y_train)**2)
    if i % 100 ==0:
        print(f'{i}/{epochs} w={w.squeeze().detach()} loss = {loss}')

    #역전파
    grad_y_pred = 2.0 * (y_pred-y_train)     # (y_pred-y_train)**2  y_pred 로 미분
    grad_w = (grad_y_pred * x_train).mean()  # x_train * w + b 를 w 로 미분 -> y_pred * x
    grad_b = grad_y_pred.mean()             # x_train * w + b 를 b 로 편미분 ->  y_pred * 1
 
    # 매개변수 업데이트
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b
    

print(f' {x_train[2]} 의 실제값 : {y_train[2]} 예측값: {x_train[2] * w + b }')