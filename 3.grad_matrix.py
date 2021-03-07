import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#H(x)= WX+b 행렬 미분 함수 직접 구현하여, 매개 변수 (W,b) 구하기
#      X       Y
# 1,  1,  1    12
# 2,  2,  2    24
# 3,  3,  3    36

x_train= torch.FloatTensor([[1,1,1],[2,2,2],[3,3,3]])
y_train = torch.FloatTensor([[12], [24], [36]])

W = torch.zeros((3,1))
b = torch.zeros(1)


epochs = 10000
learning_rate = 0.001

for i in range(epochs+1):
    y_pred = x_train.mm(W) + b     # 행렬 곱 실행
    loss = torch.mean((y_pred-y_train)**2)
    if i % 100 ==0:
        print(f'{i}/{epochs} W={W.squeeze().detach()} loss = {loss.item()}')

    #역전파 : 피분
    grad_y_pred = 2.0 * (y_pred-y_train)     #(y_pred-y)**2  y_pred 로 미분
    grad_W = x_train.T.mm(grad_y_pred)       # x * w + b 를 w 로 편미분 ->  x.T * grad_y_pred
    grad_b = grad_y_pred.mean()              # x * w + b 를 b 로 편미분 ->  grad_y_pred * 1

    # 매개변수 업데이트
    # w 에 require_grad 있는 경우  W-= learning_rate * grad_W 수행되지 않음
    W -= learning_rate * grad_W
    b -= learning_rate * grad_b



print(f' {x_train[1]} 의 실제값 : {y_train[1]}예측값 { x_train[1] @  W +b.item()} ')