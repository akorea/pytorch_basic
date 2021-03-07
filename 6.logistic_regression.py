import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 이진 분류 모델, 미분 함수 직접 구현
# 가설: h = 1 /(1+ exp-(wx+b)) = sigmoid(wx+b) 
# 비용 함수 : loss = -(y * log(h) + (1 - y) * log(1 - h)).mean()
# 역전파를 위한 미분
#  1) y* log(h) 미분= y/h 
#     y * log(h) + (1-y) * log(1-h)  미분 = y/h - (y-1)/(h-1) = (h-y)/h(h-1)
#  2) sigmoid(x)의 미분 = sigmoid(x) (1-sigmoid(x))
# https://ko.numberempire.com/derivativecalculator.php 미분 계산기 참고

#   데이터 셋
#   X    |   Y
#---------------
# 1,  2  |   0 
# 2,  3  |   0 
# 3,  1  |   0 
# 4,  3  |   1 
# 5,  3  |   1 
# 6,  2  |   1  

torch.manual_seed(1)
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)



#차원은 열의 개수 
w = torch.randn(2, 1)
b = torch.randn(1)

def sigmoid(x):
    return 1/(1+torch.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1- sigmoid(x))

epochs =9000
learning_rate = 0.1
for i in range(epochs+1):
    # 순전파 단계: 예측값 y를 계산
    y_pred=sigmoid(x_train.mm(w)+b)

    # 손실(loss)을 계산하고 출력함
    #F.binary_cross_entropy(y_pred, y_train) 함수
    loss = -(y_train * torch.log(y_pred) + (1-y_train) * torch.log(1-y_pred)).mean()

    if i % 100 ==0:
        print(f'{i}/{epochs}  loss = {loss.item()}')

    # 손실에 따른 w1, b의 변화도를 계산하고 역전파함
    espsilon = 1e-7  #0 으로 나눠지지 않도록 더함
    grad_y_pred = (y_pred-y_train) / (y_pred * (1.0-y_pred)+espsilon)
    grad_sigmoid = grad_y_pred * sigmoid_derivative(x_train.mm(w)+b)
    grad_w = x_train.T.mm(grad_sigmoid)
    grad_b = grad_sigmoid.mean()
    
    #파라미터 업데이트
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b


y_pred=sigmoid(x_train.mm(w)+b)

prediction = 1 if y_pred[1] >= torch.FloatTensor([0.5])  else 0


print(f'{x_train[1]} 의 실제값 : {y_train[1]} 예측값: {prediction}')