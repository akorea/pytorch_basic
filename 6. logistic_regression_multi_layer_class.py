import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 이진 분류
# 가설: h = 1 /(1+ exp-(wx+b)) = sigmoid(wx+b) 
# 비용 함수 : loss = -(y * log(h) + (1 - y) * log(1 - h)).mean()
# 역전파를 위한 미분
#  1) h* log(y) 미분= h/y 
#     h * log(y) + (1-h) * log(1-y)  미분 = h/y - (h-1)/(y-1) = (y-h)/y(y-1)
#  2) sigmoid(x)의 미분 = sigmoid(x) (1-sigmoid(x))
# https://ko.numberempire.com/derivativecalculator.php 미분 계산기 참고
#   X       Y
# 1,  2     0 
# 2,  3     0 
# 3,  1     0 
# 4,  3     1 
# 5,  3     1 
# 6,  2     1 

torch.manual_seed(1)
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

xdim = 2
hdim = 100
ydim =1 


class LogistClass(nn.Module):
    def __init__(self, name, xdim, hdim, ydim):
        super(LogistClass,self).__init__()  
        self.name =name
        self.xdim = xdim 
        self.hdim = hdim
        self.ydim =  ydim
        self.lin1 = nn.Linear(xdim,hdim, bias=True)
        self.lin2 = nn.Linear(hdim, ydim,bias=True)
        self.sigmoid = nn.Sigmoid()
        self.init_param()
    
    def init_param(self):
        nn.init.kaiming_normal_(self.lin1.weight)
        nn.init.zeros_(self.lin1.bias)
        nn.init.kaiming_normal_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)
    
    def forward(self, x):
        net = self.lin1(x)
        net = self.sigmoid(net)
        net = self.lin2(net)
        net = self.sigmoid(net)
        
        return net

model = LogistClass('logist',xdim,hdim, ydim)
model.init_param()

opt = optim.Adam(model.parameters(), lr = 0.1)

epochs =9000
learning_rate = 0.1
for i in range(epochs+1):
    # 순전파 단계: 예측값 y를 계산
    y_pred= model.forward(x_train)

    # 손실(loss)을 계산하고 출력함
    #loss 함수
    loss_out = F.binary_cross_entropy(y_pred, y_train)

    if i % 100 ==0:
        print(f'{i}/{epochs}  loss = {loss_out.item()}')

    # 손실에 따른 w1, b의 변화도를 계산하고 역전파함
    opt.zero_grad()
    loss_out.backward()

    #파라미터 업데이트
    opt.step()

with torch.no_grad():
    y_pred = model.forward(x_train)
    prediction = 1 if y_pred[1] >= torch.FloatTensor([0.5])  else 0


    print(f'{x_train[1]} 의 실제값 : {y_train[1]} 예측값: {prediction}')