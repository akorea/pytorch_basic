import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 선택 분류 
# 가설: h = exp(wx+b)/sum(exp(wx+b))
# 비용 함수 : loss = (y_one_hot * -log(h)).sum().mean()
# 역전파를 위한 미분 : 1-h
# https://ko.numberempire.com/derivativecalculator.php 미분 계산기 참고



################## 데이터 셋 #########################################################    
#  테스트를 위한 데이터로 값이 조작이 되어 있음, 실제 값과 다를 수 있음
#####################################################################################
#
#----------------X-----------------------------|    |---------------Y------------------|
#꽃받침길이 |꽃받침넓이 |  꽃잎길이 | 꽃잎넓이  |      |  vetosa | versicolor| virginica  |
#-----------------------------------------------|   |-----------------------------------|
#5.1        |3.5        |1.4        | 0.2       |   |   0       |    0      |   1       | -> 2
#4.9        |3          |1.4        | 0.2       |   |   0       |    0      |   1       | -> 2
#5.8        |2.6        |4          | 1.2       |   |   0       |    1      |   0       | -> 1
#6.7        |9          |5.2        | 2.3       |   |   1       |    0      |   0       | -> 0
#5.6        |2.8        |4.9        | 2         |   |   1       |    0      |   0       | -> 0
			

torch.manual_seed(1)


x_data = [[5.1,3.5,1.4,0.2],[4.9,3,1.4,0.2],[5.8,2.6,4,1.2],[6.7,9,5.2,2.3],[5.6,2.8,4.9,2]]
y_data = [2,2,1,0,0] #[0,0,1],[0,0,1],[0,1,0],[1,0,0],[0,1,0]


x_train = torch.FloatTensor(x_data)
y_train = torch.LongTensor(y_data)


xdim = x_train.shape[1]
ydim = len(set(y_data))


class SoftmaxClass():
    def __init__(self,xdim, hdim, ydim):
        super().__init__()
        self.lin1 = self.linear(xdim, hdim)
        self.lin2 = self.linear(hdim, ydim)

    def linear(idim,odim):
        return linear_func
    def forward(self, x):
        net= self.lin1(x)
        net = self.sigmoid(net)
        net = self.lin2(net)
        net = self.softmax(net)
        return net
    
    def backward():
        print('backward')
    
    def optimizer():
        print('optimizer')

if __name__ == '__main__':

    # ytrain [2,2,1,0,0] -> [[0,0,1],[0,0,1],[0,1,0],[1,0,0],[0,1,0]] 로 변경
    y_one_hot = one_hot_vector(y_train, hot_dim)

    model = SoftmaxClass(xdim, ydim)


    epochs = 1000
    for i in range(epochs + 1):
        # 1. 가설 계산  ----> 모델 계산
        y_pred = model.forward(x_train)
        # 2. 비용함수 계산
        loss = model.loss(y_pred, y_train)

        # 3. 비용함수 역전파
        delta= model.backward()
        # 4. 옵티마이저로 매개변수 갱신

        model.optimizer(delta)

        if i % 100 == 0:
            print(f'Epoch :{i}/{epochs}  loss:{loss.item()}')


    y_pred = model.forward(x_train)

    prediction = torch.argmax(y_pred[1])

    print(f' {x_train[1]} 의 실제값 : {y_train[1]} 예측값: { prediction} ')
