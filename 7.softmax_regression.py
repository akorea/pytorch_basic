import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 다중 분류
# 가설: h = exp(wx+b)/sum(exp(wx+b)) =softmax(wx+b)
# 비용 함수 : F.cross_entropy= The negative log likelihood loss(F.nll_loss) + log softmax (F.log_softmax) 
#            loss = (y_one_hot * -log(h)).sum().mean()
# 역전파를 위한 미분 : 
# loss 함수 :h- y_one_hot =  softmax(wx+b) - y_one_hot

# https://ko.numberempire.com/derivativecalculator.php 미분 계산기 참고
# http://www.matrixcalculus.org/ 행렬 미분 
#
# 참고 사이트: 
# numpy 구현 코드 : https://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/
# API 문서: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
# API 문서: https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
# torch 기본문서: https://wikidocs.net/60572
# pytorch 코드 : https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/loss.py
def softmax(x) : 
    exp_x = torch.exp(x-x.max())             # 각각의 원소에 최댓값을 뺀다 (이를 통해 overflow 방지)
    y = exp_x/exp_x.sum(dim=1).unsqueeze(1)  #열 값을 합하여 그 값을 나눔 
    """
    [[0.1, 0.1, 0.1],     [[0.1, 0.1, 0.1],   / [[0.3], 
     [0.2, 0.2, 0.2],  ->  [0.2, 0.2, 0.2],   /  [0.6], 
     [0.3, 0.3, 0.3]]      [0.3, 0.3, 0.3]]   /  [0.9]] 
    """
    return y



def one_hot_vector(y, dim):
    y_one_hot = torch.zeros(dim) 
    y_one_hot.scatter_(1, y.unsqueeze(1), 1)
    return y_one_hot


def softmax_derivative(x):
    # 다시 확인 필요
    s = x.reshape(-1,1)
    return torch.diagflat(s) - torch.mm(s, s.T)

def softmax_derv(x, y):
    mb_size, nom_size = x.shape
    derv = np.ndarray([mb_size, nom_size, nom_size])
    for n in range(mb_size):
        for i in range(nom_size):
            for j in range(nom_size):
                derv[n, i, j] = -y[n,i] * y[n,j]
            derv[n, i, i] += y[n,i]
    return derv


################## 데이터 셋 #########################################################    
#  테스트를 위한 데이터로 값이 조작이 되어 있음, 실제 값과 다를 수 있음
#####################################################################################
#
#----------------X---------------------------|    |---------------Y----------------|
#꽃받침길이	| 꽃받침넓이 |	꽃잎길이 | 꽃잎넓이 |    |  vetosa | versicolor| virginica|
#--------------------------------------------|    |--------------------------------|
#5.1	    |3.5	    |1.4	   | 0.2     |    |   0     |    0       |    1    | -> 2
#4.9	    |3	        |1.4	   | 0.2     |    |   0     |    0       |    1    | -> 2
#5.8	    |2.6	    |4	       | 1.2     |    |   0     |    1       |    0    | -> 1
#6.7	    |9	        |5.2	   | 2.3     |    |   1     |    0       |    0    | -> 0
#5.6	    |2.8	    |4.9	   | 2       |    |   1     |    0       |    0    | -> 0
			



x_data = [[5.1,3.5,1.4,0.2],[4.9,3,1.4,0.2],[5.8,2.6,4,1.2],[6.7,9,5.2,2.3],[5.6,2.8,4.9,2]]
y_data = [2,2,1,0,0]  #one_hot [0,0,1],[0,0,1],[0,1,0],[1,0,0],[0,1,0]


torch.manual_seed(1)

x_train = torch.FloatTensor(x_data)
y_train = torch.LongTensor(y_data)

w = torch.randn(4,3)
b= torch.randn(3)


dim = (5,3)

y_one_hot = one_hot_vector(y_train, dim)

epochs = 100
learning_rate = 0.01

for i in range(epochs+1):
    # 1. 가설 계산 
    z= x_train.mm(w) +b
    y_pred = softmax(z)
    # softmax 맞는지 내장 함수로 알 수 있음
    #y_pred1 = F.softmax(z,dim=1)


    # 2. 비용함수 계산
    # 1) 출력값을 one hot vector 로 변환 
    # -> softmax 가설에 log 를 씌움 
    # -> Negative log likelihood 로 비용 함수 계산
    #  1e-7 는 log(0) 이 무한대로 가는 문제를 방지하기 위해 설정
    loss = (y_one_hot * -torch.log(y_pred + 1.0e-10)).sum(dim=1).mean()
    
    # 2) 비용함수 : 내장 함수 cross_entropy 로 계산
    loss1 = F.cross_entropy(z, y_train)
    # 3) 비용 함수 : 내장 함수 Negative log likelihood + log softmax 로 계산 
    loss2= F.nll_loss(F.log_softmax(z, dim=1), y_train)

    if i % 10 ==0:
        # loss, loss1, loss2 는 유사하나 loss 값이 발산하는 경우 차이가 발생
        # loss 에서 1.0e-10 더한 것이 원인임
        print(f'{i}/{epochs}  loss = {loss.item()}, \
        cross_entropy={loss1.item()}, \
        nll+log_softmax={loss2.item()}')

    # 3. 비용함수 역전파 (계산이 잘못됨 다시 계산해야 함)
    grad_y_pred = z- y_train
    grad_z = grad_y_pred *softmax_derivative(z)
    grad_w =  x_train.T.mm(grad_z)
    grad_b = grad_z.sum(dim=1).mean()


    #4.파라미터 업데이트
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

    
