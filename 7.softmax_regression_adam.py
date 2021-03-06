import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

# 선택 분류 : loss 함수, 미분 함수, 아담 옵티마이저 구현 
#            -> 아담을 사용하면, learning rate 를 조금 증가시켜도 loss 값이 발산하지 않음 
# 가설: h = exp(wx+b)/sum(exp(wx+b)) =softmax(wx+b)
# 비용 함수 : F.cross_entropy= The negative log likelihood loss(F.nll_loss) + log softmax (F.log_softmax) 
#            loss = (y_one_hot * -log(h)).sum().mean()
# 역전파를 위한 미분 : 
# cross_entropy loss 함수 미분 :h- y_one_hot =  softmax(wx+b) - y_one_hot

# https://ko.numberempire.com/derivativecalculator.php 미분 계산기 참고
# http://www.matrixcalculus.org/ 행렬 미분 
#
# 참고 사이트: 
# 수학 풀이: http://machinelearningmechanic.com/deep_learning/2019/09/04/cross-entropy-loss-derivative.html
# numpy 구현 코드 : https://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/
# API 문서: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
# API 문서: https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
# torch 기본문서: https://wikidocs.net/60572
# pytorch 코드 : https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/loss.py


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
			



x_data = [[5.1,3.5,1.4,0.2],[4.9,3,1.4,0.2],[5.8,2.6,4,1.2],[6.7,9,5.2,2.3],[5.6,2.8,4.9,2]]
y_data = [2,2,1,0,0]  #one_hot [[0,0,1],[0,0,1],[0,1,0],[1,0,0],[0,1,0]]


torch.manual_seed(1)

x_train = torch.FloatTensor(x_data)
y_train = torch.LongTensor(y_data)

w = torch.randn((4,3))
b= torch.randn(3)
pm ={}

epochs = 10000
learning_rate = 0.01


def softmax(x) : 
    exp_x = torch.exp(x-x.max())             # 각각의 원소에 최댓값을 뺀다 (이를 통해 overflow 방지)
    y = exp_x/exp_x.sum(dim=1).unsqueeze(1)  #열 값을 합하여 그 값을 나눔 
    """         y                            exp_x            exp_x.sum(dim=1).unsqueeze(1)
    [[0.1/0.3+ 0.1/0.3+ 0.1/0.3],      [[0.1, 0.1, 0.1],    [[0.3], 
     [0.2/0.6+ 0.2/0.6+ 0.2/0.6],  =   [0.2, 0.2, 0.2],   /  [0.6], 
     [0.3/0.9+ 0.3/0.9+ 0.3/0.9]]      [0.3, 0.3, 0.3]]      [0.9]] 
    """
    return y



def one_hot_vector(y, dim):
    y_one_hot = torch.zeros(dim) 
    y_one_hot.scatter_(1, y.unsqueeze(1), 1)
    return y_one_hot


def softmax_derivative(x):
    # 증명 필요 , 현재 사용하지 않음
    s = x.reshape(-1,1)
    return torch.diagflat(s) - torch.mm(s, s.T)



#파이썬 날코딩으로 알고짜는 딥러닝 코드 
#https://github.com/KONANtechnology/Academy.ALZZA 
#https://d2l.ai/chapter_optimization/adam.html

def adam_update(pm, key, delta):
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1.0e-8

    #v : EMA of gradient squares
    #m:  momentum
    mkey, vkey, step = 'm' + key, 'v' + key, 'n' + key
    if key == 'w' : 
        size = w.shape
    else:
        size = b.shape 

    if mkey not in pm:
        pm[mkey] = torch.zeros(size)
        pm[vkey] = torch.zeros(size)
        pm[step] = 0
    
    
    #update biased first moment estimate
    m = pm[mkey] = beta1 * pm[mkey] + (1 - beta1) * delta
    #Update biased second raw moment estimate
    v = pm[vkey] = beta2 * pm[vkey] + (1 - beta2) * (delta * delta)  
    
    pm[step] += 1
    #Compute bias-corrected first moment estimate (weighted average)
    m = m / (1 - math.pow(beta1, pm[step]))
    #Compute bias-corrected second raw moment estimate 
    v = v / (1 - math.pow(beta2, pm[step])) 
    
    return m / (torch.sqrt(v)+epsilon)







if __name__ == '__main__':

    hot_dim = (5,3)
    # ytrain [2,2,1,0,0] -> [[0,0,1],[0,0,1],[0,1,0],[1,0,0],[0,1,0]] 로 변경
    y_one_hot = one_hot_vector(y_train, hot_dim)

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
        loss = (y_one_hot * -torch.log(y_pred + 1e-7)).sum(dim=1).mean()

        # 2) 비용함수 : 내장 함수 cross_entropy 로 계산
        loss1 = F.cross_entropy(z, y_train)
        # 3) 비용 함수 : 내장 함수 Negative log likelihood + log softmax 로 계산 
        loss2= F.nll_loss(F.log_softmax(z, dim=1), y_train)

        if i % 100 ==0:
            # loss, loss1, loss2 는 유사하나 loss 값이 발산하는 경우 차이가 발생
            # loss 에서 1.0e-10 더한 것이 원인임
            print(f'{i}/{epochs}  loss = {loss.item()}, \
            cross_entropy={loss1.item()}, \
            nll+log_softmax={loss2.item()}')

        # 3. 비용함수 역전파 (수정 필요 : loss 값이 큼)
        grad_z = (z- y_one_hot)
        grad_w =  x_train.T.mm(grad_z) /y_one_hot.shape[1]
        grad_b = grad_z.sum(dim=1).mean()


        #4.파라미터 업데이트
        w -= learning_rate * adam_update(pm,'w',grad_w)
        b -= learning_rate * adam_update(pm,'b',grad_b)


    z= x_train.mm(w) +b
    y_pred = torch.argmax(softmax(z),dim=1)

    #예측값은 맞으나 Loss 가 큼
    #현재 loss= 0.5515404939651489,-> epoch 를 늘리면 0.55 까지 줄일 수 있음
    # Adam 옵티마이저를 사용하면 , learning_rate 를 늘려도 loss 값이 발산하지 않음 
    print(f'{x_train[1]} 의 실제값 : {y_train[1]} 예측값: {y_pred[1]}')