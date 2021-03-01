import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 다중 분류
# 가설: h = exp(wx+b)/sum(exp(wx+b))
# 비용 함수 : loss = (y_one_hot * -log(h)).sum().mean()
# 역전파를 위한 미분 : 1-h
# https://ko.numberempire.com/derivativecalculator.php 미분 계산기 참고


def softmax(x) : 
    exp_x = torch.exp(x-x.max())  # 각각의 원소에 최댓값을 뺀다 (이를 통해 overflow 방지)
    y = exp_x / exp_x().sum()
    return y

def loss(Y, y_pred):
    y_one_hot = torch.zeros_like(hypothesis) 
    y_one_hot.scatter_(1, y.unsqueeze(1), 1)
    return (y_one_hot * -torch.log(h)).sum(dim=1).mean()

