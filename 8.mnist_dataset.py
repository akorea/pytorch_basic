import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import torch.optim as optim
import torch.nn.functional as F
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# GPU, CPU 사용 여부 확인
USE_CUDA = torch.cuda.is_available() 
device = torch.device("cuda" if USE_CUDA else "cpu") 
print("device:", device)

random.seed(1)
torch.manual_seed(1)
if device == 'cuda':
    torch.cuda.manual_seed_all(1)        


###########################################
# 2. Model 클래스 정의
# 1) __init__: 학습 레이어 구성  
# 2) forward : 모델 학습 함수
###########################################
class MLPModel(nn.Module):
    def __init__(self, name, xdim, hdim, ydim):
            super().__init__()
            self.lin1= nn.Linear(xdim, hdim, bias=True)
            self.lin2= nn.Linear(hdim, ydim, bias=True)

    def forward(self, x):
        net = self.lin1(x)
        net = F.relu(net)
        net = self.lin2(net)
        return net 

###########################################
# 1. 데이터 셋 정의
###########################################
#https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py

###train 데이터
# 이미지(data)  : 60000 개, 28 가로 ,28 세로  
# 라벨(target) :   60000 개 (0 ~ 9)
# MNIST train dataset ----  | -- data size : (60000, 28, 28) 
#                           | -- target size(60000) 

# MNIST test dataset ----  | -- data size : (10000, 28, 28) 
#                          | -- target size(10000) 

mnist_train = dsets.MNIST(root='MNIST_data/', train=True, \
                        transform=transforms.ToTensor(),\
                        download=True)               


mnist_test = dsets.MNIST(root='MNIST_data/',train=False,\
                        transform=transforms.ToTensor(),\
                        download=True)
###train 데이터 
# cpu 에서 돌리기 위해 데이터 크기 줄임
# mnist_train.data.shape : 60000 -> 10000 개, 28 가로 ,28 세로
n_train = 10000
mnist_train.data = mnist_train.data[:n_train]
mnist_train.targets = mnist_train.targets[:n_train]

###test 데이터 
# cpu 에서 돌리기 위해 데이터 크기 줄임
# mnist_train.data.shape : 10000 -> 200 개, 28 가로 ,28 세로
n_test = 200
mnist_test.data = mnist_test.data[:n_test]
mnist_test.targets = mnist_test.targets[:n_test]


#배치 사이즈 
BATCH_SIZE = 100
train_iter = torch.utils.data.DataLoader(mnist_train,\
                                        batch_size=BATCH_SIZE,\
                                        shuffle=True,\
                                        drop_last=True)
test_iter = torch.utils.data.DataLoader(mnist_test,\
                                        batch_size=BATCH_SIZE,\
                                        shuffle=True,\
                                        drop_last=True)
#https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataloader.py

# print(train_iter.dataset.data.shape)
# print(test_iter.dataset.data.shape)

# train_iter.dataset.train_data 크기 : [batch_size, 1, 28, 28]
# train_iter.dataset.train_labels 크기 : [batch_size]

          

###########################################
# 2. 학습 준비
# 1) parameter 정의
# 2) 모델 불러오기
# 3) 비용 함수 지정
# 4) 옵티마이저 지정
###########################################

#1) parameter 정의
xdim  = 28 * 28
hdim = 784
ydim = 10

learning_rate = 0.1
epochs = 200

#2) 모델 불러오기
model = MLPModel('mlp', xdim, hdim, ydim).to(device)
#3) 비용 함수 지정
loss_func = nn.CrossEntropyLoss().to(device)
#4) 옵티마이저 지정
opt = optim.Adam(model.parameters(),lr=learning_rate)

###########################################
# 3. 학습
# 1) 학습 반복
# 2) 모델 학습
# 3) 역전파 실행
# 4) 옵티마이저 실행
# 5) 평가
###########################################

# 1) 학습 반복
for i in range(epochs+1):
    loss_val_avg = 0
    for x, y in train_iter:
        # 2) 모델 학습
        #x 데이터: 256, 1, 28 ,28 - >256, 28 *28 바꿈 / 정규화 (데이터는 0 ~ 255까지 존재)
        x_data = x.view(-1,28*28).type(torch.float)/255  
        y_pred=model.forward(x_data.to(device)) 
        # 3) 역전파
        opt.zero_grad()
        loss_out = loss_func(y_pred, y.to(device))
        loss_out.backward()
        # 4) 옵티마이저
        opt.step()
        # 5) 평가
        loss_val_avg += loss_out/len(train_iter)
    

    if i % 10 == 0:
        print(f'Epoch :{i}/{epochs}  loss:{loss_val_avg}')



###########################################
# 4. 테스트 
# 샘플 데이터 10 램덤으로 추출하여 테스트
# cpu 에서 돌리기 위해 1/6 데이터만을 학습시켰음 
# 정확도 대략 90% 
###########################################
with torch.no_grad(): 
    n_sample = 10
    indices =torch.randperm(len(mnist_test.targets))[:n_sample] 
    x_test = mnist_test.data[indices]
    y_test = mnist_test.targets[indices]
    y_pred = model.forward(x_test.view(-1, 28*28).type(torch.float).to(device)/255.)
    y_pred = y_pred.argmax(axis=1)
    plt.figure(figsize=(10,10))
    for idx in range(n_sample):
        plt.subplot(5, 5, idx+1)
        plt.imshow(x_test[idx], cmap='gray')
        plt.axis('off')
        plt.title("Pred:%d, Label:%d"%(y_pred[idx],y_test[idx]))
    plt.show()    
print ("Done")
