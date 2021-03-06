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

###test 데이터 
# cpu 에서 돌리기 위해 테스트 데이터 크기 줄임
# mnist_train.data.shape : 10000 -> 200 개, 28 가로 ,28 세로
n_test = 200
mnist_test.data = mnist_test.data[:n_test]
mnist_test.targets = mnist_test.targets[:n_test]


#배치 사이즈 
BATCH_SIZE = 64
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
# 2. 학습 
# 1) 배치 사이즈 크기로 데이터 호출
# 2) 모델 학습
# 3) 역전파 실행
# 4) 옵티마이저 실행
# 5) 평가
###########################################
def train(dataloader, model, loss_fn, opt):  
        size = len(dataloader.dataset)
        #1) 배치 사이즈 크기로 데이터 호출
        for batch, (x, y) in enumerate(train_iter):
            # 2) 모델 학습
            #x 데이터: 256, 1, 28 ,28 - >256, 28 *28 바꿈 / 정규화 (데이터는 0 ~ 255까지 존재)
            x_data = x.view(-1,28*28).type(torch.float)/255  
            y_pred=model(x_data.to(device)) 
            # 3) 역전파
            opt.zero_grad()
            loss_out = loss_fn(y_pred, y.to(device))
            loss_out.backward()
            # 4) 옵티마이저
            opt.step()
            # 5) 평가
            if batch % 100 == 0:
                loss, current = loss_out.item(), batch * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        

###########################################
# 3. 평가 
# 1) 평가 모드로 전환
# 2) 모델 평가
# 3) 학습 모드로 전환  
###########################################

def evaluate(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    #1) 평가 모드로 전환
    model.eval()
    #2) 모델 평가
    with torch.no_grad():
        for x, y in dataloader:
            x_data = x.view(-1,28*28).type(torch.float)/255  
            y_pred = model(x_data.to(device)) 
            test_loss += loss_fn(y_pred, y.to(device))
            correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    #3) 학습 모드로 전환         
    model.train()        


###########################################
# 4. 테스트 
# 샘플 데이터 10 램덤으로 추출하여 테스트
###########################################
def test(dataset, model):
    mnist_test  = dataset
    with torch.no_grad(): 
        n_sample = 10
        indices =torch.randperm(len(mnist_test.targets))[:n_sample] 
        x_test = mnist_test.data[indices]
        y_test = mnist_test.targets[indices]
        y_pred = model(x_test.view(-1, 28*28).type(torch.float).to(device)/255.)
        y_pred = y_pred.argmax(axis=1)
        plt.figure(figsize=(10,10))
        for idx in range(n_sample):
            plt.subplot(5, 5, idx+1)
            plt.imshow(x_test[idx], cmap='gray')
            plt.axis('off')
            plt.title("Pred:%d, Label:%d"%(y_pred[idx],y_test[idx]))
        plt.show()    


###########################################
# 5. 학습 준비 및 학습
# 1) parameter 정의
# 2) 모델 불러오기
# 3) 비용 함수 지정
# 4) 옵티마이저 지정
# 5) 모델 학습 -> 모델 평가 
# 6) 모델 테스트
###########################################


if __name__ == "__main__":

    #1) parameter 정의
    xdim  = 28 * 28
    hdim = 784
    ydim = 10

    learning_rate = 0.1
    epochs = 10

    #2) 모델 불러오기
    model = MLPModel('mlp', xdim, hdim, ydim).to(device)
    #3) 비용 함수 지정
    loss_fn = nn.CrossEntropyLoss().to(device)
    #4) 옵티마이저 지정
    opt = optim.Adam(model.parameters(),lr=learning_rate)
    #5) 모델 학습 및 평가
    for i in range(epochs+1):
        train(train_iter, model, loss_fn,opt)
        evaluate(test_iter, model, loss_fn)
    #6) 모델 테스트
    test(mnist_test,model)
