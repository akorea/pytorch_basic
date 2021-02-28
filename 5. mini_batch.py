import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더

# 전체 데이터를 미니 배치 단위로 나눠서 학습하는 것이 성능을 높임
# 미니 배치 학습을 하게되면 미니 배치만큼만 가져가서 미니 배치에 대한 대한 비용를 계산 및  경사 하강법을 수행 
# 그리고 다음 미니 배치를 가져가서 경사 하강법을 수행하고 마지막 미니 배치까지 이를 반복
# 전체 데이터에 대한 학습이 1회 끝나면 1 에포크(Epoch)가 됨

torch.manual_seed(1)
x  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  90], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

#dataset 만듬
dataset = TensorDataset(x, y)

#dataloder 로 배치 사이즈 2 설정하여 데이터를 가져옴
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

#차원은 열의 개수 

i_dim = 3
o_dim = 1

w1 = torch.randn(i_dim,o_dim)

epochs =100
learning_rate = 1e-6
for i in range(epochs+1):
    for batch_idx, samples in enumerate(dataloader):
        # 순전파 단계: 예측값 y를 계산
        x_train, y_train = samples
        y_pred=x_train.mm(w1)

        # 손실(loss)을 계산하고 출력합니다.
        loss = (y_pred - y_train).pow(2).mean().item()
        if i % 10 ==0:
            print(f'epoch: {i}/{epochs} batch_id:{batch_idx+1} loss = {loss} ')

        # 손실에 따른 w1, w2의 변화도를 계산하고 역전파함
        grad_y_pred = 2.0 * (y_pred - y_train)
        grad_w1 = x_train.T.mm(grad_y_pred)
        
        #파라미터 업데이트
        w1 -= learning_rate * grad_w1



y_pred = x.mm(w1)


## 예측값을 높이기 위해서는
# 1. torch.manual_seed 설정함 -> w1, w2 의 매개변수 초기값에 따라 성능이 차이가 많음
# 2. epoch 늘리기
# 3. h_dim 사이즈 늘리기 
#   1) 4 일 경우 loss : 0.0008823976386338472  
#   2) 100일 경우 loss : 1.5570549294352531e-09
# 4. learning_rate 작게 할 수록 정답에 가까워 질 확률이 높아짐 (ecoch를 늘려야 함)
print(f'{x[1]} 의 실제값 : {y[1]} 예측값: {y_pred[1]}')