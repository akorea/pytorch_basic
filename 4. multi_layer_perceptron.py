import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



#H(x)= WX+b 행렬 미분 함수 직접 구현하여, 매개 변수 (W,b) 구하기
#      X       Y
# 1,  1,  1    12
# 2,  2,  2    24
# 3,  3,  3    36

torch.manual_seed(1)
x_train= torch.FloatTensor([[1,1,1],[2,2,2],[3,3,3]])  # 3 * 3
y_train = torch.FloatTensor([[12], [24], [36]])  # 3 * 1

#차원은 열의 개수 

i_dim = 3
h_dim = 100
o_dim = 1

w1 = torch.randn(i_dim,h_dim)
w2 = torch.randn(h_dim,o_dim)

epochs =9000
learning_rate = 1e-6
for i in range(epochs+1):
    # 순전파 단계: 예측값 y를 계산
    h=x_train.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)
    # 손실(loss)을 계산하고 출력함
    loss = (y_pred - y_train).pow(2).mean().item()
    if i % 100 ==0:
        print(f'{i}/{epochs}  loss = {loss}')

    # 손실에 따른 w1, w2의 변화도를 계산하고 역전파함
    grad_y_pred = 2.0 * (y_pred - y_train)
    grad_w2 = h_relu.T.mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.T)
    grad_h = grad_h_relu.clone()
    grad_h[h<0] =0
    grad_w1= x_train.T.mm(grad_h)
    
    #파라미터 업데이트
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2


h = x_train.mm(w1)
h_relu = h.clamp(min=0)
y_pred = h_relu.mm(w2)


## 예측값을 높이기 위해서는
# 1. torch.manual_seed 설정함 -> w1, w2 의 매개변수 초기값에 따라 성능이 차이가 많음
# 2. epoch 늘리기
# 3. h_dim 사이즈 늘리기 
#   1) 4 일 경우 loss : 0.0008823976386338472  
#   2) 100일 경우 loss : 1.5570549294352531e-09
# 4. learning_rate 작게 할 수록 정답에 가까워 질 확률이 높아짐 (ecoch를 늘려야 함)
print(f'{x_train[1]} 의 실제값 : {y_train[1]} 예측값: {y_pred[1]}')