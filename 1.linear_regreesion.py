import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#랜덤 시드
torch.manual_seed(1)



# 선형 회귀
# 가설: h = wx+b
# 비용 함수 : loss = mean((h-y) **2) 


##### 데이터 셋 ##########
# 공부 시간 |  성적
#   1      |   20
#   2      |   40
#   3      |   60


x_train = torch.FloatTensor([[1], [2],[3]])
y_train = torch.FloatTensor([[20],[40],[60]])

#텐서 출력
print(x_train)
#텐서의 차원 출력
print(x_train.shape)
#텐서서 0 axis의 갯수 출력 -> 여기서는 행의 개수
print(x_train.shape[0])

#매개 변수 W, b 선언 -> 학습을 통해 갱신됨
#requires_grad: 학습을 통해 계속 값이 변경되는 변수
W =torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

#가설 : y = wx +b
y_pred = x_train * W +b
print(y_pred)

#비용 함수 (예측값 - 실제값) **2 의 평균
loss = torch.mean((y_pred-y_train)**2)
print(loss)

#옵티마이저 정의 옵티마이저(매개변수)
optimizer =optim.Adam([W,b],lr=0.01)

epochs = 8000 # 반복
for i in range(epochs+1):
    # 1. 가설 계산  ----> 모델 계산
    y_pred  = x_train * W +b
    # 2. 비용함수 계산
    loss = torch.mean((y_pred-y_train)**2)
  
    # 3. 미분값 초기화    
    optimizer.zero_grad()
    # 4. 비용함수 역전파
    loss.backward()
    # 5. 옵티마이저로 매개변수 갱신
    optimizer.step()

    # 100번마다 로그 출력
    if i % 100 == 0:
        print(f'Epoch :{i}/{epochs} W: {W.item()},  b:{b.item()} loss:{loss.item()}')


print(f' {x_train[1]} 의 실제값 : {y_train[1]}예측값: { x_train[1] * W.item() +b.item()} ')