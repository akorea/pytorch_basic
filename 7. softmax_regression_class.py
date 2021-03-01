import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 다중 분류
# 가설: h = exp(wx+b)/sum(exp(wx+b))
# 비용 함수 : loss = (y_one_hot * -log(h)).sum().mean()
# 역전파를 위한 미분 : 1-h
# https://ko.numberempire.com/derivativecalculator.php 미분 계산기 참고

torch.manual_seed(1)

x_data = [[5.1,3.5,1.4,0.2],[4.9,3,1.4,0.2],[5.8,2.6,4,1.2],[6.7,9,5.2,2.3],[5.6,2.8,4.9,2]]
#[0,0,1],[0,0,1],[0,1,0],[1,0,0],[0,1,0]
y_data = [2,2,1,0,0]



x_train = torch.FloatTensor(x_data)
y_train = torch.LongTensor(y_data)


xdim = x_train.shape[1]
ydim = len(set(y_data))


class SoftmaxClass(nn.Module):
    def __init__(self,xdim, ydim):
        super().__init__()
        self.linear = nn.Linear(xdim, ydim)

    def forward(self, x):
        return self.linear(x)


model = SoftmaxClass(xdim, ydim)
# opt 설정
opt = optim.SGD(model.parameters(), lr=0.1)

epochs = 1000
for i in range(epochs + 1):
    # 1. 가설 계산  ----> 모델 계산
    y_pred = model(x_train)
    # 2. 비용함수 계산
    loss = F.cross_entropy(y_pred, y_train)

    # 3. 미분값 초기화    
    opt.zero_grad()
    # 4. 비용함수 역전파
    loss.backward()
    # 5. 옵티마이저로 매개변수 갱신
    opt.step()

    if i % 100 == 0:
        print(f'Epoch :{i}/{epochs}  loss:{loss.item()}')


y_pred = model(x_train)

prediction = torch.argmax(y_pred[1])

print(f' {x_train[1]} 의 실제값 : {y_train[1]} 예측값: { prediction} ')

#파라미터 출력
with torch.no_grad():
    n_param = 0
    for p_idx,(param_name,param) in enumerate(model.named_parameters()):
        param_data = param.detach().cpu()
        n_param += len(param_data.reshape(-1))
        print ("[%d] name:[%s] shape:[%s]."%(p_idx,param_name,param_data.shape))
        print ("    val:%s"%(param_data.reshape(-1)))
    print ("Total number of parameters:[%s]."%(format(n_param,',d')))
