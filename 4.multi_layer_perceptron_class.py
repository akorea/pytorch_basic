import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



# 다층 퍼셉트론(MLP)를 nn.Module 클래스로 구현
#      X       Y
# 1,  1,  1    12
# 2,  2,  2    24
# 3,  3,  3    36

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device:[{device}]")

torch.manual_seed(1)
x_train= torch.FloatTensor([[1,1,1],[2,2,2],[3,3,3]]).to(device)  # 3 * 3
y_train = torch.FloatTensor([[12], [24], [36]]).to(device)  # 3 * 1

#차원은 열의 개수 

xdim = 3
hdim = 100
ydim = 1


class MLPClass(nn.Module):
    def __init__(self, name, xdim, hdim, ydim): 
       super(MLPClass, self).__init__()  
       self.lin1 = nn.Linear(xdim, hdim, bias=True) 
       self.lin2 = nn.Linear(hdim, ydim ,bias=True)
       self.init_param()

    def init_param(self):
        nn.init.kaiming_normal_(self.lin1.weight)
        nn.init.zeros_(self.lin1.bias)
        nn.init.kaiming_normal_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, x):  
        net = self.lin1(x)
        net = F.relu(net)
        net = self.lin2(net)
        return net


model = MLPClass('mlp',xdim, hdim,ydim)
loss = nn.MSELoss()
optm = optim.Adam(model.parameters(),lr=1e-3)

epochs =2000
model.init_param()

for i in range(epochs+1):
    # 순전파 단계: 예측값 y를 계산
    y_pred=model.forward(x_train)
    # 손실(loss)을 계산하고 출력함
    loss_out = loss(y_pred, y_train)
    if i % 100 ==0:
        print(f'{i}/{epochs}  loss = {loss_out.item()}')

    optm.zero_grad()
    loss_out.backward()
    optm.step()

with torch.no_grad():
    y_pred= model.forward(x_train).detach()
    print(f'{x_train[1]} 의 실제값 : {y_train[1]} 예측값: {y_pred[1]}')
