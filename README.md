# Pytorch 기초 

1. linear1_numpy : $y = x^2+2$ 함수로 데이터를 생성하여 Linear Regression 모델로 데이터를 학습시켜 그 결과를 그래프로 표시
* 입력층 dim=1: x   ----- >  출력층 dim=1: y 으로 모델을 구성할때는 전혀 학습이 되지 않음
* 입력층 dim=1: x   --->  은닉층 dim:=10   --->   출력층 dim=1: y 으로 모델을 학습이 됨 (손실 함수 : MSE, 최적화 함수:SDG)
