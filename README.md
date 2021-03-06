# Pytorch 기초 

Pytorch 를 이용하여 기본 모델을 구현한 프로젝트로 CPU 환경에서 프로그램을 실행할 수 있습니다. 이 문서는 계속 업데이트 할 예정입니다.

### 개발 환경
* python : 3.8.5
* torch  : 1.8.0
* matplotlib 3.3.2
* pandas : 1.2.2 
* torchvision : 0.9.0
  
## 0. 기초 코드
* 텐서 기본 조작법을 모아둔 소스 코드
* 소스 코드

    | 파일           |  설명           |
    | ---------------|:----------------------- |
    | [0.pytorch.py](./0.pytorch.py)   | 텐서 나눗셈, 합, 텐서 합치기 , 자동 미분    |
 

## 1. 단순 선형 회귀 
* 학습 시간에 따라 점수가 달라진다.  1시간 공부한 학생은 점수가 20점이다.
* 점수 = 공부시간 * w + b  라고 할때, w 와 b 를 매개 변수 값을 인공지능 학습을 통해 알아보자.

    | 학습 시간(y)  | 점수(x)         |
    | ----------|:------------------ |
    | 1    | 20   |
    | 2    | 40   |
    | 3    | 60   |

* 소스 코드
    | 파일           |  설명           |
    | ---------------|:----------------------- |
    | [1.linear_regression.py](./1.linear_regression.py)|  학습시간, 점수의 데이터가 행렬로 구성   |
    | [1.linear_regression_vector.py](./1.linear_regression_vector.py)|  학습시간, 점수의 데이터가 벡터로 구성     |

## 2.다중 선형 회귀 
  * 소스 코드
    | 파일           |  설명           |
    | ---------------|:----------------------- |
    | [2.multi_linear_regression.py](./2.multi_linear_regression.py) |  학습시간, 과외시간, 학원 시간을 과 점수를 행렬로 구성   |

## 3. 역전파 미분 함수 
  * 소스 코드
    | 파일           |  설명           |
    | ---------------|:----------------------- |
    | [3.grad_matrix.py](./3.grad_matrix.py)| 역전파 행렬 미분 기능 구현  |
    | [3.grad_vector.py](./3.grad_vector.py) | 역전파 벡터 미분 기능 구현   |

## 4. 다층 퍼셉트론 (MLP) 
  * 소스 코드
    | 파일           |  설명           |
    | ---------------|:----------------------- |
    | [4.multi_layer_perceptron.py](./4.multi_layer_perceptron.py) | MLP를 비용함수, 역전파 기능 구현   |
    | [4.multi_layer_perceptron_class.py](./4.multi_layer_perceptron_class.py)    |  MLP를 파이토치 내장 함수로 구현    |
## 5. 미니배치
* 소스 코드
    | 파일           |  설명           |
    | ---------------|:----------------------- |
    | [5.mini_batch.py](./5.mini_batch.py) | DataLoader를 이용하여 데이터를 미니 배치 단위로 나눠서 데이터 학습 시킴   |
## 6. 이진 분류
* 소스 코드
    | 파일           |  설명           |
    | ---------------|:----------------------- |
    | [6.logistic_regression.py](./6.logistic_regression.py) | 이진 분류의 비용 함수, 역전파 기능 구현   |
    | [6.logistic_regression_multi_layer.py](./6.logistic_regression_multi_layer.py)  | 이진 분류의 다층 레이어 모델 비용 함수, 역전파 기능 구현    |
    | [6.logistic_regression_multi_layer_class.py](./6.logistic_regression_multi_layer_class.py) | 이전 분류를 파이토치 내장 함수로 구현      |
## 7. 선택 분류
* 소스 코드
    | 파일           |  설명           |
    | ---------------|:----------------------- |
    | [7.softmax_data.xlsx](./7.softmax_data.xlsx) | 선택 분류 데이터셋을 시각화한 엑셀 문서   |
    | [7.softmax_regression.py](./7.softmax_regression.py) | 선택분류의  비용 함수, 역전파 기능 구현    |
    |  [7.softmax_regression_adam.py](./7.softmax_regression_adam.py) | 선택 분류의 비용함수, 역전파, 아담 옵티마이저 기능 구현   |
    |  [7.softmax_regression_class.py](./7.softmax_regression_class.py) |  선택 분류를 파이토치 내장 함수로 구현   |
    |  [7.softmax_regression_multi_layer.py](./7.softmax_regression_multi_layer.py) | 선택 분류 다층 레이어 모델을 파이토치 내장 함수로 구현  |
## 8. 데이터 셋
* 소스 코드
    | 파일           |  설명           |
    | ---------------|:----------------------- |
    | [8.mnist_dataset.py](./8.mnist_dataset.py) | 선택 분류로 Mnist 이미지 분류하기 (CPU 환경에서 실행할 수 있도록 데이터를 일부만 사용하여 정확도는 조금 떨어짐) |
## 9. 시각화
* TBD


## 참고 자료 
파이썬 날코딩으로 알고짜는 딥러닝 https://github.com/KONANtechnology/Academy.ALZZA

Pytorch 로 시작하는 딥러닝 입문 https://wikidocs.net/55580