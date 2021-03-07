from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import numpy as np
import os 
import glob
import ntpath, re
import shutil
from sklearn.model_selection import train_test_split
import tarfile
import urllib.request
import urllib.error
from torch.utils.model_zoo import tqdm
os.environ['KMP_DUPLICATE_LIB_OK']='True'


###########################################
#데이터 셋 준비하는 파일 
#nontMIST 파일을 사이트에서 다운로드 받아,  train test 폴더에 나눠서 복사함
#이미지 파일을 정규화 하여 학습을 할 수 있게 전처리 함
#nontMIST 파일 다음 사이트에서 받을 수 있음
#http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz 
#http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz

###########################################

###########################################
# 1. 압축 파일 다운로드 및 압축 풀기
#1) 압축 파일 다운로드 (데이터 셋 종류 선택)
#2) 압축 풀기:notMNIST_small 에 자동으로 폴더 생성 후 압축이 풀어짐
#3) 다운로드 파일 삭제 
###########################################
def download_and_extract(url):
    file_name = ntpath.basename(url)
    chunk_size = 1024
    #1) 압축 파일 다운로드 (데이터 셋 종류 선택)
    try:
        with open(file_name, "wb") as fh:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "pytorch/vision"})) as response:
                with tqdm(total=response.length) as pbar:
                    for chunk in iter(lambda: response.read(chunk_size), ""):
                        if not chunk:
                            break
                        pbar.update(chunk_size)
                        fh.write(chunk)
        #2) 압축 풀기
        print(f"Extract {file_name}")
        with tarfile.open(file_name, 'r:gz') as tar:
            tar.extractall()
        #3) 다운로드 파일 삭제    
        os.remove(file_name)
    except Exception as e:
        raise e
###########################################
# 2. 압축 푼 폴더의 png 파일을 읽어서 리스트로 저장
###########################################
 
def read_files(data_path):
    file_list = []
    for filepath in glob.iglob(f'{data_path}/**/*.png', recursive=True):
        filelist = re.split("/|\\\\", filepath)
        file_name = filelist[-2]
        file_list.append([filepath, file_name])
    file_list = np.array(file_list)
    return file_list

###########################################
# 2. 압축 폴더의 파일 train, test 로 옮기기
# 1) 파일 리스트를 0.25: test, 0.75 : train 으로 나눔
# 2)notMNIST_small 폴더에 train, test 폴더 생성 
# 3) 압축을 푼 png 파일을 train, test 파일 복사 
###########################################

def move_file(data_path, dataset):
    # 1) 파일 리스트를 0.25: test, 0.75 : train 으로 나눔
    train_image, test_image, train_target, test_target = train_test_split(dataset[:,0], \
                                                                        dataset[:,1], \
                                                                        stratify=dataset[:,1])
    # 2)notMNIST_small 폴더에 train, test 폴더 생성
    os.makedirs(os.path.join(data_path, "train"))
    os.makedirs(os.path.join(data_path, "test"))
    
    for class_name in set(test_target):
        os.makedirs(os.path.join(data_path, "train", class_name))
        os.makedirs(os.path.join(data_path, "test", class_name))
    
    # 3) 압축을 푼 png 파일을 train, test 파일 복사 

    print("Move files to train folder. It's take some times")
 
    for src_path, target_dir in zip(train_image.tolist(), train_target.tolist()):       
        filename = ntpath.basename(src_path)
        target_path = os.path.join(data_path, "train", target_dir, filename)
        shutil.copy(src_path, target_path)
    
    print("Move files to test folder. It's take some times")
    for src_path, target_dir in zip(test_image.tolist(), test_target.tolist()):        
        filename = ntpath.basename(src_path)
        target_path = os.path.join(data_path, "test", target_dir, filename)
        shutil.copy(src_path, target_path) 
    
    for dir in os.listdir(data_path):
        if dir != "train" and  dir != "test":
            rm_dir = os.path.join(data_path, dir)
            shutil.rmtree(rm_dir)

###########################################
# 3. 이미지 출력 함수 
###########################################

def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    return ax



###########################################
# 4.  폴더가 없는 경우 파일 다운로드 및 압축 해제 
###########################################

def preprocess():
    url = "http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz"
    data_path ="notMNIST_small"
    if not os.path.exists(data_path):
        download_and_extract(url)
        file_list=read_files(data_path)
        move_file(data_path,file_list)


###########################################
# DataLoader 생성
#1) Image 정규화 하기
#2) ImageFolder 데이터셋 호출
#3) 데이터셋으로 DataLoader 생성
# - trainloader, testloader 학습으로 사용할 수 있음
###########################################
preprocess()
#1) Image 정규화 하기
trans = transforms.Compose([transforms.Resize((28,28)), \
                            transforms.ToTensor(),\
                            transforms.Normalize([0.485, 0.456, 0.406],\
                                                [0.229, 0.224, 0.225])])

#2) ImageFolder 데이터셋 호출
trainset = torchvision.datasets.ImageFolder(root="notMNIST_small/train", transform=trans)
testsest = torchvision.datasets.ImageFolder(root="notMNIST_small/test", transform=trans)
classes = trainset.classes
print(classes)


#3) 데이터셋으로 DataLoader 생성
trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
testloader = DataLoader(testsest, batch_size=16, shuffle=True)

dataiter= iter(trainloader)
images, lables = dataiter.next()
print(lables)

## 이미지 보여주기
fig, axes = plt.subplots(figsize=(10,4), ncols=4)
for ii in range(4):
    ax = axes[ii]
    imshow(images[ii], ax=ax, normalize=True)
plt.show()
# print(images.shape)
# print("".join("%5s" %classes[lables[j]] for j in range(16)))