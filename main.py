from knn_class import Knn
import sys, os

# 부모 디렉토리에서 import할 수 있도록 설정
sys.path.append(os.pardir)
import numpy as np

# mnist data load할 수 있는 함수 import
from dataset.mnist import load_mnist

# python image processing library
k = int(input())
label_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# training data, test data
# flatten: 이미지를 1차원 배열로 읽음
# normalize: 0~1 실수로. 그렇지 않으면 0~255
(x_train,t_train), (x_test, t_test) = load_mnist(flatten = True, normalize= False)

#인스턴스 생성
kn = Knn(k, x_train, t_train, x_test, t_test)

#100개 test 랜덤 생성
size = 100
sample = np.random.randint(0, t_test.shape[0], size)

accuracy = 0
weighted_major_target =[]
label =[]
num = []
try_test_num = 0

for i in sample:
    distance_memo = []
    for j in range(len(x_train)):
        distance_memo.append(kn.distance(kn.x_test[i], kn.x_train[j]))
        try_test_num += 1

        #몇번째 데이터를 계산하는지 숫자로 출력.(프로그램이 잘 돌고 있는지 확인용)
        print(try_test_num)

    # 최소거리를 오름차순으로 정렬
    # 첫번째 인덱스는 테스트케이스 자신이여서 이후에 계산시에는 인덱스 1부터 시작
    distance_memo_index = np.argsort(distance_memo)

    # k개의 이웃한 neighbor 구해서 정렬하는 K-Nearest Neighbor 함수
    list_target = kn.neighbor(distance_memo_index)

    # 최소거리부터 거리에 따르는 가중치를 더한 weighted major vote함수
    weighted_major_target.append(kn.weighted_majority(list_target))
    label.append(kn.t_test[i])
    num.append(i)

#정확도, ouput 출력
for i in range(100):
    if weighted_major_target[i] == label[i]:
        accuracy += 1
    print(f"{num[i]}th data result : {weighted_major_target[i]}, label: {label[i]}")
print(f"accuracy = {accuracy/100}")

