import numpy as np
import collections

# knn 클래스 생성
class Knn:
    def __init__(self, k, x_train, t_train, x_test, t_test):
        self.k = k
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test

    # 테스트케이스와의 거리 계산하는 distance 함수
    def distance(self, a, b):
        d = 0
        for m in range(784):
            tmp1 = float(a[m])
            tmp2 = float(b[m])
            d += np.power(tmp1 - tmp2, 2)
        d = np.sqrt(d)
        return d

    # k개의 이웃한 neighbor 구해서 정렬하는 K-Nearest Neighbor 함수
    def neighbor(self, distance_memo_index):
        sort_target = []
        for m in range(self.k):
            sort_target.append(self.t_train[distance_memo_index[m + 1]])
        return sort_target

    # 가까운 데이터중에서 갯수가 많은 데이터 결정 함수
    def majority(self, list_target):
        cnt = collections.Counter(list_target)
        target_major = cnt.most_common(1)[0][0]
        return target_major

    # 최소거리부터 거리에 따르는 가중치를 더한 weighted major vote함수
    def weighted_majority(self, list_target):
        weighted = [0] * 10
        for a in range(self.k):
            temp = int(list_target[a])
            weighted[temp] += self.k - a
        tmp = max(weighted)
        return weighted.index(tmp)
