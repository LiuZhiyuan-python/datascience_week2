import argparse
from collections import defaultdict
import numpy as np

# 这个是用来对比准确率的
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from loguru import logger


class KNNClassifier():
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.xtrain = X
        self.y = y

    def predict(self, x):
        # 用idx记录与x最近的k个样本的索引
        idx = np.argsort([np.linalg.norm(x - x_sample) for x_sample in self.xtrain])[:self.k]
        knn_y = self.y[idx]
        vote = defaultdict(int)
        for y_pred in knn_y:
            vote[y_pred] += 1
        vote_list = list(vote.items())
        # 按出现次数进行排序，返回首个元素
        vote_list.sort(key=lambda x: x[1], reverse=True)
        return vote_list[0][0]


def main():
    parser = argparse.ArgumentParser(description="kNN算法Scratch实现命令行参数")
    parser.add_argument("--k",type=int, default=5, help="k")
    parser.add_argument("--enable_weight", action="store_true", help="在计算k个最紧邻的平均值时是否使用加权平均")
    args = parser.parse_args()

    X, y = load_iris(return_X_y=True)
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, shuffle=True)

    model = KNNClassifier(args.k)
    model.fit(X, y)

    n_test = xtest.shape[0]
    n_right = 0
    for i in range(n_test):
        y_pred = model.predict(xtest[i])
        if y_pred == ytest[i]:
            n_right += 1
        else:
            logger.info(f"这个样本点所得到的预测类别是{y_pred}，但真实类别是{ytest[i]}")
    logger.info(f"模型预测的准确率是{n_right*100/n_test}%")

    # 使用sklearn的模型进行对比
    skmodel = Perceptron()
    skmodel.fit(xtrain, ytrain)
    logger.info(f"sklearn模型预测的准确率是{n_right*100/n_test}%")


if __name__ == '__main__':
    main()