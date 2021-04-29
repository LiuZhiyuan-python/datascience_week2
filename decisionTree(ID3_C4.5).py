import argparse

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from loguru import logger


class TreeNode(object):
    def __init__(self, feature_idx=None, feature_val=None, feature_name=None, node_val=None, child=None):
        """
        :param feature_idx: 特征索引
        :param feature_val: 特征取值
        :param feature_name: 特征名
        :param node_val: 类别标签（叶节点）
        :param child: 子节点（中间节点）
        """
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.feature_name = feature_name
        self.node_val = node_val
        self.child = child


class DecisionTree(object):
    def __init__(self, feature_name, etype, epsilon=0.01):
        self.root = None
        self.feature_name = feature_name
        self.etype = etype
        self.epsilon = epsilon

    def fit(self, X, y):
        """模型训练构建决策树"""
        self.root = self.build_tree(X, y)

    def predict(self, x):
        """预测样本类别"""
        return self.predict_help(x, self.root)

    def predict_help(self, x, tree=None):
        if tree is None:
            tree = self.root
        if tree.node_val is not None:
            return tree.node_val

        # 遍历特征所有的取值，
        fea_idx = tree.feature_idx
        for fea_val, child_node in tree.child.items():
            # if fea_val == 3.0 or 3.2:
            #     # print(x[fea_idx], fea_val, child_node, child_node.node_val)
            if x[fea_idx] == fea_val:
                if child_node.node_val is not None:
                    return child_node.node_val
                else:
                    """
                    错误原因在此处：
                    每次出错的数据都是[5.9 3.2 1.8 4.8] 和 [5.9 3.0 5.1 1.8]。
                    决策树中这两个数据对应的节点能够查到node_val值，但是预测不出来，说明是predict_help函数出了问题。
                    检查之后发现，predict_help函数中，每次递归用到的x都时一样的，但是在建立树时，每次会删掉一列。
                    因此，两者用到的索引意义是不同的，这里存在一个错误。
                    具体到这两个数据集，函数执行到fea_idx = tree.feature_idx时:
                    开始这个索引是3，x[3]==1.8。
                    接下来是0，x[0]==5.9,
                    但接下来，索引依然是0（创造子树时把x显著特征的列删掉了）,对应x[0]==5.9不是3.0或者3.2，因此这里没有返回值。
                    """
                    try:
                        x = np.delete(x, fea_idx, 1)
                    except:
                        x = np.delete(x, fea_idx, 0)

                    return self.predict_help(x, child_node)

    def build_tree(self, X, y):
        """构建决策树"""

        # 1. 剩余实例属于同一类，置T为单节点树，将他们的标签作为类别标签
        if np.unique(y).shape[0] == 1:
            return TreeNode(node_val=y[0])

        # 2. 不能继续划分，置T为单节点树，将最多的类别标签该节点的类别标签
        elif X.shape[1] == 1:
            node_val = self.vote_label(y)
            return TreeNode(node_val=node_val)

        # 3. 否则，计算各特征的信息增益，记录最大的信息增益和对应的特征
        else:
                n_feature = X.shape[1]
                # 将最大增益（比）初始化为负无穷
                max_gain = -np.inf
                max_fea_idx = 0
                for i in range(n_feature):
                    # 传入第i维特征的值，计算信息增益（比）
                    if self.etype == "gain":
                        gain = self.calc_gain(X[:, i], y)
                    else:
                        gain = self.calc_gain_ratio(X[:, i], y)
                    # 记录最大增益值和对应的维度i
                    if gain > max_gain:
                        max_gain = gain
                        max_fea_idx = i

                # 4.信息增益(比)小于epsilon，置T为单节点树
                if max_gain <= self.epsilon:
                    node_val = self.vote_label(y)
                    return TreeNode(node_val=node_val)

                # 5.否则，对特征的每一个取值构建子结点，返回结点和子结点形成的树T
                else:
                    feature_name = self.feature_name[max_fea_idx]
                    child_tree = dict()
                    # 构建一维子树，删除一个特征
                    # 作递归，直到所有的树枝都达到叶结点
                    for fea_val in np.unique(X[:, max_fea_idx]):
                        child_X = X[X[:, max_fea_idx] == fea_val]
                        child_y = y[X[:, max_fea_idx] == fea_val]
                        child_X = np.delete(child_X, max_fea_idx, 1)
                        child_tree[fea_val] = self.build_tree(child_X, child_y)
                    return TreeNode(max_fea_idx, feature_name=feature_name, child=child_tree)

    def vote_label(self, y):
        """统计剩余的样本中次数出现最多的类别作为叶节点的类别"""
        label, num_label = np.unique(y, return_counts=True)
        return label[np.argmax(num_label)]

    def calc_entropy(self, y):
        """计算熵"""
        entropy = 0
        # unique:记录y的取值以及每个取值出现的次数
        _, num_ck = np.unique(y, return_counts=True)
        for n in num_ck:
            p = n / y.shape[0]
            entropy -= p * np.log2(p)
        return entropy

    def calc_conditional_entropy(self, x, y):
        """计算条件熵"""
        cond_entropy = 0
        xval, num_x = np.unique(x, return_counts=True)
        for v, n in zip(xval, num_x):
            y_sub = y[x == v]
            sub_entropy = self.calc_entropy(y_sub)
            p = n / y.shape[0]
            cond_entropy += p * sub_entropy
        return cond_entropy

    def calc_gain(self, x, y):
        return self.calc_entropy(y) - self.calc_conditional_entropy(x, y)

    def calc_gain_ratio(self, x, y):
        return self.calc_gain(x, y) / self.calc_entropy(x)


def main():
    parser = argparse.ArgumentParser(description="决策树算法Scratch实现命令行参数")
    parser.add_argument("--epsilon", type=float, default=0.01, help="当信息增益或信息增益比小于阈值的时候不再进行划分，而是直接把剩余样本设置为叶节点")
    parser.add_argument("--etype", type=str, default="gain", help="可选值类型：gain/ratio")
    args = parser.parse_args()

    dataset = load_iris()
    feature_name = dataset.feature_names
    xtrain, _, ytrain, _ = train_test_split(dataset.data, dataset.target, train_size=0.8, shuffle=True)

    model = DecisionTree(feature_name, args.etype, args.epsilon)
    model.fit(xtrain, ytrain)

    n_test = xtrain.shape[0]
    n_right = 0
    for i in range(n_test):
        y_pred = model.predict(xtrain[i])
        if y_pred == ytrain[i]:
            n_right += 1
        else:
            print(xtrain[i], ytrain[i])
            logger.info(f"这个样本点所得到的预测类别是{y_pred}，但真实类别是{ytrain[i]}")
    logger.info(f"模型预测的准确率是{n_right * 100 / n_test}%")
    print('complete')
    """
    bug描述：有一定几率会出现概率不为100%的情况
    但是在这种情况下显示的y预测值显示的是none
    Debug时查看model--root--child--...找到相应的节点，却发现node_val有值
    于是我怀疑是predict函数的问题，但是没能查出错误
    """


if __name__ == '__main__':
    main()
