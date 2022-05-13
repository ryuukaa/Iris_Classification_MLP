# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from sklearn.cluster import KMeans
import matplotlib as mpl

#将标签类别置为数字0，1，2
label_irisName = {"Iris-setosa": "0",
                  "Iris-versicolor": "1",
                  "Iris-virginica": "2", }
#获取数据集并进行随机打乱训练
def in_put():
    dataset = []
    a = []
    iris = open('iris.data', 'r')
    for line in iris:
        line = line.strip('\n').split(',')#存入列表
        if line[-1] != '':#确认最后一个元素非空
            line[-1] = label_irisName[line[-1]]#存入3个标签类别
            dataset.append(line)
            a.append(int(line[-1]))
    random.seed(17)
    random.shuffle(dataset)
    dataset = np.array(dataset, dtype=float)#置为浮点型数组
    '''查看数据集
    print("data size: \n", dataset.size)
    print("data shape: \n", dataset.shape)
    print("label: \n", a[0:29],"\n", a[30:59], "\n", a[60:89],"\n", a[90:119],"\n", a[120:149])
    print("Type of data:", type(dataset))
    print("First five rows of data: \n", dataset[:5])
    '''
    return dataset

def CrossValidation(data = in_put()):
    np.random.seed(17)
    (sum, kfold) = data.shape#得出数据的行列
    index = np.arange(sum).astype(int)
    idx = 0
    while idx < sum:
        idx_ = idx+30
        test_idx = index[idx:idx_]#划分训练集和测试集
        train_idx = np.setdiff1d(index, test_idx)
        train_data = data[train_idx, 0:4]
        train_label = data[train_idx, 4]
        test_data = data[test_idx, 0:4]
        test_label = data[test_idx, 4]
        idx += int(sum/kfold)
        '''查看交叉验证每次划分的数据集和训练集
        print("traning set index : ", train_idx)
        print("test set index : ", test_idx)
        '''
        '''可视化/鸢尾花k-means聚类/绘制k-means结果
        estimator = KMeans(n_clusters=3)
        estimator.fit(data)
        label_pred = estimator.labels_
        x0 = data[label_pred == 0]
        x1 = data[label_pred == 1]
        x2 = data[label_pred == 2]
        plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
        plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
        plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
        plt.xlabel('petal length')
        plt.ylabel('petal width')
        plt.legend(loc=2)
        plt.show()
        '''
        yield train_data, train_label, test_data, test_label#生成器，进行迭代

class MLP_Classification():

    # 初始化网络参数
    def __init__(self, input_node=4, hidden_node=20, output_node=3, learning_rate=0.001):
        self.input_node = input_node
        self.hidden_node = hidden_node
        self.output_node = output_node
        self.W_ij = np.random.uniform(
            size=(input_node + 1, hidden_node),
            high=np.sqrt(6 / (input_node + output_node + 1)),
            low=-np.sqrt(6 / (input_node + output_node + 1)))
        self.W_jk = np.random.uniform(
            size=(hidden_node, output_node),
            high=np.sqrt(6 / (input_node + output_node)),
            low=-np.sqrt(6 / (input_node + output_node)))  # low= -np.sqrt(6/(input_node+output_node+1)))

        self.learning_rate = learning_rate

    # 前向传播
    def forward(self, X):
        #第一层输出
        a_j = np.dot(X, self.W_ij[1:, :]) + self.W_ij[0, :]
        # 隐藏层激活函数
        z_j = self.tanh(a_j)
        # 矩阵点乘
        y_k = np.dot(z_j, self.W_jk)
        return y_k, z_j, a_j

    # 损失函数计算
    def loss_function(self, y_k, y_label):
        loss = np.sum(np.square(y_k - y_label)) / 2
        return loss
    # 权重梯度计算
    def calculate_gradient(self, x, y_k, z_j, y_label):
        #输出层yk与特征对应的标签label相减
        gradient_y_k = y_k - y_label
        # 输出20*3的矩阵
        gradient_w_jk = np.outer(z_j, gradient_y_k)
        #激活函数梯度
        d_tanh = 1 - np.square(z_j)
        # (1,20)和(3,20)转置矩阵点乘得到(1,20)的矩阵
        gradient_z_j = np.dot(gradient_y_k, np.transpose(self.W_jk))
        gradient_a_j = gradient_z_j * d_tanh
        gradient_w_ij = np.outer(x, gradient_a_j)
        gradient_W_i_j_1 = gradient_a_j

        return gradient_w_jk, gradient_w_ij, gradient_W_i_j_1
    #激活函数tanh
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    # 权重更新
    def train(self, x, y_label):
        # 前向传播
        y_k, z_j, a_j = self.forward(x)
        # 损失函数计算
        loss = self.loss_function(y_k, y_label)
        # 权重计算
        gradient_w_jk, gradient_w_ij, gradient_W_i_j_1 = self.calculate_gradient(x, y_k, z_j, y_label)
        # w权重更新
        self.W_jk = self.W_jk - self.learning_rate * gradient_w_jk
        self.W_ij[1:, :] = self.W_ij[1:, :] - self.learning_rate * gradient_w_ij
        self.W_ij[0, :] = self.W_ij[0, :] - self.learning_rate * gradient_W_i_j_1

        return loss

    def predict_label(self, X):
        y_k, z_j, a_j = self.forward(X)
        plt.show()
        return y_k

    def accuracy(self, X, y_label):
        y_predict_label = self.predict_label(X)
        y_preds = np.argmax(y_predict_label, axis=1)
        y_k = np.argmax(y_label, axis=1)
        return np.mean(y_preds == y_k)

    def save(self, path='MLP_Parameters/model1/'):
        # save W_jk, W_ij
        data_path = os.path.join('%sW_jk-%s-%s-%s.npy' % (path, self.input_node, self.hidden_node, self.output_node))
        np.save(data_path, self.W_jk)
        data_path = os.path.join('%sW_ij-%s-%s-%s.npy' % (path, self.input_node, self.hidden_node, self.output_node))
        np.save(data_path, self.W_ij)

    def load(self, path='MLP_Parameters/model1/'):
        # save W_jk, W_ij
        data_path = os.path.join('%sW_jk-%s-%s-%s.npy' % (path, self.input_node, self.hidden_node, self.output_node))
        self.W_jk = np.load(data_path)
        data_path = os.path.join('%sW_ij-%s-%s-%s.npy' % (path, self.input_node, self.hidden_node, self.output_node))
        self.W_ij = np.load(data_path)


def one_hot(n_classes,y):
    return np.eye(n_classes)[y]

accuracy_train_epoch = [[],[],[],[],[]]# 将每轮的acc记录在此列表中，为后续画acc曲线提供数据
accuracy_test =[]
losses = [[],[],[],[],[]]# 将每轮的loss记录在此列表中，为后续画loss曲线提供数据
crossValidation = 0
epoches = 500#迭代次数

for train_data, train_label, test_data, \
    test_label in CrossValidation(in_put()):
    print('CrossValidation: %s' % (crossValidation))
    #训练模型
    #初始化
    model = MLP_Classification(4, 20, 3, 0.001)
    for epoch in range(epoches):
        loss = 0# 每轮分4个step，loss记录四个step生成的4个loss的和
        #训练数据的总数
        N = np.shape(train_data)[0]
        #训练部分
        for num_data in range(N):
            x = train_data[num_data, :].reshape(1, 4)
            # 将标签值转换为独热码格式，方便计算loss和accuracy
            label = one_hot(3,
                            train_label[num_data].astype(int)).reshape(1, 3)
            # 计算loss对各个参数的梯度
            #更新网络权重
            loss_ = model.train(x, label)
            loss = loss + loss_
        # 每次训练数据的精度
        x = train_data
        label = one_hot(3, train_label.astype(int))
        accuracy = model.accuracy(x, label)
        accuracy_train_epoch[crossValidation].append(accuracy)
        '''输出每次迭代的损失和精度
        print("epoch:%d   loss:%f" % (epoch, loss_ / 4))
        print("acc:%f" %(accuracy))
        print("**************************")
        '''
        # 记录每次迭代的损失函数
        losses[crossValidation].append(loss / N)
        # 存储模型
    path = os.path.join('MLP_Parameters/model%s/' % (crossValidation))
    folder = os.getcwd() + '/' + path
    if not os.path.exists(folder):
        os.makedirs(folder)
    model.save(path)

    #测试部分
    model = MLP_Classification(4, 20, 3, 0.001)
    model.load(path)

    # 测试集的精度
    x = test_data
    label = one_hot(3, test_label.astype(int))
    accuracy = model.accuracy(x, label)
    accuracy_test.append(accuracy)
    print('  test accuracy: %.2f' % (accuracy))
    # 5个部分交叉验证
    crossValidation += 1
print('5 fold cross validation average accuracy : %.2f' % (np.mean(accuracy_test)))

accuracy_train_epoch = np.array(accuracy_train_epoch, dtype=float)
# 训练集损失模型
plt.figure(num="loss")
plt.xlabel("The number of iterations")
plt.ylabel("Loss")
for i in range(len(losses)):
    plt.plot(losses[i], label="model" + str(i) + " - training set" + str(i))
    plt.legend()
    plt.title("The loss of five different models per epoch");
plt.savefig('/Users/liujia/Documents/iris_image/loss.png')

# 训练集精确度
plt.figure(num="accuracy rate train data")
plt.xlabel("The number of iterations")
plt.ylabel("Accuracy")
for i in range(len(accuracy_train_epoch)):
    plt.plot(accuracy_train_epoch[i], label="model" + str(i) + " - training set" + str(i))
    plt.legend()
    plt.title("accuracy rate train data");
plt.savefig('/Users/liujia/Documents/iris_image/data.png')