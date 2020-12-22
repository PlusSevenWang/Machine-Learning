import pandas as pd
import numpy as np

# 数据处理
def dataProcess(df):
    x_list, y_list = [], []
    df = df.replace(['NR'], [0.0])  # 将空数据NR替换为0
    array = np.array(df).astype(float)  # 转换数据类型
    for i in range(0, 4320, 18):    # 将数据集拆分为多个数据帧
        for j in range(24-9):
            mat = array[i:i+18, j:j+9]
            label = array[i+9, j+9]
            x_list.append(mat)
            y_list.append(label)
    x = np.array(x_list)
    y = np.array(y_list)
    return x, y, array

# 训练模型
def train(x_train, y_train, epoch):
    bias = 0    # 初始化偏置值
    weights = np.ones(9)    # 初始化权重
    learning_rate = 1   # 初始化学习率
    reg_rate = 0.001    # 正则项系数
    bg2_sum = 0     # 梯度平方和(存放偏置值)
    wg2_sum = np.zeros(9)   # 梯度平方和(存放权重)

    for i in range(epoch):
        b_g = 0
        w_g = np.zeros(9)
        for j in range(3200):   # 在所有数据上计算Loss_label的梯度
            b_g += (y_train[j] - weights.dot(x_train[j, 9, :]) - bias) * (-1)
            for k in range(9):
                w_g[k] += (y_train[j] - weights.dot(x_train[j, 9, :]) - bias) * (-x_train[j, 9, k])
        b_g /= 3200     # 求均值
        w_g /= 3200
        for m in range(9):  # 与Loss_regularization在w上的梯度相加
            w_g[m] += reg_rate * weights[m]
        bg2_sum += b_g**2   # adagrad算法
        wg2_sum += w_g**2
        bias -= learning_rate/bg2_sum**0.5 * b_g    # 更新权重和偏置
        weights -= learning_rate/wg2_sum**0.5 * w_g
        if i%200 == 0:  # 输出训练集上的损失 200轮/次
            loss = 0
            for j in range(3200):
                loss += (y_train[j] - weights.dot(x_train[j, 9, :]) - bias)**2
            print('{} 轮后, 训练集上的损失:'.format(i), loss/3200)
    return weights, bias

# 验证效果
def validate(x_val, y_val, weights, bias):
    loss = 0
    for i in range(400):
        loss += (y_val[i] - weights.dot(x_val[i, 9, :]) - bias)**2
    return loss / 400

def main():
    df = pd.read_csv('train.csv', usecols=range(2,26))  # 从csv中读取信息
    x, y, _ = dataProcess(df)
    x_train, y_train = x[0:3200], y[0:3200]     # 划分训练集与验证集
    x_val, y_val = x[3200:3600], y[3200:3600]
    epoch = 2000    # 训练轮数
    w, b = train(x_train, y_train, epoch)   # 开始训练
    loss = validate(x_val, y_val, w, b)     # 在验证集上的损失
    print('验证集上的损失:', loss)

if __name__ == '__main__':
    main()