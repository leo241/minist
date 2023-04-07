import numpy as np
from numpy.random import randn
import gzip
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

warnings.filterwarnings('ignore')

class DataProcess:
    def __init__(self):
        dic = dict()
        for i in range(10):
            a = np.zeros(10)
            a[i] = 1
            dic[i] = a
        self.dic = dic
    def load_data(self,xtrain_dir='train-images-idx3-ubyte.gz', ytrain_dir='train-labels-idx1-ubyte.gz', xtest_dir='t10k-images-idx3-ubyte.gz', ytest_dir='t10k-labels-idx1-ubyte.gz'):
        dic = self.dic
        with gzip.open(xtrain_dir) as all_img:
            xtrain = all_img.read()
        with gzip.open(ytrain_dir) as all_img:
            ytrain = all_img.read()
        with gzip.open(xtest_dir) as all_img:
            xtest = all_img.read()
        with gzip.open(ytest_dir) as all_img:
            ytest = all_img.read()
        xtrainl, ytrainl, xtestl, ytestl = [],[],[],[]
        for i in tqdm(range(60000)):
            xtrainb = xtrain[16 + 784 * i:16 + 784 * (i + 1)]
            ytrainl.append(dic[ytrain[8 + i]])
            xtrainl.append([xtrainb[j] for j in range(784)])
        for i in tqdm(range(10000)):
            xtestb = xtest[16 + 784 * i:16 + 784 * (i + 1)]
            ytestl.append(dic[ytest[8 + i]])
            xtestl.append([xtestb[j] for j in range(784)])
        return np.asarray(xtrainl),np.asarray(ytrainl),np.asarray(xtestl),np.asarray(ytestl)

    def view_x(self, x):
        img = x.reshape(28,28)
        plt.imshow(img)
        plt.show()



def predict(x,w1,w2):
    h = 1.0 / (1.0 + np.exp(-x@w1))
    y_pred = h@w2
    return y_pred

def get_loss(y, y_pred):
    loss = np.square(y_pred - y).sum()
    return loss/len(y)
if __name__ == '__main__': # lambda 1e-3 epoch 4540
    dp = DataProcess()
    xtrain_dir, ytrain_dir, xtest_dir, ytest_dir = 'train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz'
    xtrain, ytrain, xtest, ytest = dp.load_data(xtrain_dir, ytrain_dir, xtest_dir, ytest_dir)
    xtest,ytest, xvalid,yvalid = xtest[0:5000],ytest[0:5000],xtest[5000:10000], ytest[5000:10000]
    # print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape,xvalid.shape, yvalid.shape)
    N, Din, H, Dout = 60000, 784, 100, 10
    lambda3 = 1e-3
    N_valid = 5000
    w1, w2 = randn(Din, H), randn(H, Dout)  # 使用随机初始化参数
    # weights= np.load('weights.npz')
    # w1, w2 = weights['arr_0'], weights['arr_1'] # 使用预训练参数继续训练
    lambda1 = lambda2 = lambda3  # 正则化参数
    batch_size = 30
    # Epoch = 100000
    np.random.seed(2023)
    lr0 = lr1 = 1e-4
    lr2 = lr1 * 10 # 最佳 1e-4 -> 1e-3
    beta = 0.9999999  # 学习率下降策略衰减因子
    epoch_list = []
    val_loss_list = []
    mini_val_loss = float('inf')
    # for t in range(Epoch):
    t = 0
    t_star = 0
    w1_star = None
    w2_star = None
    while 1:
        id = np.random.randint(N - batch_size)
        x, y = xtrain[id:id + batch_size], ytrain[id: id + batch_size]
        h = 1.0 / (1.0 + np.exp(-x @ w1))
        y_pred = h @ w2
        loss = np.square(y_pred - y).sum()
        if (t + 1) % 20000 == 0:
            #         id2 = np.random.randint(N_valid - batch_size)
            #         xval, yval = xvalid[id2:id2 + batch_size], yvalid[id2: id2+ batch_size]
            yval_pre = predict(xvalid, w1, w2)
            yval_loss = get_loss(yvalid, yval_pre)
            ytrain_pre = predict(xtrain, w1, w2)
            ytrain_loss = get_loss(ytrain, ytrain_pre)
            if yval_loss < mini_val_loss:
                mini_val_loss = yval_loss
                t_star = t
                w1_star = w1
                w2_star = w2
            print(f'Epoch {int((t + 1) / 20000)} 0 train loss:', round(ytrain_loss, 2), 'valid loss:',
                  round(yval_loss, 2))
            if t - t_star > t_star:
                break
        dy_pred = 2.0 * (y_pred - y)
        dw2 = h.T.dot(dy_pred) + lambda2 * w2  # L2正则化
        dh = dy_pred.dot(w2.T)
        dw1 = x.T.dot(dh * h * (1 - h)) + lambda1 * w1  # L2正则化
        w1 -= lr1 * dw1
        w2 -= lr1 * dw2
        if lr1 < lr2:
            lr1 = lr1 ** beta
        t += 1
    print('验证集最小损失：', mini_val_loss, '对应epoch:', t_star / 2000)
    with open('para.txt', 'a') as file:
        file.write(f'{H}\t{lambda1}\t{lr0}\t{t_star + 1}\t{mini_val_loss}\n')
        file.close()