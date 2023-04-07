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

def accuracy(ytrue, ypre):
    ytrue = np.argmax(ytrue,1)
    ypre = np.argmax(ypre,1)
    compare = (ytrue == ypre)
    return round(sum(compare)/len(compare),3)

if __name__ == '__main__': # lambda 1e-3 epoch 4540
    dp = DataProcess()
    xtrain_dir, ytrain_dir, xtest_dir, ytest_dir = 'train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz'
    xtrain, ytrain, xtest, ytest = dp.load_data(xtrain_dir, ytrain_dir, xtest_dir, ytest_dir)
    weights = np.load('weights.npz') # 读取模型
    b1, b2 = weights['arr_0'], weights['arr_1']
    ypre = predict(xtest, b1, b2)
    print('准确率为：',accuracy(ytest, ypre))
