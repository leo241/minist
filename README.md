# minist
使用numpy构建两层神经网络分类器（课程作业）  
train and test.ipynb:完整的训练-测试-保存-读取-可视化代码 （不含验证集）  
train.py:训练部分代码，直接运行即可（划分了验证集）  
test.py:读取模型，输出预测精度。  

以上代码文件需在同目录下放置以下文件  
  
minist源数据集（包含以下四个文件）  

t10k-images-idx3-ubyte.gz   
t10k-labels-idx1-ubyte.gz  
train-images-idx3-ubyte.gz  
train-labels-idx1-ubyte.gz  

模型：weights.npz(在训练过程中也会自动生成)  
交叉验证参数日志：para.txt(用于在交叉验证中记录最优参数，如果不进行交叉验证可以忽略)  

