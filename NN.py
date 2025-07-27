import numpy as np
import os

np.random.seed(0)

#随机生成一个矩阵
UserInport = np.array([
    [1, 1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
])

#定义神经元和输出层数量
HiddenLayerNeuron_Number = 6
OutputLayer_Number = 2

#设置初始化
def initialization(HiddenLayerNeuron_Number,OutputLayer_Number):
    global Matrix_stretched
    global HiddenLayer_bias, OutputLayer_bias
    global HiddenLayer_weight, OutputLayer_weight
    global HiddenLayer_Output, Output
    

    #将矩阵拉伸到36*1
    Matrix_stretched = UserInport.reshape(36,1)

    #初始化偏振量
    HiddenLayer_bias = np.random.rand(HiddenLayerNeuron_Number,1)*0.01
    OutputLayer_bias = np.random.rand(OutputLayer_Number,1)*0.01

    #初始化权重
    HiddenLayer_weight = np.random.rand(HiddenLayerNeuron_Number,36)*0.01
    OutputLayer_weight = np.random.rand(OutputLayer_Number,HiddenLayerNeuron_Number)*0.01

    folder_path='TrainingData'


    HiddenLayer_weight = np.loadtxt(os.path.join(folder_path, 'HiddenLayer_weight.csv'), delimiter=',')
    HiddenLayer_bias = np.loadtxt(os.path.join(folder_path, 'HiddenLayer_bias.csv'), delimiter=',').reshape(-1, 1)
    OutputLayer_weight = np.loadtxt(os.path.join(folder_path, 'OutputLayer_weight.csv'), delimiter=',')
    OutputLayer_bias = np.loadtxt(os.path.join(folder_path, 'OutputLayer_bias.csv'), delimiter=',').reshape(-1, 1)

    #初始化输出
    HiddenLayer_Output = np.zeros((HiddenLayerNeuron_Number,1))
    Output = np.zeros((OutputLayer_Number,1))

#设置隐藏层
def HiddenLayerNeuron():
    HiddenLayer_Output = np.maximum(0,np.dot(HiddenLayer_weight, Matrix_stretched) + HiddenLayer_bias)
    return HiddenLayer_Output
#设置输出层
def OutputLayer():
    Output = np.dot(OutputLayer_weight, HiddenLayer_Output) + OutputLayer_bias
    exp_output = np.exp(Output)
    softmax = exp_output / np.sum(exp_output)
    return softmax



initialization(HiddenLayerNeuron_Number,OutputLayer_Number)
HiddenLayer_Output = HiddenLayerNeuron()
softmax = OutputLayer()
print(softmax)
