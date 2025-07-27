import numpy as np
import os

np.random.seed(0)
# 标准 T
T_matrix = np.array([
    [1, 1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
])
Userinput = np.array([
    [0, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
])
#设置初始化
def initialization(HiddenLayerNeuron_Number,OutputLayer_Number):
    #全局化变量
    global HiddenLayer_bias, OutputLayer_bias
    global HiddenLayer_weight, OutputLayer_weight
    global HiddenLayer_Output, Output
    global data, labels

    #初始化训练集
    data=[]
    labels=[]

    #初始化偏振量
    HiddenLayer_bias = np.random.rand(HiddenLayerNeuron_Number,1)*0.01
    OutputLayer_bias = np.random.rand(OutputLayer_Number,1)*0.01

    #初始化权重
    HiddenLayer_weight = np.random.rand(HiddenLayerNeuron_Number,36)*0.01
    OutputLayer_weight = np.random.rand(OutputLayer_Number,HiddenLayerNeuron_Number)*0.01

    #初始化输出
    HiddenLayer_Output = np.zeros((HiddenLayerNeuron_Number,1))
    Output = np.zeros((OutputLayer_Number,1))



def ReadTrainingData(Folder='Matrix'):
    
    folder_path = Folder
    #file_list = os.listdir(folder_path) #有.DS_Store得排除
    file_list = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    for file_name in file_list:
        #读取训练集
        matrix = np.loadtxt(folder_path+'/'+file_name, delimiter=',')
        #重置6*6到36*1
        matrix = matrix.reshape(-1,1)
        data.append(matrix)

        if "J" in file_name:
            labels.append(np.array([[1], [0]]))  # 2×1向量
        elif "T" in file_name:
            labels.append(np.array([[0], [1]]))
        else:
            labels.append(np.array([[0], [0]]))  # 避免出错

def Forward(Matrix):
    global HiddenLayer_Output, Output
    #隐藏层（ReLu）
    HiddenLayer_Output = np.maximum(0,np.dot(HiddenLayer_weight, Matrix) + HiddenLayer_bias)
    Output = np.dot(OutputLayer_weight, HiddenLayer_Output) + OutputLayer_bias
    #输出层（Softmax）
    exp_output = np.exp(Output)
    softmax = exp_output / np.sum(exp_output)
    return softmax

def Loss(Prediction, Real):
    epsilon = 1e-15
    Prediction_epi = np.clip(Prediction, epsilon, 1 - epsilon)  # 防止 log(0)
    loss = -np.sum(Real*np.log(Prediction_epi))
    return loss

def BackProp(Matrix, Real, Prediction):
    learning_rate = 0.01
    global HiddenLayer_bias, OutputLayer_bias
    global HiddenLayer_weight, OutputLayer_weight
    #softmax求导
    dSoftmax = Prediction - Real
    #对输出层求导
    OutputLayer_weight -= learning_rate * np.dot(dSoftmax, HiddenLayer_Output.T)
    OutputLayer_bias -= learning_rate * dSoftmax
    dHiddenLayer = np.dot(OutputLayer_weight.T, dSoftmax)
    #ReLU求导
    dHiddenLayer[HiddenLayer_Output <= 0] = 0
    # 对隐藏层参数求导
    HiddenLayer_weight -= learning_rate * np.dot(dHiddenLayer, Matrix.T)
    HiddenLayer_bias -= learning_rate * dHiddenLayer
    np.savetxt('TrainingData/HiddenLayer_weight.csv', HiddenLayer_weight, delimiter=',', fmt='%.40f')
    np.savetxt('TrainingData/HiddenLayer_bias.csv', HiddenLayer_bias, delimiter=',', fmt='%.40f')
    np.savetxt('TrainingData/OutputLayer_weight.csv', OutputLayer_weight, delimiter=',', fmt='%.40f')
    np.savetxt('TrainingData/OutputLayer_bias.csv', OutputLayer_bias, delimiter=',', fmt='%.40f')






def Update_Weights(Training_Data,Training_Labels):
    for Matrix_index in range(len(Training_Data)):
        prediction = Forward(Training_Data[Matrix_index])
        real = Training_Labels[Matrix_index]
        loss = Loss(prediction,real)
        print(loss)
        BackProp(Training_Data[Matrix_index], real, prediction)
    

initialization(6,2)
ReadTrainingData()
for epoch in range(1000):
    Update_Weights(data,labels)
    
print(Forward(Userinput.reshape(-1, 1)))