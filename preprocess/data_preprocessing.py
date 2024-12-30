import os
from os import path
from scipy import signal
from scipy import io as scio
import pywt
from Compute_Features import *

#读一个受试者的睡眠信号数据
def Read_raw_data(path_RawData, channels, num, resample):
    psg = scio.loadmat(path.join(path_RawData, 'subject%d.mat' % (num)))
    psg_use = []
    for c in channels:
        psg_use.append(
            np.expand_dims(signal.resample(psg[c], resample, axis=-1), 1))
    psg_use = np.concatenate(psg_use, axis=1)
    return psg_use

#离散小波变换
def Wavelet(data,wave_let,wavelet_level):
    #对一个一维信号进行小波分解
    data = np.array(data)
    w = wave_let
    n = wavelet_level
    a4, d4, d3, d2, d1 = pywt.wavedec(data, w, level=n)
    return a4, d4, d3, d2, d1

# 计算一个受试者睡眠信号的特征并保存
def Compute_features(path_RawData,channels,resample,wave_let,wavelet_level,path_Feature,num):
    data = Read_raw_data(path_RawData,channels,num,resample)
    num_time_slice = data.shape[0]#30s时间片数量
    num_channels = data.shape[1]#采样通道数量
    fea_a4 = np.zeros([num_time_slice, num_channels, 9])
    fea_d4 = np.zeros([num_time_slice, num_channels, 9])
    fea_d3 = np.zeros([num_time_slice, num_channels, 9])
    fea_d2 = np.zeros([num_time_slice, num_channels, 9])
    fea_d1 = np.zeros([num_time_slice, num_channels, 9])

    for i in range(0,num_time_slice):
        for j in range(0,num_channels):
            use_data = data[i][j]
            a4,d4,d3,d2,d1 = Wavelet(use_data,wave_let,wavelet_level)
            fea_a4[i][j] = ComputeFeatures(a4)
            fea_d4[i][j] = ComputeFeatures(d4)
            fea_d3[i][j] = ComputeFeatures(d3)
            fea_d2[i][j] = ComputeFeatures(d2)
            fea_d1[i][j] = ComputeFeatures(d1)
    if not os.path.isdir(path_Feature):
        os.makedirs(path_Feature)
    scio.savemat(path_Feature + str(num) + '_features.mat', {'a4': fea_a4, 'd4': fea_d4,'d3': fea_d3,'d2': fea_d2,'d1': fea_d1})
    print('Successfully calculated features!')
    return

#读一个受试者的标签数据
def Read_label(path_Label, sub_id, ignore):
    label = []
    with open(path.join(path_Label, '%d/%d_2.txt' % (sub_id, sub_id))) as f:
        s = f.readline()
        while True:
            a = s.replace('\n', '')
            label.append(int(a))
            s = f.readline()
            if s == '' or s == '\n':
                break
    llable = np.array(label[:-ignore])
    llable = np.where(llable != 5, llable, 4)
    llable = llable.reshape((-1,1))
    return llable

#读一个受试者的特征
def Read_features(path_Feature, num):
    read_mat = scio.loadmat(path.join(path_Feature, '%d_features.mat' % (num)))
    read_features_a4 = read_mat['a4']
    read_features_d4 = read_mat['d4']
    read_features_d3 = read_mat['d3']
    read_features_d2 = read_mat['d2']
    read_features_d1 = read_mat['d1']
    return read_features_a4,read_features_d4,read_features_d3,read_features_d2,read_features_d1

def Perpare_data(path_Feature, ignore, path_Label, subject_number):
    Out_Data_a4 = []
    Out_Data_d4 = []
    Out_Data_d3 = []
    Out_Data_d2 = []
    Out_Data_d1 = []
    Out_Label = []

    Fold_Num = np.zeros([subject_number], dtype=int)
    for i in range(subject_number):
        nnum = i + 1
        FoldData_a4,FoldData_d4,FoldData_d3,FoldData_d2,FoldData_d1 = Read_features(path_Feature,nnum)
        FoldLabel = Read_label(path_Label,nnum,ignore)
        Fold_Num[i] = FoldLabel.shape[0]

        Out_Data_a4.append(FoldData_a4)
        Out_Data_d4.append(FoldData_d4)
        Out_Data_d3.append(FoldData_d3)
        Out_Data_d2.append(FoldData_d2)
        Out_Data_d1.append(FoldData_d1)
        Out_Label.append(FoldLabel)

        if i == 0:
            All_Data_a4 = FoldData_a4
            All_Data_d4 = FoldData_d4
            All_Data_d3 = FoldData_d3
            All_Data_d2 = FoldData_d2
            All_Data_d1 = FoldData_d1
            All_Label = FoldLabel
        else:
            All_Data_a4 = np.row_stack((All_Data_a4, FoldData_a4))
            All_Data_d4 = np.row_stack((All_Data_d4, FoldData_d4))
            All_Data_d3 = np.row_stack((All_Data_d3, FoldData_d3))
            All_Data_d2 = np.row_stack((All_Data_d2, FoldData_d2))
            All_Data_d1 = np.row_stack((All_Data_d1, FoldData_d1))
            All_Label = np.row_stack((All_Label, FoldLabel))
        i += 1

    # 数据标准化
    mean_a4 = All_Data_a4.mean(axis=0)
    std_a4 = All_Data_a4.std(axis=0)
    All_Data_a4 -= mean_a4
    All_Data_a4 /= std_a4
    for i in range(subject_number):
        Out_Data_a4[i] -= mean_a4
        Out_Data_a4[i] /= std_a4

    mean_d4 = All_Data_d4.mean(axis=0)
    std_d4 = All_Data_d4.std(axis=0)
    All_Data_d4 -= mean_d4
    All_Data_d4 /= std_d4
    for i in range(subject_number):
        Out_Data_d4[i] -= mean_d4
        Out_Data_d4[i] /= std_d4

    mean_d3 = All_Data_d3.mean(axis=0)
    std_d3 = All_Data_d3.std(axis=0)
    All_Data_d3 -= mean_d3
    All_Data_d3 /= std_d3
    for i in range(subject_number):
        Out_Data_d3[i] -= mean_d3
        Out_Data_d3[i] /= std_d3

    mean_d2 = All_Data_d2.mean(axis=0)
    std_d2 = All_Data_d2.std(axis=0)
    All_Data_d2 -= mean_d2
    All_Data_d2 /= std_d2
    for i in range(subject_number):
        Out_Data_d2[i] -= mean_d2
        Out_Data_d2[i] /= std_d2

    mean_d1 = All_Data_d1.mean(axis=0)
    std_d1 = All_Data_d1.std(axis=0)
    All_Data_d1 -= mean_d1
    All_Data_d1 /= std_d1
    for i in range(subject_number):
        Out_Data_d1[i] -= mean_d1
        Out_Data_d1[i] /= std_d1

    return {'Fold_Num': Fold_Num,
            'Fold_Data_a4': Out_Data_a4,
            'Fold_Data_d4': Out_Data_d4,
            'Fold_Data_d3': Out_Data_d3,
            'Fold_Data_d2': Out_Data_d2,
            'Fold_Data_d1': Out_Data_d1,
            'Fold_Label': Out_Label}