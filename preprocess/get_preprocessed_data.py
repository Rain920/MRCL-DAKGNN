import configparser
from data_preprocessing import *

#读参数
config_path = '../config/preprocessing.config'
config = configparser.ConfigParser()
config.read(config_path)
cfgPre = config['preprocessing']

path_RawData = cfgPre["path_RawData"]
path_Label = cfgPre["path_Label"]
path_Feature = cfgPre["path_Feature"]
path_Save = cfgPre["path_Save"]
resample = int(cfgPre["resample"])
ignore = int(cfgPre["ignore"])
wave_let = cfgPre["wave_let"]
wavelet_level = int(cfgPre["wavelet_level"])
subject_number = int(cfgPre["subject_number"])

if __name__ == "__main__":
    #通道名
    channels = ["C3_A2", "C4_A1", "F3_A2", "F4_A1", "O1_A2", "O2_A1", "LOC_A2", "ROC_A1", "X1", "X2", "X3"]

    #计算每个受试者睡眠数据的特征并保存
    for i in range(subject_number):
        print('Subject', i)
        Compute_features(path_RawData, channels, resample, wave_let, wavelet_level, path_Feature, i + 1)

    #特征标准化、特征与标签对应、数据整合
    ReadList = Perpare_data(path_Feature, ignore, path_Label, subject_number)

    #保存
    np.savez(
        path_Save,
        Fold_Num=ReadList['Fold_Num'],
        Fold_Data_a4=ReadList['Fold_Data_a4'],
        Fold_Data_d4=ReadList['Fold_Data_d4'],
        Fold_Data_d3=ReadList['Fold_Data_d3'],
        Fold_Data_d2=ReadList['Fold_Data_d2'],
        Fold_Data_d1=ReadList['Fold_Data_d1'],
        Fold_Label=ReadList['Fold_Label'],
        dtype=object
    )
    print('Save OK')
    print('Preprocess over.')