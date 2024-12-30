import numpy as np
import math

#计算上、下边界
def do_cal_min_max(q1, q3):
    down = q1 - 1.5 * (q3 - q1)
    up = q3 + 1.5 * (q3 - q1)
    return down, up

#获得中位数索引
def get_mid_idx(data):
    length = len(data)
    if length % 2 == 0:
        idx1 = length / 2 - 1
        idx2 = idx1 + 1
        idx = np.mean([idx1, idx2])
    else:
        idx = math.ceil(length / 2)
    return idx

#计算四分位数
def do_cal_quarter(data):
    # 排序
    data.sort()
    # 获取中位数
    q2 = np.median(data)
    #获取中位数索引
    idx = get_mid_idx(data)
    # 中位数将数组分为的两部分
    part1 = [v for i, v in enumerate(data) if i < idx]
    part2 = [v for i, v in enumerate(data) if i > idx]
    # 获取下四分位数
    q1 = np.median(part1)
    # 获取上四分位数
    q3 = np.median(part2)
    return q1, q2, q3

def ComputeFeatures(data):
    Max = np.max(data)  # 最大值
    Min = np.min(data)  # 最小值
    q1, q2, q3 = do_cal_quarter(data)  # 下四分位数，中位数，上四分位数
    down, up = do_cal_min_max(q1, q3)  # 下边界，上边界
    mean = np.mean(data)#平均值
    std = np.std(data)#标准差
    return Min,down,q1,q2,q3,up,Max,mean,std
