import os
import numpy as np
import torch
from model.Utils import *

class kFoldGenerator():
    '''
    Data Generator
    '''
    k = -1      # the fold number
    x_list = [] # x list with length=k
    y_list = [] # x list with length=k

    # Initializate
    def __init__(self, k, context, d1 , d2, d3, d4, a4, y):
        if len(a4)!=k or len(y)!=k:
            assert False,'Data generator: Length of x or y is not equal to k.'
        self.k=k
        self.d1_list = d1
        self.d2_list = d2
        self.d3_list = d3
        self.d4_list = d4
        self.a4_list = a4
        self.y_list=y
        self.context = context

    # Get i-th fold
    def getFold(self, i):
        isFirst=True
        train_len = []
        for p in range(self.k):
            if p!=i:
                if isFirst:
                    train_data_d1 = self.d1_list[p]
                    train_data_d1 = AddContext(train_data_d1,self.context)
                    train_data_d2 = self.d2_list[p]
                    train_data_d2 = AddContext(train_data_d2, self.context)
                    train_data_d3 = self.d3_list[p]
                    train_data_d3 = AddContext(train_data_d3, self.context)
                    train_data_d4 = self.d4_list[p]
                    train_data_d4 = AddContext(train_data_d4, self.context)
                    train_data_a4 = self.a4_list[p]
                    train_data_a4 = AddContext(train_data_a4, self.context)
                    train_targets = self.y_list[p]
                    train_targets = AddContext(train_targets, self.context, label=True)
                    len = np.array(train_data_d1).shape[0]
                    train_len.append(len)
                    isFirst = False
                else:
                    d1 = self.d1_list[p]
                    d1 = AddContext(d1, self.context)
                    train_data_d1 = np.concatenate((train_data_d1, d1))
                    d2 = self.d2_list[p]
                    d2 = AddContext(d2, self.context)
                    train_data_d2 = np.concatenate((train_data_d2, d2))
                    d3 = self.d3_list[p]
                    d3 = AddContext(d3, self.context)
                    train_data_d3 = np.concatenate((train_data_d3, d3))
                    d4 = self.d4_list[p]
                    d4 = AddContext(d4, self.context)
                    train_data_d4 = np.concatenate((train_data_d4, d4))
                    a4 = self.a4_list[p]
                    a4 = AddContext(a4, self.context)
                    train_data_a4 = np.concatenate((train_data_a4, a4))
                    lab = self.y_list[p]
                    lab = AddContext(lab, self.context, label=True)
                    train_targets   = np.concatenate((train_targets, lab))
                    len = np.array(d1).shape[0]
                    train_len.append(len)
            else:
                val_data_d1 = self.d1_list[p]
                val_data_d1 = AddContext(val_data_d1, self.context)
                val_data_d2 = self.d2_list[p]
                val_data_d2 = AddContext(val_data_d2, self.context)
                val_data_d3 = self.d3_list[p]
                val_data_d3 = AddContext(val_data_d3, self.context)
                val_data_d4 = self.d4_list[p]
                val_data_d4 = AddContext(val_data_d4, self.context)
                val_data_a4 = self.a4_list[p]
                val_data_a4 = AddContext(val_data_a4, self.context)
                val_targets = self.y_list[p]
                val_targets = AddContext(val_targets, self.context, label=True)

        num_val = np.array(val_data_d1).shape[0]

        train_data_d1 = torch.from_numpy(train_data_d1.astype(np.float32))
        train_data_d1 = torch.FloatTensor(train_data_d1)
        train_data_d2 = torch.from_numpy(train_data_d2.astype(np.float32))
        train_data_d2 = torch.FloatTensor(train_data_d2)
        train_data_d3 = torch.from_numpy(train_data_d3.astype(np.float32))
        train_data_d3 = torch.FloatTensor(train_data_d3)
        train_data_d4 = torch.from_numpy(train_data_d4.astype(np.float32))
        train_data_d4 = torch.FloatTensor(train_data_d4)
        train_data_a4 = torch.from_numpy(train_data_a4.astype(np.float32))
        train_data_a4 = torch.FloatTensor(train_data_a4)
        train_targets = torch.LongTensor(train_targets)
        train_targets = train_targets.squeeze()
        train_len = np.array(train_len)

        val_data_d1 = torch.from_numpy(val_data_d1.astype(np.float32))
        val_data_d1 = torch.FloatTensor(val_data_d1)
        val_data_d2 = torch.from_numpy(val_data_d2.astype(np.float32))
        val_data_d2 = torch.FloatTensor(val_data_d2)
        val_data_d3 = torch.from_numpy(val_data_d3.astype(np.float32))
        val_data_d3 = torch.FloatTensor(val_data_d3)
        val_data_d4 = torch.from_numpy(val_data_d4.astype(np.float32))
        val_data_d4 = torch.FloatTensor(val_data_d4)
        val_data_a4 = torch.from_numpy(val_data_a4.astype(np.float32))
        val_data_a4 = torch.FloatTensor(val_data_a4)
        val_targets = torch.LongTensor(val_targets)
        val_targets = val_targets.squeeze()


        return train_data_d1,train_data_d2,train_data_d3,train_data_d4,train_data_a4,train_targets,val_data_d1,val_data_d2,val_data_d3,val_data_d4,val_data_a4,val_targets,train_len,num_val

    # Get all data x
    def getX(self):
        All_X = self.x_list[0]
        for i in range(1,self.k):
            All_X = np.append(All_X,self.x_list[i], axis=0)
        return All_X

    # Get all label y
    def getY(self):
        All_Y = self.y_list[0]
        for i in range(1,self.k):
            All_Y = np.append(All_Y,self.y_list[i], axis=0)
        return All_Y

    # Get all label y
    def getY_one_hot(self):
        All_Y = self.getY()
        return np.argmax(All_Y, axis=1)