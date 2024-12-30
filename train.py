import os
from os import path
import torch.optim as optim
import configparser
from model.DG import *
from model.MR_DAKGNN import *
from model.Utils import *
import scipy.io

#读参数
config_path = './config/MR_DAKGNN.config'
config = configparser.ConfigParser()
config.read(config_path)
cfgTrain = config['train']
cfgModel = config['model']
cfgSave = config['save']
#train
context = int(cfgTrain["context"])
epoch_num = int(cfgTrain["epoch_num"])
fold_num = int(cfgTrain["fold_num"])
lr = float(cfgTrain["lr"])
cuda = cfgTrain["cuda"]
#model
bandwidth = float(cfgModel["bandwidth"])
K = int(cfgModel["K"])
in_dim = int(cfgModel["in_dim"])
out_dim = int(cfgModel["out_dim"])
num_of_nodes = int(cfgModel["num_of_nodes"])
higru_hid = int(cfgModel["higru_hid"])
higru_out = int(cfgModel["higru_out"])
temperature = float(cfgModel["temperature"])
scale = float(cfgModel["scale"])
#save_path
data_path = cfgSave["data_path"]
model_save_path = cfgSave["model_save_path"]
result_save_path = cfgSave["result_save_path"]

#读入数据
ReadList = np.load(data_path ,allow_pickle=True)
Fold_Num = ReadList['Fold_Num']
Fold_Data_a4 = ReadList['Fold_Data_a4']
Fold_Data_d4 = ReadList['Fold_Data_d4']
Fold_Data_d3 = ReadList['Fold_Data_d3']
Fold_Data_d2 = ReadList['Fold_Data_d2']
Fold_Data_d1 = ReadList['Fold_Data_d1']
Fold_Label   = ReadList['Fold_Label']
print("Read data successfully")

#十折交叉验证
DataGenerator = kFoldGenerator(fold_num,context,Fold_Data_d1,Fold_Data_d2,Fold_Data_d3,Fold_Data_d4,Fold_Data_a4,Fold_Label)

for k in range(fold_num):

    print('Fold #', k)
    train_data_d1, train_data_d2, train_data_d3, train_data_d4, train_data_a4, train_targets, val_data_d1, val_data_d2, val_data_d3, val_data_d4, val_data_a4, val_targets, num_train, num_val = DataGenerator.getFold(k)
    DEVICE = torch.device(cuda)
    net = make_model(bandwidth=bandwidth,
                     K=K, in_dim=in_dim,
                     out_dim=out_dim,
                     num_of_nodes=num_of_nodes,
                     higru_hid=higru_hid,
                     higru_out=higru_out,
                     temperature=temperature,
                     scale=scale,
                     DEVICE = DEVICE)
    net.to(DEVICE)
    optimizer = optim.Adam(net.parameters(),lr=lr)
    #model_loss = nn.CrossEntropyLoss()

    best_epoch = 0
    best_acc = 0

    for epoch in range(epoch_num):
        print('Epoch #', epoch)
        net.train()
        loss_ = 0
        acc_ = 0

        start_num = 0
        end_num = 0
        for i in range(9):
            end_num = end_num + num_train[i]
            num_time_slice = num_train[i]
            train_input_d1 = train_data_d1[start_num:end_num, :, :]
            train_input_d2 = train_data_d2[start_num:end_num, :, :]
            train_input_d3 = train_data_d3[start_num:end_num, :, :]
            train_input_d4 = train_data_d4[start_num:end_num, :, :]
            train_input_a4 = train_data_a4[start_num:end_num, :, :]
            label = train_targets[start_num:end_num]
            train_input_d1 = train_input_d1.to(DEVICE)
            train_input_d2 = train_input_d2.to(DEVICE)
            train_input_d3 = train_input_d3.to(DEVICE)
            train_input_d4 = train_input_d4.to(DEVICE)
            train_input_a4 = train_input_a4.to(DEVICE)
            label = label.to(DEVICE)
            start_num = end_num
            optimizer.zero_grad()
            outputs, loss = net(train_input_d1,train_input_d2,train_input_d3,train_input_d4,train_input_a4,label)
            #loss = model_loss(outputs, label)
            loss.backward(retain_graph=True)
            optimizer.step()
            acc_train = accuracy(outputs, label)
            loss_ = loss_ + loss
            acc_ = acc_ + acc_train
        loss_ = loss_ / (i + 1)
        acc_ = acc_ / (i + 1)

        net.eval()
        with torch.no_grad():
            val_data_d1 = val_data_d1.to(DEVICE)
            val_data_d2 = val_data_d2.to(DEVICE)
            val_data_d3 = val_data_d3.to(DEVICE)
            val_data_d4 = val_data_d4.to(DEVICE)
            val_data_a4 = val_data_a4.to(DEVICE)
            val_targets = val_targets.to(DEVICE)
            outputs_val, loss_val = net(val_data_d1,val_data_d2,val_data_d3,val_data_d4,val_data_a4,val_targets)
            #loss_val = model_loss(outputs_val, val_targets)
            acc_val = accuracy(outputs_val, val_targets)
        if acc_val > best_acc:
            best_acc = acc_val
            best_epoch = epoch
            best_output = outputs_val
            best_label = val_targets
            torch.save(net, path.join(model_save_path, 'Best_%d.pth' % (k)))
            #torch.save(net.state_dict(), './result/Best_' + str(k) + '.pt')
            print('better')
        print('loss_train=', loss_.item(), 'acc_train=', acc_, 'loss_val=', loss_val.item(), 'acc_val=', acc_val)

    #Evaluate
    model = torch.load(path.join(model_save_path, 'Best_%d.pth' % (k)))
    model.eval()
    with torch.no_grad():
        evaluate,_ = model(val_data_d1,val_data_d2,val_data_d3,val_data_d4,val_data_a4,val_targets)
        fold_acc = accuracy(evaluate, val_targets)
        pre = np.array(nn.Softmax(dim=1)(evaluate).cpu())
        pre = np.argmax(pre, 1)
        tru = np.array(val_targets.cpu())
        # fold_acc = accuracy(best_output,val_targets)
        if k == 0:
            Tru = tru
            Pre = pre
        else:
            Tru = np.concatenate((Tru, tru), axis=0)
            Pre = np.concatenate((Pre, pre), axis=0)
    print('------------------------Fold----------------------------')
    print('best-epoc', best_epoch, 'best_acc', best_acc, 'fold-acc',fold_acc)
    print('------------------------Fold----------------------------')
print(Tru.shape, Pre.shape)
scipy.io.savemat('predict_result_1.mat', {'Tru': Tru, 'Pre': Pre})
PrintScore(Tru, Pre, result_save_path)