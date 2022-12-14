import torch
import torch.nn.functional as F

def MIL(y_pred, batch_size, is_transformer=0, model=None):
    loss = torch.tensor(0.).cuda()
    loss_intra = torch.tensor(0.).cuda()
    sparsity = torch.tensor(0.).cuda()
    smooth = torch.tensor(0.).cuda()
    if is_transformer == 0:
        y_pred = y_pred.view(batch_size, -1)
    else:
        y_pred = torch.sigmoid(y_pred)
    indexes_anomaly = []
    for i in range(batch_size):
        # anomaly_index = torch.randperm(32).cuda()
        # normal_index = torch.randperm(32).cuda()
        #
        # y_anomaly = y_pred[i, :32][anomaly_index]
        # y_normal  = y_pred[i, 32:][normal_index]

        y_anomaly = y_pred[i, :32]
        y_normal = y_pred[i, 32:]

        y_anomaly_max = torch.max(y_anomaly)  # anomaly
        index_anomaly = torch.argmax(y_anomaly)
        indexes_anomaly.append(index_anomaly)

        y_anomaly_min = torch.min(y_anomaly)

        y_normal_max = torch.max(y_normal)  # normal
        y_normal_min = torch.min(y_normal)

        loss += F.relu(1.-y_anomaly_max+y_normal_max) # 和下面那个没有太大的区别
        # loss += F.relu(2. - y_anomaly_max + y_normal_max)
        # 试着去掉其他的loss函数，看看会有什么变化
        sparsity += torch.sum(y_anomaly)*0.00008
        smooth += torch.sum((y_pred[i,:31] - y_pred[i,1:32])**2)*0.00008
    # print("sparsity loss:", sparsity/batch_size)
    # print("smooth loss:", smooth/batch_size)
    loss = (loss+sparsity+smooth)/batch_size

    return loss
