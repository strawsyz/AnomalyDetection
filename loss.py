import os.path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

from file_util import make_directory


def MIL(y_pred, batch_size, is_transformer=0, model=None, args=None):
    loss = torch.tensor(0.).cuda()
    loss_intra = torch.tensor(0.).cuda()
    sparsity = torch.tensor(0.).cuda()
    smooth = torch.tensor(0.).cuda()
    if is_transformer == 0:
        y_pred = y_pred.view(batch_size, -1)
    else:
        y_pred = torch.sigmoid(y_pred)

    if args is not None:
        k = args.loss_topk
    else:
        k = 2

    indexes_anomaly = []
    for i in range(batch_size):
        # anomaly_index = torch.randperm(32).cuda()
        # normal_index = torch.randperm(32).cuda()
        #
        # y_anomaly = y_pred[i, :32][anomaly_index]
        # y_normal  = y_pred[i, 32:][normal_index]

        y_anomaly = y_pred[i, :32]
        y_normal = y_pred[i, 32:]

        if k == 1:
            y_anomaly_max = torch.max(y_anomaly)  # anomaly
        else:
            values, indices = y_anomaly.topk(k=k, dim=0, largest=True, sorted=True)
            y_anomaly_max = torch.mean(values)

        index_anomaly = torch.argmax(y_anomaly)
        indexes_anomaly.append(index_anomaly)

        # y_anomaly_min = torch.min(y_anomaly)

        # y_normal_max = torch.mean(y_normal)
        if k == 1:
            y_normal_max = torch.max(y_normal)  # normal
        else:
            values, indices = y_normal.topk(k=k, dim=0, largest=True, sorted=True)
            y_normal_max = torch.mean(values)
        # y_normal_min = torch.min(y_normal)

        loss += F.relu(1. - y_anomaly_max + y_normal_max)  # 和下面那个没有太大的区别
        # loss += F.relu(2. - y_anomaly_max + y_normal_max)
        # 试着去掉其他的loss函数，看看会有什么变化
        sparsity += torch.sum(y_anomaly) * 0.00008
        smooth += torch.sum((y_pred[i, :31] - y_pred[i, 1:32]) ** 2) * 0.00008
    # print("sparsity loss:", sparsity/batch_size)
    # print("smooth loss:", smooth/batch_size)
    loss = (loss + sparsity + smooth) / batch_size

    return loss


def save_result(video_name, prediction, gt):
    result_root_path = "results/"
    video_name = video_name.split("/")[-1]
    make_directory(result_root_path)
    target_filepath = os.path.join(result_root_path, video_name)
    data = np.array([prediction, gt])
    np.save(target_filepath, data)


def save_result_2(anomaly_video_name, normal_video_name, a_prediction, a_gt, n_prediction, n_gt):
    result_root_path = "results/"
    anomaly_video_name = anomaly_video_name.split("/")[-1]
    normal_video_name = normal_video_name.split("/")[-1]
    make_directory(result_root_path)
    target_filepath = os.path.join(result_root_path, normal_video_name)
    data = np.array([anomaly_video_name, normal_video_name, a_prediction, a_gt, n_prediction, n_gt])
    np.save(target_filepath, data)


def post_process(scores: list, len_a=None):
    # 直接二值化. 不管用什么阈值，效果都比较差，
    # threshold = 0.1  # 二值化的阈值需要考虑, 效果很差最少需要0.1
    # # threshold = np.mean(scores[:len_a])  # -0.07
    # # threshold = np.median(scores)  # -11.1
    # # threshold = np.max(scores[len_a:])  # 0.15
    # scores[:len_a] = np.where(scores[:len_a] > threshold, 1, 0)
    # return scores

    # 找到数据的最高点，根据最高点找到两边结束的位置
    # 如何找到结束的位置
    #     找到连续X次不增加的点
    #     根据两边的点，找到分界的线

    # 感觉有太多的frame上的分数过于高了

    # 将前后的分数变化大的点作为分界点
    # 根据分界点内的平均异常分数高低，决定是否异常或者是正常

    # if np.max(scores[:len_a]) < 0.5:  #  先用最大值判断是否有异常
    #     scores[:len_a] = np.where(scores[:len_a] <np.median(scores[:len_a]),0, 1)
    # if np.max(scores[len_a:]) < 0.5:
    #     scores[len_a:] = np.where(scores[len_a:] <np.median(scores[len_a:]),0, 1)
    #     # scores[len_a:] = 0
    # return scores

    def find_range(scores, index):
        start = index
        end = index
        max_score = scores[index]
        slope = 2

        for i in range(start, 0, -1):
            if scores[i] > max_score / slope:
                start -= 1
                max_score = scores[i]
            # elif scores[i] == max_score:
            #     start += 1
            else:
                break
        max_score = scores[index]
        for i in range(end, len(scores), 1):
            if scores[i] > max_score / slope:
                end += 1
            else:
                break

        return (start, end)

    def set_range_pos(scores, range):
        scores[range[0]: range[1]] = 1
        return scores

    # for index, score in enumerate(scores):
    # for index in range(len(scores)):
    #     score = scores[index]
    tmp_scores = scores.copy()
    ranges = []
    while True:
        index = np.argmax(scores)
        if scores[index] > 0.5:
            range_ = find_range(scores, index)
            scores[range_[0]:range_[1]] = 0
            ranges.append(range_)
        else:
            break

    for range_ in ranges:
        tmp_scores = set_range_pos(tmp_scores, range_)

    return tmp_scores


def get_auc(prediction, gt):
    fpr, tpr, thresholds = metrics.roc_curve(gt, prediction, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc


def check_post_process(data_filepath, visual_flag=False):
    data = np.load(data_filepath, allow_pickle=True)
    if len(data) != 6:
        return None
    anomaly_video_name, normal_video_name, a_prediction, a_gt, n_prediction, n_gt = data

    # data = np.load("/workspace/AnomalyDetection/results/Abuse030_x264.mp4.npy")
    assert len(a_prediction) == len(a_gt)
    assert len(n_prediction) == len(n_gt)
    old_prediction = np.concatenate((a_prediction, n_prediction), axis=0)
    gt = np.concatenate((a_gt, n_gt), axis=0)
    old_auc = get_auc(old_prediction, gt)
    prediction = old_prediction.copy()
    prediction = post_process(prediction, len(a_prediction))
    new_auc = get_auc(prediction, gt)
    from matplotlib import pyplot as plt

    if old_auc < new_auc:
        print("better")
        if visual_flag:
            # print(video_names, video_names2)
            plt.plot(gt, label='gt')
            plt.plot(old_prediction, label='old_prediction')
            # plt.plot(prediction, label='prediction')

            # 　画分界线
            plt.axvline(len(a_prediction), 1.0, 0.0, color='green')
            plt.text(0, 0.5, anomaly_video_name)
            plt.text(0, 1, normal_video_name)
            plt.title("auc: {:.2} -> {:.2}".format(old_auc, new_auc))
            # plt.title("auc: {},\n {}, \n{}".format(sample_auc, anomaly_video_name, normal_video_name))
            plt.legend()
            plt.xlabel('frames')
            plt.ylabel('anomaly score')
            plt.show()

            plt.figure()
            plt.plot(gt, label='gt')
            # plt.plot(old_prediction, label='old_pred')
            plt.plot(prediction, label='new_prediction')

            # 　画分界线
            plt.axvline(len(a_prediction), 1.0, 0.0, color='green')
            plt.text(0, 0.5, anomaly_video_name)
            plt.text(0, 1, normal_video_name)
            plt.title("auc: {:.2} -> {:.2}".format(old_auc, new_auc))
            # plt.title("auc: {},\n {}, \n{}".format(sample_auc, anomaly_video_name, normal_video_name))
            plt.legend()
            plt.xlabel('frames')
            plt.ylabel('anomaly score')
            plt.show()
    elif old_auc > new_auc:
        print("worse")
    else:
        pass

    return (new_auc - old_auc), new_auc, old_auc


if __name__ == '__main__':
    # test post-preprocess
    data_filepath = r"/workspace/AnomalyDetection/results/Abuse030_x264.mp4.npy"
    data_root_path = r"/workspace/AnomalyDetection/results"
    all_gain = 0
    new_aucs = 0
    old_aucs = 0
    visual_flag = False
    sampels = len(os.listdir(data_root_path)) / 2
    for filename in os.listdir(data_root_path):
        data_filepath = os.path.join(data_root_path, filename)
        gain = check_post_process(data_filepath, visual_flag=visual_flag)
        if gain is None:
            continue
        gain, new_auc, old_auc = gain
        all_gain += gain
        new_aucs += new_auc
        old_aucs += old_auc
    print("average gain", all_gain / sampels)
    print("new auc", new_aucs / sampels)
    print("old auc", old_aucs / sampels)
# 0.8359331027138271  k,1
# 0.7903967146544392  n.k.all
# 0.8381323656829667  k,3
# 0.8150558935418355  CLIP
# 0.8274786908983395　CLIP + I3D
# 0.8163858463081177 CLIP,32
