from torch.autograd import Variable
from torch.utils.data import DataLoader
from learner import Learner
from loss import *
from dataset import *
import os
from sklearn import metrics
import argparse
from FFC import *
from matplotlib import pyplot as plt


def train(epoch):
    print('\nEpoch: %d' % epoch)
    global n_iter
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    i = 0
    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_train_loader, anomaly_train_loader)):
        i += 1
        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=1)
        # inputs = Variable(inputs)
        batch_size = inputs.shape[0]
        inputs = inputs.view(-1, inputs.size(-1)).to(device)
        outputs = model(inputs)
        loss = criterion(outputs, batch_size)
        optimizer.zero_grad()
        # grads = [x.grad for x in optimizer.param_groups[0]]
        # print(grads)
        loss.backward()
        # print(loss)
        print("len(a_memory): ", len(model.a_memory), "len(n_memory): ", len(model.n_memory))
        # print("len(n_memory)", len(model.n_memory))

        optimizer.step()
        train_loss += loss.item()
        n_iter += 1
        if args.nk and n_iter % optimize_iter == 0:
            with torch.no_grad():
                print("start optimizing memory")
                # model.clear_memory(epoch=epoch)
                model.optimize_memory()
                print("end optimizing memory")

    # print(model.a_memory)
    # print(model.n_memory)
    print('loss = {}'.format(train_loss / len(normal_train_loader)))
    scheduler.step()


# def test_abnormal_0(epoch):
#     model.eval()
#     global best_auc
#     auc = 0
#     visulization_iter = 200
#     visualize_flag = False
#     with torch.no_grad():
#         for i, (data, data2) in enumerate(zip(anomaly_test_loader, normal_test_loader)):
#             inputs, gts, frames = data
#             inputs = inputs.view(-1, inputs.size(-1)).to(torch.device('cuda'))
#             score = model(inputs)
#             score = score.cpu().detach().numpy()
#             score_list = np.zeros(frames[0])
#             step = np.round(np.linspace(0, frames[0] // 16, 33))
#
#             for j in range(32):
#                 score_list[int(step[j]) * 16:(int(step[j + 1])) * 16] = score[j]
#
#             gt_list = np.zeros(frames[0])
#             for k in range(len(gts) // 2):
#                 s = gts[k * 2]
#                 e = min(gts[k * 2 + 1], frames)
#                 gt_list[s - 1:e] = 1
#
#             inputs2, gts2, frames2 = data2
#             inputs2 = inputs2.view(-1, inputs2.size(-1)).to(torch.device('cuda'))
#             score2 = model(inputs2)
#             score2 = score2.cpu().detach().numpy()
#             score_list2 = np.zeros(frames2[0])
#             step2 = np.round(np.linspace(0, frames2[0] // 16, 33))
#             for kk in range(32):
#                 score_list2[int(step2[kk]) * 16:(int(step2[kk + 1])) * 16] = score2[kk]
#             gt_list2 = np.zeros(frames2[0])
#             score_list3 = np.concatenate((score_list, score_list2), axis=0)
#             gt_list3 = np.concatenate((gt_list, gt_list2), axis=0)
#             if visualize_flag and i // visulization_iter:
#                 plt.plot(gt_list3)
#                 plt.plot(score_list3)
#                 plt.show()
#             fpr, tpr, thresholds = metrics.roc_curve(gt_list3, score_list3, pos_label=1)
#             # print(thresholds)
#             auc += metrics.auc(fpr, tpr)
#
#         print('auc = {}', auc / 140)
#
#         if best_auc < auc / 140:
#             # ==========
#             print("No Saving")
#             best_auc = auc / 140
#             # ==========
#
#             # print('Saving..')
#             # state = {
#             #     'net': model.state_dict(),
#             # }
#             # if not os.path.isdir('checkpoint'):
#             #     os.mkdir('checkpoint')
#             # torch.save(state, './checkpoint/ckpt.pth')
#             # best_auc = auc / 140
#     return auc / 140


# def test_abnormal(epoch):
#     model.eval()
#     global best_auc
#     auc = 0
#     visulization_iter = 200
#     with torch.no_grad():
#         for i, (data, data2) in enumerate(zip(anomaly_test_loader, normal_test_loader)):
#             inputs, gts, frames = data
#             inputs = inputs.view(-1, inputs.size(-1)).to(torch.device('cuda'))
#             score = model(inputs)
#             score = score.cpu().detach().numpy()
#             score_list = np.zeros(frames[0])
#             step = np.round(np.linspace(0, frames[0] // 16, 33))
#             num_frames = frames[0]
#             # print(num_frames)
#             for idx, i in enumerate(range(0, num_frames - 16, 16)):
#                 score_list[i:i + 16] = score[idx]
#             score_list[i + 16:] = score[idx]
#             assert score_list[-1] != 0
#
#             # for j in range(32):
#             #     score_list[int(step[j]) * 16:(int(step[j + 1])) * 16] = score[j]
#
#             gt_list = np.zeros(frames[0])
#             for k in range(len(gts) // 2):
#                 s = gts[k * 2]
#                 e = min(gts[k * 2 + 1], frames)
#                 gt_list[s - 1:e] = 1
#
#             inputs2, gts2, frames2 = data2
#             inputs2 = inputs2.view(-1, inputs2.size(-1)).to(torch.device('cuda'))
#             score2 = model(inputs2)
#             score2 = score2.cpu().detach().numpy()
#             score_list2 = np.zeros(frames2[0])
#
#             # step2 = np.round(np.linspace(0, frames2[0] // 16, 33))
#             # for kk in range(32):
#             #     score_list2[int(step2[kk]) * 16:(int(step2[kk + 1])) * 16] = score2[kk]
#             num_frames = frames2[0]
#             for idx, i in enumerate(range(0, num_frames - 16, 16)):
#                 score_list2[i:i + 16] = score2[idx]
#             score_list2[i + 16:] = score2[idx]
#             assert score_list2[-1] != 0
#
#             gt_list2 = np.zeros(frames2[0])
#             score_list3 = np.concatenate((score_list, score_list2), axis=0)
#             gt_list3 = np.concatenate((gt_list, gt_list2), axis=0)
#             fpr, tpr, thresholds = metrics.roc_curve(gt_list3, score_list3, pos_label=1)
#             sample_auc = metrics.auc(fpr, tpr)
#             auc += sample_auc
#             if i // visulization_iter:
#                 plt.plot(gt_list3)
#                 plt.plot(score_list3)
#                 plt.title("auc: " + str(sample_auc))
#                 plt.show()
#             # print(thresholds)
#         auc = auc / 140
#         print('auc = {}', auc)
#
#         if best_auc < auc:
#             # ==========
#             print("No Saving")
#             best_auc = auc
#             # ==========
#
#             # print('Saving..')
#             # state = {
#             #     'net': model.state_dict(),
#             # }
#             # if not os.path.isdir('checkpoint'):
#             #     os.mkdir('checkpoint')
#             # torch.save(state, './checkpoint/ckpt.pth')
#             # best_auc = auc / 140
#     return auc

def test_abnormal(epoch):
    """对所有的视频一起进行计算auc"""
    model.eval()
    global best_auc
    auc = 0
    visualization = 200
    all_pred = []
    all_gt = []
    with torch.no_grad():
        for i, (data, data2) in enumerate(zip(anomaly_test_loader, normal_test_loader)):
            inputs, gts, frames = data
            inputs = inputs.view(-1, inputs.size(-1)).to(torch.device('cuda'))
            score = model(inputs)
            score = score.cpu().detach().numpy()
            num_frames = frames[0]
            score_list = np.zeros(num_frames)
            # print(num_frames)
            for idx, i in enumerate(range(0, num_frames - 16, 16)):
                assert len(score_list) > i + 16, "len of list:" + str(len(score_list)) + ", index: " + str(i + 16)
                # if i + 16 == len(score_list):
                score_list[i:] = score[idx]
                # else:
                #     score_list[i:i + 16] = score[idx]

            score_list[i + 16:] = score[idx]
            assert score_list[-1] != 0, score_list

            # step = np.round(np.linspace(0, frames[0] // 16, 33))
            # for j in range(32):
            #     score_list[int(step[j]) * 16:(int(step[j + 1])) * 16] = score[j]

            gt_list = np.zeros(frames[0])
            for k in range(len(gts) // 2):
                s = gts[k * 2]
                e = min(gts[k * 2 + 1], frames)
                gt_list[s - 1:e] = 1

            inputs2, gts2, frames2 = data2
            inputs2 = inputs2.view(-1, inputs2.size(-1)).to(torch.device('cuda'))
            score2 = model(inputs2)
            score2 = score2.cpu().detach().numpy()
            score_list2 = np.zeros(frames2[0])

            # step2 = np.round(np.linspace(0, frames2[0] // 16, 33))
            # for kk in range(32):
            #     score_list2[int(step2[kk]) * 16:(int(step2[kk + 1])) * 16] = score2[kk]
            num_frames = frames2[0]
            for idx, i in enumerate(range(0, num_frames - 16, 16)):
                score_list2[i:i + 16] = score2[idx]
            score_list2[i + 16:] = score2[idx]
            assert score_list2[-1] != 0

            gt_list2 = np.zeros(frames2[0])
            score_list3 = np.concatenate((score_list, score_list2), axis=0)
            gt_list3 = np.concatenate((gt_list, gt_list2), axis=0)

            all_gt.append(gt_list3)
            all_pred.append(score_list3)

            fpr, tpr, thresholds = metrics.roc_curve(gt_list3, score_list3, pos_label=1)
            sample_auc = metrics.auc(fpr, tpr)
            auc += sample_auc
            if i // visualization:
                plt.plot(gt_list3)
                plt.plot(score_list3)
                plt.title("auc: " + str(sample_auc))
            plt.show()
            # print(thresholds)
        auc_video = auc / 140
        print('auc_video = {}', auc_video)
        all_pred = np.concatenate(all_pred, axis=0)
        all_gt = np.concatenate(all_gt, axis=0)
        print("len of pred", len(all_pred))
        print("len of gt", len(all_gt))
        fpr, tpr, thresholds = metrics.roc_curve(all_gt, all_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print('auc = {}', auc)
        if best_auc < auc:
            # ==========
            print("No Saving")
            best_auc = auc
            # ==========

            # print('Saving..')
            # state = {
            #     'net': model.state_dict(),
            # }
            # if not os.path.isdir('checkpoint'):
            #     os.mkdir('checkpoint')
            # torch.save(state, './checkpoint/ckpt.pth')
            # best_auc = auc / 140
    return auc


def set_seed(random_state: int = 0):
    if random_state is not None:
        torch.manual_seed(random_state)  # cpu
        torch.cuda.manual_seed(random_state)  # gpu
        torch.cuda.manual_seed_all(random_state)
        np.random.seed(random_state)  # numpy
        random.seed(random_state)  # random and transforms
        torch.backends.cudnn.deterministic = True  # cudnn
        torch.backends.cudnn.benchmark = True
        os.environ['PYTHONHASHSEED'] = str(random_state)


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='PyTorch MIL Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--w', default=0.0010000000474974513, type=float, help='weight_decay')
    parser.add_argument('--modality', default='TWO', type=str, help='modality')
    parser.add_argument('--input_dim', default=512, type=int, help='input_dim')
    parser.add_argument('--epoch', default=80, type=int, help='max_epoch')
    parser.add_argument('--train_batch_size', default=30, type=int, help='train_batch_size')
    parser.add_argument('--test_batch_size', default=1, type=int, help='test_batch_size')
    parser.add_argument('--optimize_iter', default=3, type=int, help='optimize_iter')
    parser.add_argument('--num_key_memory', default=10, type=int, help='num_key_memory')
    parser.add_argument('--max_memory_size', default=15, type=int, help='max_memory_size')
    parser.add_argument('--memory_rate', default=0.5, type=float, help='memory_rate')
    parser.add_argument('--clear_rate', default=0.3, type=float, help='clear_rate')
    # 越大，越容易增加新的数据
    parser.add_argument('--threshold_caption_score', default=1.0, type=float, help='threshold_caption_score')
    parser.add_argument('--drop', default=0.6, type=float, help='dropout_rate')
    parser.add_argument('--FFC', '-r', action='store_true', help='FFC')
    parser.add_argument('--nk', action='store_false', help='nk')
    args = parser.parse_args()

    set_seed(0)

    best_auc = 0
    n_iter = 0

    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    optimize_iter = args.optimize_iter

    normal_train_dataset = Normal_Loader(is_train=1, modality=args.modality, feature_dim=args.input_dim)
    normal_test_dataset = Normal_Loader(is_train=0, modality=args.modality, feature_dim=args.input_dim)
    anomaly_train_dataset = Anomaly_Loader(is_train=1, modality=args.modality, feature_dim=args.input_dim)
    anomaly_test_dataset = Anomaly_Loader(is_train=0, modality=args.modality, feature_dim=args.input_dim)

    normal_train_loader = DataLoader(normal_train_dataset, batch_size=train_batch_size, shuffle=True)
    normal_test_loader = DataLoader(normal_test_dataset, batch_size=test_batch_size, shuffle=False)
    anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=train_batch_size, shuffle=True)
    anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=test_batch_size, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.FFC:
        model = Learner2(input_dim=args.input_dim, drop_p=args.drop).to(device)
    else:
        # model = Learner(input_dim=args.input_dim, drop_p=args.drop).to(device)
        model = Learner(input_dim=args.input_dim, drop_p=args.drop, memory_rate=args.memory_rate,
                        num_key_memory=args.num_key_memory, max_memory_size=args.max_memory_size,
                        threshold_caption_score=args.threshold_caption_score, nk=args.nk).to(device)

    # print('Loading..')
    # state = {
    #     'net': model.state_dict(),
    # }
    # if not os.path.isdir('checkpoint'):
    #     os.mkdir('checkpoint')
    # model_save_path = r"./checkpoint/ckpt.pth"
    # assert os.path.exists(model_save_path)
    # state = torch.load(model_save_path)
    # model.load_state_dict(state['net'])

    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.w)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50])
    criterion = MIL
    aucs = []
    for epoch in range(0, args.epoch):
        train(epoch)
        auc = test_abnormal(epoch)
        aucs.append(auc)
        # model.optimize_memory()
        model.clear_memory(rate=args.clear_rate, epoch=epoch)
        print("best_auc", best_auc)
    # print(aucs)
    print("best_auc", best_auc)
