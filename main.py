from torch.autograd import Variable
from torch.utils.data import DataLoader
from learner import Learner
from loss import MIL, post_process, save_result_2
from dataset import *
import os
from sklearn import metrics
import argparse
from my_learner2 import Learner2
from matplotlib import pyplot as plt

from transformer import TFLeaner


def train(epoch):
    print('\nEpoch: %d' % epoch)
    global n_iter
    first_flag = False
    iter_in_epoch = 0
    first_optimize_iter = args.first_optimize_iter
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    i = 0
    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_train_loader, anomaly_train_loader)):
        i += 1
        if args.input_dim in [2049, 1280, 1024]:
            anomaly_inputs, a_names = anomaly_inputs
            normal_inputs, n_names = normal_inputs
            names = torch.cat([a_names, n_names], dim=0)  # (1,60)
        else:
            names = None
        # print(a_names)
        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=1)
        # names = torch.stack([a_names, n_names], dim=0)  # (2,30)
        # inputs = Variable(inputs)
        batch_size = inputs.shape[0]
        inputs = inputs.view(-1, inputs.size(-1)).to(device)
        outputs = model(inputs, names)
        loss = criterion(outputs, batch_size, args=args)
        # sta = model.memory_stability() * 0.0001
        # print("memory_stability", sta)
        # loss += sta
        # similaity_a_n = model.similarity_a_n_memory_space() * 0.0001
        # print("similaity_a_n", similaity_a_n)
        # loss += similaity_a_n
        optimizer.zero_grad()
        # grads = [x.grad for x in optimizer.param_groups[0]]
        # print(grads)
        loss.backward()
        # print(loss)
        optimizer.step()
        train_loss += loss.item()
        n_iter += 1
        iter_in_epoch += 1
        if args.nk:
            print("len(a_memory): ", len(model.a_memory), "len(n_memory): ", len(model.n_memory))

            if first_flag and iter_in_epoch == first_optimize_iter:
                with torch.no_grad():
                    print("start optimizing memory")
                    model.optimize_memory()
                    print("end optimizing memory")
                first_flag = False

            if n_iter % optimize_iter == 0 and not first_flag:
                with torch.no_grad():
                    print("start optimizing memory")
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

def test_abnormal(epoch, patient, args):
    """对所有的视频一起进行计算auc"""
    model.eval()
    global best_auc
    auc = 0
    visualization = 20
    all_pred = []
    all_gt = []
    with torch.no_grad():
        for i, (data, data2) in enumerate(zip(anomaly_test_loader, normal_test_loader)):
            inputs, gts, frames, video_names = data
            anomaly_video_name = video_names[0]
            inputs = inputs.view(-1, inputs.size(-1)).to(torch.device('cuda'))
            score = model(inputs)
            score = score.cpu().detach().numpy()
            num_frames = frames[0]
            score_list = np.zeros(num_frames)
            # print(num_frames)

            if args.input_dim in [2048, 512]:
                # 将视频分为32个部分然后分别计算分数
                step = np.round(np.linspace(0, num_frames // 16, 33))
                for j in range(32):
                    score_list[int(step[j]) * 16:(int(step[j + 1])) * 16] = score[j]
            else:
                for idx, i in enumerate(range(0, num_frames - 16, 16)):
                    assert len(score_list) > i + 16, "len of list:" + str(len(score_list)) + ", index: " + str(i + 16)
                    # if i + 16 == len(score_list):
                    score_list[i:] = score[idx]
                    # else:
                    #     score_list[i:i + 16] = score[idx]
                score_list[i + 16:] = score[idx]
                # assert score_list[-1] != 0, score_list

            # 分别处理各自的分数
            gt_list = np.zeros(frames[0])
            for k in range(len(gts) // 2):
                s = gts[k * 2]
                e = min(gts[k * 2 + 1], frames)
                gt_list[s - 1:e] = 1
            # save_result(anomaly_video_name, score_list, gt_list)

            inputs2, gts2, frames2, video_names2 = data2
            normal_video_name = video_names2[0]
            inputs2 = inputs2.view(-1, inputs2.size(-1)).to(torch.device('cuda'))
            score2 = model(inputs2)
            score2 = score2.cpu().detach().numpy()
            score_list2 = np.zeros(frames2[0])

            if args.input_dim in [2048, 512]:
                ### 将测试结果分为21个断层，然后计算结果
                step2 = np.round(np.linspace(0, frames2[0] // 16, 33))
                for kk in range(32):
                    score_list2[int(step2[kk]) * 16:(int(step2[kk + 1])) * 16] = score2[kk]
            else:
                num_frames = frames2[0]
                for idx, i in enumerate(range(0, num_frames - 16, 16)):
                    score_list2[i:i + 16] = score2[idx]
                score_list2[i + 16:] = score2[idx]
                # assert score_list2[-1] != 0 ,score2[idx]

            gt_list2 = np.zeros(frames2[0])
            # save_result(normal_video_name, score_list2, gt_list2)

            save_result_2(anomaly_video_name, normal_video_name, score_list, gt_list, score_list2, gt_list2)
            score_list3 = np.concatenate((score_list, score_list2), axis=0)
            gt_list3 = np.concatenate((gt_list, gt_list2), axis=0)

            all_gt.append(gt_list3)
            all_pred.append(score_list3)

            fpr, tpr, thresholds = metrics.roc_curve(gt_list3, score_list3, pos_label=1)
            sample_auc = metrics.auc(fpr, tpr)
            auc += sample_auc
            if random.random() < 0.1:
                # print(video_names, video_names2)
                plt.plot(gt_list3, label='gt')
                plt.plot(score_list3, label='prediction')

                # 　画分界线
                plt.axvline(len(score_list), 1.0, 0.0, color='green')
                plt.text(0, 0.5, anomaly_video_name)
                plt.text(0, 1, normal_video_name)
                plt.title("auc: {}".format(sample_auc))
                # plt.title("auc: {},\n {}, \n{}".format(sample_auc, anomaly_video_name, normal_video_name))
                plt.legend()
                plt.xlabel('frames')
                plt.ylabel('anomaly score')
                plt.show()
        auc_video = auc / 140
        print('auc_video = {}', auc_video)
        all_pred = np.concatenate(all_pred, axis=0)
        all_gt = np.concatenate(all_gt, axis=0)
        # print("len of pred", len(all_pred))
        # print("len of gt", len(all_gt))
        # tmp_all_gt = all_gt.copy()
        # all_gt = post_process(tmp_all_gt)
        fpr, tpr, thresholds = metrics.roc_curve(all_gt, all_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print('auc = {}', auc)
        if best_auc < auc:
            # ==========
            print("No Saving")
            best_auc = auc
            # ==========
            patient = args.patient

            # print('Saving..')
            # state = {
            #     'net': model.state_dict(),
            # }
            # if not os.path.isdir('checkpoint'):
            #     os.mkdir('checkpoint')
            # torch.save(state, './checkpoint/ckpt.pth')
            # best_auc = auc / 140
        else:
            patient = patient - 1
            print("patient decrease", patient)

    return auc, patient


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
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--w', default=0.0010000000474974513, type=float, help='weight_decay')
    parser.add_argument('--modality', default='TWO', type=str, help='modality')
    parser.add_argument('--input_dim', default=512, type=int, help='input_dim')
    parser.add_argument('--epoch', default=80, type=int, help='max_epoch')
    parser.add_argument('--train_batch_size', default=30, type=int, help='train_batch_size')
    parser.add_argument('--test_batch_size', default=1, type=int, help='test_batch_size')
    parser.add_argument('--optimize_iter', default=3, type=int, help='optimize_iter')
    parser.add_argument('--num_key_memory', default=10, type=int, help='num_key_memory')
    parser.add_argument('--max_memory_size', default=15, type=int, help='max_memory_size')
    parser.add_argument('--embedding_dim', default=512, type=int, help='embedding_dim')
    parser.add_argument('--n_layer', default=2, type=int, help='n_layer')
    parser.add_argument('--n_head', default=1, type=int, help='n_head')
    parser.add_argument('--memory_rate', default=0.5, type=float, help='memory_rate')
    parser.add_argument('--clear_rate', default=0.5, type=float, help='clear_rate')  # 越小删除的就越多,就算变为0，也会留下一个memory
    # 越大，越容易增加新的数据
    parser.add_argument('--threshold_caption_score', default=1.0, type=float, help='threshold_caption_score')
    parser.add_argument('--drop', default=0.6, type=float, help='dropout_rate')
    parser.add_argument('--FFC', '-r', action='store_true', help='FFC')
    parser.add_argument('--tf', action='store_true', help='transformer')
    parser.add_argument('--nk', default=False, action='store_true', help='nk')
    parser.add_argument('--update_threshold', default=False, action='store_true', help='update_threshold')
    parser.add_argument('--patient', default=5, type=int, help='patient')
    parser.add_argument('--a_topk', default=1, type=int, help='add top k anomaly into anomaly memory space')
    parser.add_argument('--topk_score', default=7, type=int,
                        help='use the distance with top k memory to represent the similarity')
    parser.add_argument('--first_optimize_iter', default=3, type=int, help='first_optimize_iter')
    parser.add_argument('--distance', default="mul", type=str,
                        help='how to calculate distance between two features. [mul, cos, mse]')
    parser.add_argument('--feature_name', default="", type=str,
                        help='select the feature to be used, [uio-caption, uio-vqa1, uio-caption-vqa, uio_opt_region, uio_fixed_region]')

    parser.add_argument('--init_memory', default="one", type=str,
                        help='when to init memory init memory  [fisrt, epoch, no]')
    parser.add_argument('--check_caption_memory', default=False, action='store_true', help='nk')

    parser.add_argument('--loss_topk', default=1, type=int, help='loss_topk')

    args = parser.parse_args()

    set_seed(0)

    feature_names = ["uio_caption", "uio_vqa1", "uio_caption_vqa1", "uio_opt_region", "uio_fixed_region", "clip", "i3d"]

    if "uio" in args.feature_name:
        args.check_caption_memory = True

    best_auc = 0
    n_iter = 0

    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    optimize_iter = args.optimize_iter
    patient = args.patient

    if args.input_dim == 512:
        print("Using CLIP feature")
    elif args.input_dim == 2048:
        print("Using I3D feature")
    elif args.input_dim == 2560:
        print("Using I3D+CLIP feature")
    else:
        print("Using UIO feature")
        # raise RuntimeError("Not Support such dataset")

    normal_train_dataset = Normal_Loader(is_train=1, modality=args.modality, feature_dim=args.input_dim, args=args)
    normal_test_dataset = Normal_Loader(is_train=0, modality=args.modality, feature_dim=args.input_dim, args=args)
    anomaly_train_dataset = Anomaly_Loader(is_train=1, modality=args.modality, feature_dim=args.input_dim, args=args)
    anomaly_test_dataset = Anomaly_Loader(is_train=0, modality=args.modality, feature_dim=args.input_dim, args=args)

    normal_train_loader = DataLoader(normal_train_dataset, batch_size=train_batch_size, shuffle=True)
    normal_test_loader = DataLoader(normal_test_dataset, batch_size=test_batch_size, shuffle=False)
    anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=train_batch_size, shuffle=True)
    anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=test_batch_size, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.FFC:
        model = Learner2(input_dim=args.input_dim, drop_p=args.drop).to(device)
    elif args.tf:
        model = TFLeaner(args.input_dim, args.embedding_dim, args.n_layer, args.n_head, args).to(device)
    else:
        # model = Learner(input_dim=args.input_dim, drop_p=args.drop).to(device)
        model = Learner(input_dim=args.input_dim, drop_p=args.drop, memory_rate=args.memory_rate,
                        num_key_memory=args.num_key_memory, max_memory_size=args.max_memory_size,
                        threshold_caption_score=args.threshold_caption_score, nk=args.nk, args=args).to(device)

    # normal_train_dataset.get_snippet_feature("203-9")

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
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 5], gamma=0.1)
    criterion = MIL
    aucs = []
    if args.init_memory == "one":
        model.init_memory_space(anomaly_train_dataset, normal_train_dataset, args)

    for epoch in range(0, args.epoch):
        # 训练之前初始化一下记忆空间
        if args.init_memory == "epoch":
            model.init_memory_space(anomaly_train_dataset, normal_train_dataset, args)

        train(epoch)
        # print results
        if args.check_caption_memory:
            a_caption_memory, n_caption_memory = model.show_stored_snippet_ids()
            print("a_caption_memory: ")
            for snippet_id in a_caption_memory:
                result = anomaly_train_dataset.show_caption(snippet_id)
                print(snippet_id, result)
            print("n_caption_memory: ")
            for snippet_id in n_caption_memory:
                result = normal_train_dataset.show_caption(snippet_id)
                print(snippet_id, result)

        auc, patient = test_abnormal(epoch, patient, args)
        aucs.append(auc)
        if args.nk:
            model.clear_memory(rate=args.clear_rate, epoch=epoch)
        # print("a_caption_memory:", a_caption_memory)
        # print("n_caption_memory:", n_caption_memory)
        print("best_auc", best_auc)
        if patient == 0:
            print("Easy stop!")
            break
    print(aucs)
    print("best_auc", best_auc)
    plt.plot(aucs)
    plt.show()

# 0.8180977589971383  post-processed
# 0.8181509890961944　　no post-processed better


# 0.8254352179071174 output = attn + output.mean(dim=2)
# 0.5008233030470083 output = attn
# 0.8196318089303685
# 0.8254352179071174  output= output /2
# output = 2 * attn + output.mean(dim=2)  0.8190337505714058
# 　0.8221853467803804　　 attn + output.mean(dim=2) * 10

# 0.8263817288643429
# 0.8250175510179918

# 0.8353636423009376 2048 128
# 0.8354656209751689　2048 256

# 0.8292918654273516
#  0.8239086002514562

# 0.8112955221306541  留下尽量相似的记录
# 0.8312737009212092  留下不相似的记录


# 将caption对应的id保存到记忆空间，随着记忆空间的更新而更新captionid
#     captionid 保存形式 “video_id-snippet-id"
# 输出最后的captionid
# 根据captionid找到对应的caption内容和视频的内容
#     dataset根据snippetId找到对应的video
#     更具video找到对应的caption文件
#     在用sippetid找到对应的caption

# epoch-1
# ['234-6', '94-16', '568-5', '750-18', '679-14', '763-0', '641-25', '545-27', '171-13', '429-7']
# ['752-16', '739-25', '645-31', '632-11', '125-16', '388-9', '113-23', '609-9', '433-11', '557-19']
