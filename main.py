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
    key_snippet_ids = []
    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_train_loader, anomaly_train_loader)):
        i += 1
        # if args.input_dim in [2049, 1280, 1024]:
        # if "uio" in args.feature_name:
        anomaly_inputs, a_names, anomaly_embedding = anomaly_inputs
        normal_inputs, n_names, normal_embedding = normal_inputs
        names = torch.cat([a_names, n_names], dim=0)  # (1,60)
        # else:
        #     anomaly_inputs, anomaly_embedding = anomaly_inputs
        #     normal_inputs, normal_embedding = normal_inputs
        # print(a_names)
        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=1)
        embeddings = torch.cat([anomaly_embedding, normal_embedding], dim=1)
        # names = torch.stack([a_names, n_names], dim=0)  # (2,30)
        # inputs = Variable(inputs)
        batch_size = inputs.shape[0]
        inputs = inputs.view(-1, inputs.size(-1)).to(device)
        embeddings = embeddings.view(-1, embeddings.size(-1)).to(device)
        # outputs = model(inputs, names)
        outputs, snippet_ids = model(inputs, caption_embeddings=embeddings, video_ids=names)
        if snippet_ids != []:
            key_snippet_ids.extend(snippet_ids)
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
        optimizer.step()

        if args.nk:
            print("len(a_memory): ", len(model.a_memory), "len(n_memory): ", len(model.n_memory))

            if (first_flag and iter_in_epoch == first_optimize_iter) or (
                    n_iter % optimize_iter == 0 and not first_flag):
                with torch.no_grad():
                    print("start optimizing memory")
                    model.optimize_memory()
                    print("end optimizing memory")
                first_flag = False

        train_loss = train_loss + loss.item()
        n_iter += 1
        iter_in_epoch += 1

    # print(model.a_memory)
    # print(model.n_memory)
    print('loss = {}'.format(train_loss / len(normal_train_loader)))
    scheduler.step()
    print(key_snippet_ids)


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
            inputs, gts, frames, video_names, embedding = data
            # inputs = embedding
            anomaly_video_name = video_names[0]
            inputs = inputs.view(-1, inputs.size(-1)).to(torch.device('cuda'))
            embedding = embedding.view(-1, embedding.size(-1)).to(torch.device('cuda'))
            score = model(inputs, caption_embeddings=embedding)
            score = score.cpu().detach().numpy()
            num_frames = frames[0]
            score_list = np.zeros(num_frames)
            # print(num_frames)

            step = np.round(np.linspace(0, num_frames // 16, 33))
            for j in range(32):
                score_list[int(step[j]) * 16:(int(step[j + 1])) * 16] = score[j]

            # if args.input_dim in [2048, 512]:
            #     # 将视频分为32个部分然后分别计算分数
            #     step = np.round(np.linspace(0, num_frames // 16, 33))
            #     for j in range(32):
            #         score_list[int(step[j]) * 16:(int(step[j + 1])) * 16] = score[j]
            # else:
            # 消除的代码0627
            # for idx, i in enumerate(range(0, num_frames - 16, 16)):
            #     assert len(score_list) > i + 16, "len of list:" + str(len(score_list)) + ", index: " + str(i + 16)
            #     score_list[i:] = score[idx]
            # score_list[i + 16:] = score[idx]

            # if i + 16 == len(score_list):
            # else:
            #     score_list[i:i + 16] = score[idx]
            # assert score_list[-1] != 0, score_list

            # 分别处理各自的分数
            gt_list = np.zeros(frames[0])
            for k in range(len(gts) // 2):
                s = gts[k * 2]
                e = min(gts[k * 2 + 1], frames)
                gt_list[s - 1:e] = 1
            # save_result(anomaly_video_name, score_list, gt_list)

            inputs2, gts2, frames2, video_names2, embedding2 = data2
            # inputs2 = embedding2
            normal_video_name = video_names2[0]
            inputs2 = inputs2.view(-1, inputs2.size(-1)).to(torch.device('cuda'))
            embedding2 = embedding2.view(-1, embedding2.size(-1)).to(torch.device('cuda'))
            # score2 = model(inputs2)
            score2 = model(inputs2, caption_embeddings=embedding2)
            score2 = score2.cpu().detach().numpy()
            score_list2 = np.zeros(frames2[0])

            step2 = np.round(np.linspace(0, frames2[0] // 16, 33))
            for kk in range(32):
                score_list2[int(step2[kk]) * 16:(int(step2[kk + 1])) * 16] = score2[kk]

            # if args.input_dim in [2048, 512]:
            #     ### 将测试结果分为21个断层，然后计算结果
            #     step2 = np.round(np.linspace(0, frames2[0] // 16, 33))
            #     for kk in range(32):
            #         score_list2[int(step2[kk]) * 16:(int(step2[kk + 1])) * 16] = score2[kk]
            # else:
            # num_frames = frames2[0]
            # for idx, i in enumerate(range(0, num_frames - 16, 16)):
            #     score_list2[i:i + 16] = score2[idx]
            # score_list2[i + 16:] = score2[idx]
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
            # if random.random() < 0.1:
            if anomaly_video_name == "Arson/Arson022_x264.mp4":
                # print(video_names, video_names2)
                plt.plot(gt_list3, label='gt')
                plt.plot(score_list3, label='prediction')
                # np.save(fr"feature-prediction-epoch{epoch}", score_list)
                # np.save(fr"gt-epoch", gt_list)
                # 　画分界线
                plt.axvline(len(score_list), 1.0, 0.0, color='green')
                plt.text(0, 0.5, anomaly_video_name)
                plt.text(0, 1, normal_video_name)
                plt.title(f"epoch: {epoch}, auc: {sample_auc:.2}")
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

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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
    parser.add_argument('--nk', action='store_true', help='nk')
    parser.add_argument('--update_threshold', default=False, action='store_true', help='update_threshold')
    parser.add_argument('--patient', default=8, type=int, help='patient')
    parser.add_argument('--a_topk', default=1, type=int, help='add top k anomaly into anomaly memory space')
    parser.add_argument('--topk_score', default=7, type=int,
                        help='use the distance with top k memory to represent the similarity')
    parser.add_argument('--first_optimize_iter', default=3, type=int, help='first_optimize_iter')
    parser.add_argument('--distance', default="mul", type=str,
                        help='how to calculate distance between two features.  ["mul", "mul-avg", "ss", "cos", "cos-abs", "mse", "mul-abs"]')

    parser.add_argument('--feature_name', default="", type=str,
                        help='select the feature to be used, [uio-caption, uio-vqa1, uio-caption-vqa, uio_opt_region, uio_fixed_region]')

    parser.add_argument('--init_memory', default="no", type=str,
                        help='when to init memory init memory  [first, epoch, no]')
    parser.add_argument('--check_caption_memory', default=False, action='store_true', help='nk')

    parser.add_argument('--loss_topk', default=1, type=int, help='loss_topk')
    parser.add_argument('--seed', default=0, type=int, help='seed, previous default value is 0')
    parser.add_argument('--caption_temp', default=0.1, type=float,
                        help='temperature of caption, -1 :only use caption embedding, 0: only use video features')

    args = parser.parse_args()

    set_seed(args.seed)
    # set_seed(0)

    feature_names = ["uio_caption", "uio_vqa1", "uio_caption_vqa1", "uio_opt_region", "uio_fixed_region", "clip", "i3d",
                     "i3d_clip", "uio_caption_vqa1_1280", "uio_caption_vqa1_68", "uio_caption_34", "uio_vqa1_34",
                     "vqas_170"]
    cal_method = ["mul", "mul-avg", "ss", "cos", "cos-abs", "mse", "mul-abs"]

    if args.feature_name in ["i3d", "uio_caption_vqa1"]:
        args.input_dim = 2048
    elif args.feature_name == "clip":
        args.input_dim = 512
    elif args.feature_name == "i3d_clip":
        args.input_dim = 2560
    elif args.feature_name == "uio_caption_vqa1_68":
        args.input_dim = 68
    elif args.feature_name in ["uio_caption_34", "uio_vqa1_34"]:
        args.input_dim = 34
    elif args.feature_name in ["uio_caption", "uio_vqa1", "uio_opt_region", "uio_fixed_region"]:
        args.input_dim = 1024
    elif args.feature_name == "vqas_170":
        args.input_dim = 170

    if "uio" in args.feature_name:
        args.check_caption_memory = True

    best_auc = 0
    n_iter = 1

    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    optimize_iter = args.optimize_iter
    patient = args.patient

    # if args.input_dim == 512:
    #     print("Using CLIP feature")
    # elif args.input_dim == 2048:
    #     print("Using I3D feature")
    # elif args.input_dim == 2560:
    #     print("Using I3D+CLIP feature")
    # else:
    #     print("Using UIO feature")
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
    # args.input_dim = 768
    if args.FFC:
        model = Learner2(input_dim=args.input_dim, drop_p=args.drop).to(device)
    elif args.tf:
        model = TFLeaner(args.input_dim, args.embedding_dim, args.n_layer, args.n_head, args).to(device)
    elif args.nk:
        # model = Learner(input_dim=args.input_dim, drop_p=args.drop).to(device)
        model = Learner(input_dim=args.input_dim, drop_p=args.drop, memory_rate=args.memory_rate,
                        num_key_memory=args.num_key_memory, max_memory_size=args.max_memory_size,
                        threshold_caption_score=args.threshold_caption_score, nk=args.nk, args=args).to(device)
    else:
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
    torch.set_grad_enabled(True)

    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.w)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50])
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10])
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 5], gamma=0.1)
    criterion = MIL
    aucs = []
    if args.init_memory == "first":
        model.init_memory_space(anomaly_train_dataset, normal_train_dataset, args)

    # 检查保存的异常片段是否合理
    # snippet_idxs = ['765-3', '493-20', '316-10', '766-16', '695-15', '182-2', '774-14', '759-25', '89-9', '276-4',
    #                 '51-10', '177-11', '271-29', '225-10', '334-0', '717-11', '662-28', '207-1', '547-19', '480-26',
    #                 '603-25', '755-6', '320-29', '552-12', '594-8', '769-19', '79-10', '581-0', '778-12', '87-26']
    # snippet_idxs = ['421-26', '403-15', '769-31', '195-19', '347-21', '670-0', '1-27', '94-5', '253-15', '533-3',
    #                 '250-11', '794-13', '289-7', '199-22', '5-18', '293-17', '175-9', '530-22', '147-10', '680-22',
    #                 '746-17', '665-28', '602-5', '59-29', '391-17', '88-14', '215-15', '563-20', '21-8', '106-12']
    # snippet_idxs = ['37-0', '622-23', '69-10', '588-2', '372-5', '228-14', '334-6', '262-15', '391-17', '62-20',
    #                 '709-1', '344-4', '280-19', '549-13', '38-21', '369-17', '682-21', '533-27', '579-4', '576-4',
    #                 '431-14', '660-25', '306-3', '100-3', '97-20', '331-0', '264-13', '726-27', '187-4', '699-1']

    # 0
    # snippet_idxs = ['369-25', '682-17', '588-19', '68-15', '280-31', '344-3', '372-14', '62-17', '228-13', '38-1',
    #                 '37-23', '622-29', '549-9', '709-9', '262-11', '334-8', '533-3', '391-17']
    # 4
    # snippet_idxs = ['369-27', '280-31', '709-10', '622-5', '68-16', '334-2', '372-9', '228-14', '37-13', '262-28', '38-1', '549-6', '682-14', '62-24', '391-1', '533-4', '344-10', '588-29']
    # 7
    # snippet_idxs =  ['709-25', '62-5', '391-21', '37-13', '682-2', '344-14', '588-7', '372-20', '68-16', '262-21', '334-8', '228-19', '38-16', '622-10', '533-18', '369-12', '280-13', '549-2']
    # 11
    # snippet_idxs = ['549-1', '62-23', '334-29', '369-4', '37-21', '372-21', '622-9', '588-25', '709-24', '38-4', '344-24', '228-10', '280-15', '682-31', '262-0', '68-22', '391-25', '533-15']
    # anomaly_train_dataset.check_captions_from_snippet_idxs(snippet_idxs)

    for epoch in range(0, args.epoch):
        model.used_caption_in_inference = []
        # 训练之前初始化一下记忆空间
        if args.init_memory == "epoch":
            model.init_memory_space(anomaly_train_dataset, normal_train_dataset, args)
        # if epoch == 5:
        #     args.nk = True
        #     model.nk = True
        train(epoch)

        # print captions saved in the memory space
        if args.check_caption_memory:
            a_caption_memory, n_caption_memory = model.show_stored_snippet_ids()
            print("a_caption_memory: ")
            for snippet_id in a_caption_memory:
                video_name, caption = anomaly_train_dataset.show_caption(snippet_id)
                print(f"{snippet_id}\t{video_name}\t{caption}")
            # print("n_caption_memory: ")
            # for snippet_id in n_caption_memory:
            #     result = normal_train_dataset.show_caption(snippet_id)
            #     print(snippet_id, result)

        auc, patient = test_abnormal(epoch, patient, args)
        from collections import Counter

        counter = Counter(model.used_caption_in_inference)
        print(counter)
        aucs.append(auc)
        if args.nk:
            pass
            # model.clear_memory(rate=args.clear_rate, epoch=epoch)
            # model.reset_memory(args)
        # print("a_caption_memory:", a_caption_memory)
        # print("n_caption_memory:", n_caption_memory)
        print("best_auc", best_auc)
        if patient == 0:
            print("Easy stop!")
            break
    print(aucs)
    print("best_auc", best_auc)
    plt.plot(aucs)
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.show()
