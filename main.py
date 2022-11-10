from torch.autograd import Variable
from torch.utils.data import DataLoader
from learner import Learner
from loss import *
from dataset import *
import os
from sklearn import metrics
import argparse
from FFC import *


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
        print(loss)
        print("len(a_memory)", len(model.a_memory))
        print("len(n_memory)", len(model.n_memory))

        optimizer.step()
        train_loss += loss.item()
        n_iter +=1
        if n_iter % optimize_iter == 0 :
            model.optimize_memory()
    # print(model.a_memory)
    # print(model.n_memory)
    print('loss = {}'.format(train_loss / len(normal_train_loader)))
    scheduler.step()


def test_abnormal(epoch):
    model.eval()
    global best_auc
    auc = 0
    with torch.no_grad():
        for i, (data, data2) in enumerate(zip(anomaly_test_loader, normal_test_loader)):
            inputs, gts, frames = data
            inputs = inputs.view(-1, inputs.size(-1)).to(torch.device('cuda'))
            score = model(inputs)
            score = score.cpu().detach().numpy()
            score_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, frames[0] // 16, 33))

            for j in range(32):
                score_list[int(step[j]) * 16:(int(step[j + 1])) * 16] = score[j]

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
            step2 = np.round(np.linspace(0, frames2[0] // 16, 33))
            for kk in range(32):
                score_list2[int(step2[kk]) * 16:(int(step2[kk + 1])) * 16] = score2[kk]
            gt_list2 = np.zeros(frames2[0])
            score_list3 = np.concatenate((score_list, score_list2), axis=0)
            gt_list3 = np.concatenate((gt_list, gt_list2), axis=0)

            fpr, tpr, thresholds = metrics.roc_curve(gt_list3, score_list3, pos_label=1)
            auc += metrics.auc(fpr, tpr)

        print('auc = {}', auc / 140)

        if best_auc < auc / 140:
            # ==========
            print("No Saving")
            best_auc = auc/140
            # ==========

            # print('Saving..')
            # state = {
            #     'net': model.state_dict(),
            # }
            # if not os.path.isdir('checkpoint'):
            #     os.mkdir('checkpoint')
            # torch.save(state, './checkpoint/ckpt.pth')
            # best_auc = auc / 140
    return auc / 140

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
    parser = argparse.ArgumentParser(description='PyTorch MIL Training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--w', default=0.0010000000474974513, type=float, help='weight_decay')
    parser.add_argument('--modality', default='TWO', type=str, help='modality')
    parser.add_argument('--input_dim', default=2048, type=int, help='input_dim')
    parser.add_argument('--drop', default=0.6, type=float, help='dropout_rate')
    parser.add_argument('--FFC', '-r', action='store_true', help='FFC')
    args = parser.parse_args()

    set_seed(0)
    best_auc = 0
    train_batch_size = 30
    test_batch_size = 1
    n_iter = 0
    optimize_iter = 10
    args.lr = 0.01
    normal_train_dataset = Normal_Loader(is_train=1, modality=args.modality)
    normal_test_dataset = Normal_Loader(is_train=0, modality=args.modality)

    anomaly_train_dataset = Anomaly_Loader(is_train=1, modality=args.modality)
    anomaly_test_dataset = Anomaly_Loader(is_train=0, modality=args.modality)

    normal_train_loader = DataLoader(normal_train_dataset, batch_size=train_batch_size, shuffle=True)
    normal_test_loader = DataLoader(normal_test_dataset, batch_size=test_batch_size, shuffle=False)

    anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=train_batch_size, shuffle=True)
    anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=test_batch_size, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.FFC:
        model = Learner2(input_dim=args.input_dim, drop_p=args.drop).to(device)
    else:
        model = Learner(input_dim=args.input_dim, drop_p=args.drop).to(device)


    print('Loading..')
    state = {
        'net': model.state_dict(),
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    model_save_path = r"./checkpoint/ckpt.pth"
    assert os.path.exists(model_save_path)
    state = torch.load(model_save_path)
    model.load_state_dict(state['net'])

    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.w)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50])
    criterion = MIL
    aucs = []
    for epoch in range(0, 80):
        train(epoch)
        auc = test_abnormal(epoch)
        aucs.append(auc)
        # model.optimize_memory()

    print(aucs)
    print("best_auc", best_auc)

# 0.8156065926627678
# [0.8117029463709068, 0.8130345234237035, 0.8171689556235578, 0.8172385604588104, 0.8187716316690391, 0.8201633459461355, 0.8195225584917958, 0.8196659159927594, 0.8204130430919842, 0.818847364200364, 0.8193324552236797, 0.8213235591228235, 0.8206593569440848, 0.8193816601474595, 0.82084027572169, 0.8218115295956012, 0.8214297981658624, 0.8188759037541987, 0.8199527431660002, 0.819894472255975]
# best_auc 0.8218115295956012
# [0.8033265455517071, 0.8097801997718168, 0.8130846698926032, 0.8141432599697392, 0.8143548727728809, 0.8166694247764738, 0.8185075263910628, 0.8194151814679275, 0.8208107651092229, 0.8202154635688103, 0.8214318450019817, 0.8212536690548183, 0.8208324027745479, 0.8205616991375815, 0.8211732097840999, 0.8202472499508425, 0.8208943783944282, 0.8207565441370577, 0.8212233241806033, 0.8219510328874924, 0.8222229738802699, 0.8217298191774914, 0.8219591542201881, 0.8215491175912784, 0.8229374515565978, 0.8230656609573972, 0.8227225513223081, 0.8227581473353478, 0.8229432548618077, 0.8227148432581423, 0.8226185326680103, 0.82263037559405, 0.8226308967328987, 0.8226337189268107, 0.8226900240075642, 0.8225229032782149, 0.8225622805757851, 0.8225779723287703, 0.8225482022545768, 0.8227004351374729, 0.8224747107150944, 0.8224585401076305, 0.8224448622953057, 0.8225281825988858, 0.8224104762063952, 0.8224821241357495, 0.8223163761509933, 0.8225394763775631, 0.8223524270410597, 0.822399671902933, 0.8223343563607554, 0.8223123932165562, 0.8222881001705565, 0.8223565613327496, 0.8223331300753839, 0.8222535670450929, 0.8223157533198472, 0.8222229766042635, 0.8223207827376745, 0.8223370424201842, 0.8223370424201842, 0.8223309958535693, 0.8223040403461129, 0.8222508801183419, 0.8221988127681634, 0.8222127473271209, 0.8222690592384558, 0.822193716729693, 0.8221697849248158, 0.8221728241177012, 0.8221665425257103, 0.8221230660086284, 0.822129193723613, 0.8220949745116529, 0.8221205370216955, 0.8220872101108577, 0.8220929248980651, 0.8221216071998362, 0.8221281456669158, 0.8221539678938298]
# best_auc 0.8230656609573972

# Epoch: 74
# loss = 0.4205668259550024
# auc = {} 0.829994200302661

# Epoch: 74
# loss = 0.4456628836967327
# auc = {} 0.8278700375003192

# Abuse 1
# Arrest 2
# Arson 3
# Assault 4
# Burglary 5
# Explosion 6
# Fighting 7
# Normal_Videos_event 8
# RoadAccidents 9
# Robbery 10
# Shooting 11
# Shoplifting 12
# Stealing 13
# Vandalism 14
