import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import mmd
import numpy as np
from sklearn import metrics
from net import DSAN
import time
import utils
from torch.utils.data import TensorDataset, DataLoader
from contrastive_loss import SupConLoss
from config_UP2PC import *
from sklearn import svm
import cleanlab

##################################
data_path_s = './datasets/Pavia/paviaU.mat'
label_path_s = './datasets/Pavia/paviaU_gt_7.mat'
data_path_t = './datasets/Pavia/pavia.mat'
label_path_t = './datasets/Pavia/pavia_gt_7.mat'

data_s,label_s = utils.load_data_pavia(data_path_s,label_path_s)
data_t,label_t = utils.load_data_pavia(data_path_t,label_path_t)
print(data_s.shape,label_s.shape)
print(data_t.shape,label_t.shape)

# Loss Function
crossEntropy = nn.CrossEntropyLoss().cuda()
domain_criterion = nn.BCEWithLogitsLoss().cuda()
ContrastiveLoss_s = SupConLoss(temperature=0.1).cuda()
ContrastiveLoss_t = SupConLoss(temperature=0.1).cuda()
DSH_loss = utils.Domain_Occ_loss().cuda()
criterion_w = utils.Weighted_CrossEntropy

acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None

def get_probs(data_loader, data_loader_t):
    train_features, train_labels = utils.extract_embeddings(feature_encoder, data_loader)
    clt = svm.SVC(probability=True)
    clt.fit(train_features, train_labels)
    test_features, test_labels = utils.extract_embeddings(feature_encoder, data_loader_t)
    probs = clt.predict_proba(test_features)
    return probs

def clean_sampling_epoch(labels, probabilities, output):
    labels = np.array(labels)
    probabilities = np.array(probabilities)
    print("start clean samples")
    print(labels.shape)
    print(probabilities.shape)
    ###################
    # 过滤样本的方法
    ###################
    label_error_mask = np.zeros(len(labels), dtype=bool)
    label_error_indices = cleanlab.latent_estimation.compute_confident_joint(
        labels, probabilities, return_indices_of_off_diagonals=True
    )[1]
    for idx in label_error_indices:
        label_error_mask[idx] = True

    label_errors_bool = cleanlab.pruning.get_noise_indices(labels, probabilities, prune_method='prune_by_class')

    ordered_label_errors = cleanlab.pruning.order_label_errors(
        label_errors_bool=label_errors_bool,
        psx=probabilities,
        labels=labels,
        sorted_index_method='normalized_margin',
    )

    true_labels_idx = []  # 置信样本的索引
    all_labels_idx = []  # 所有标签的索引
    print('len of all_lables', len(labels))
    print('len of errors_lables', len(ordered_label_errors))
    for i in range(len(labels)):
        all_labels_idx.append(i)
    if len(ordered_label_errors) == 0:
        true_labels_idx = all_labels_idx
    else:
        for j in range(len(ordered_label_errors)):
            all_labels_idx.remove(ordered_label_errors[j])
            true_labels_idx = all_labels_idx
    print('len of true_lables', len(true_labels_idx))
    # weights
    orig_class_count = np.bincount(labels, minlength=CLASS_NUM)
    train_bool_mask = ~label_errors_bool
    imgs = [labels[i] for i in range(len(labels)) if train_bool_mask[i]]
    clean_class_counts = np.bincount(imgs, minlength=CLASS_NUM)
    print(orig_class_count)
    print(clean_class_counts)
    class_weights = torch.Tensor(orig_class_count / clean_class_counts).cuda()

    print(class_weights)
    # 获取目标域中对应的置信样本和伪标签
    target_labels = []
    target_datas = []

    for i in range(len(true_labels_idx)):
        if output[true_labels_idx[i]] >= 0.8:
            target_datas.append(testX[true_labels_idx[i]])
            target_labels.append(labels[true_labels_idx[i]])

    target_datas = np.array(target_datas)
    target_labels = np.array(target_labels)
    ####################
    # 输出所挑选的置信样本中真正正确的样本所占的比重
    ######################
    right_score = 0

    for i in range(len(true_labels_idx)):
        # testY true label
        if testY[true_labels_idx[i]] == labels[true_labels_idx[i]]:
            right_score += 1
    clean_accuracy = right_score / len(true_labels_idx)
    print('clean samples finished')
    return target_datas, target_labels, class_weights, clean_accuracy

for iDataSet in range(nDataSet):
    print('#######################idataset######################## ', iDataSet)
    utils.set_seed(seeds[iDataSet])

    trainX, trainY = utils.get_sample_data(data_s, label_s, HalfWidth, 180)
    testID, testX, testY, G, RandPerm, Row, Column = utils.get_all_data(data_t, label_t, HalfWidth)

    train_dataset = TensorDataset(torch.tensor(trainX), torch.tensor(trainY))
    test_dataset = TensorDataset(torch.tensor(testX), torch.tensor(testY))

    train_loader_s = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    train_loader_t = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
    test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False,drop_last=False)

    len_source_loader = len(train_loader_s)
    len_target_loader = len(train_loader_t)

    # model
    feature_encoder = DSAN(nBand, patch_size, CLASS_NUM).cuda()

    print("Training...")

    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    total_hit, total_num = 0.0, 0.0
    size = 0.0
    test_acc_list = []


    train_start = time.time()

    #loss plot
    loss1 = []
    loss2 = []
    loss3 = []


    for epoch in range(1, epochs + 1):
        LEARNING_RATE = lr #/ math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
        print('learning rate{: .4f}'.format(LEARNING_RATE))
        optimizer = torch.optim.SGD([
            {'params': feature_encoder.feature_layers.parameters(),},
            {'params': feature_encoder.fc1.parameters(), 'lr': LEARNING_RATE},
            {'params': feature_encoder.fc2.parameters(), 'lr': LEARNING_RATE},
            {'params': feature_encoder.head1.parameters(), 'lr': LEARNING_RATE},
            {'params': feature_encoder.head2.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE , momentum=momentum, weight_decay=l2_decay)

        feature_encoder.train()

        if (epoch >= train_num and epoch < epochs) and epoch % 20 == 0:
            print('get  fake label,ep = ', epoch)
            fake_label, output = utils.obtain_label(test_loader, feature_encoder)

            print('get probs,ep=', epoch)
            probs = get_probs(train_loader_s, test_loader)

            clean_datas, clean_labels, class_weights, clean_acc = clean_sampling_epoch(fake_label, probs, output)

            clean_datasets = TensorDataset(torch.tensor(clean_datas), torch.tensor(clean_labels))
            clean_loader = DataLoader(clean_datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,drop_last=True)

        iter_source = iter(train_loader_s)
        iter_target = iter(train_loader_t)
        if epoch >= train_num:
            iter_clean = iter(clean_loader)
            len_clean_loader = len(iter_clean)
        num_iter = len_source_loader

        for i in range(num_iter):
            source_data, source_label = iter_source.next()
            target_data, target_label = iter_target.next()

            if (i+1) % len_target_loader == 0 and (i+1) != num_iter:
                iter_target = iter(train_loader_t)

            # 0
            source_data0 = utils.radiation_noise(source_data)
            source_data0 = source_data0.type(torch.FloatTensor)
            # 1
            source_data1 = utils.flip_augmentation(source_data)
            # 2
            target_data0 = utils.radiation_noise(target_data)
            target_data0 = target_data0.type(torch.FloatTensor)
            # 3
            target_data1 = utils.flip_augmentation(target_data)

            source_features, source1, _, source_outputs, source_out= feature_encoder(source_data.cuda())
            _, source2, _, _ ,_ = feature_encoder(source_data0.cuda())
            _, source3, _, _ ,_= feature_encoder(source_data1.cuda())
            target_features, _, target1, target_outputs, target_out = feature_encoder(target_data.cuda())
            _, _, target2, t1, _= feature_encoder(target_data0.cuda())
            _, _, target3, t2, _= feature_encoder(target_data1.cuda())

            softmax_output_t = nn.Softmax(dim=1)(target_outputs).detach()
            _, pseudo_label_t = torch.max(softmax_output_t, 1)

            # Supervised Contrastive Loss
            all_source_con_features = torch.cat([source2.unsqueeze(1), source3.unsqueeze(1)],dim=1)
            all_target_con_features = torch.cat([target2.unsqueeze(1), target3.unsqueeze(1)], dim=1)

            # Loss Cls
            cls_loss = crossEntropy(source_outputs, source_label.cuda())
            # Loss Lmmd
            lmmd_loss = mmd.lmmd(source_features, target_features, source_label,
                                 torch.nn.functional.softmax(target_outputs, dim=1), BATCH_SIZE=BATCH_SIZE,
                                 CLASS_NUM=CLASS_NUM)
            lambd = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1
            # Loss Con_s
            contrastive_loss_s = ContrastiveLoss_s(all_source_con_features, source_label)
            # Loss Con_t
            contrastive_loss_t = ContrastiveLoss_t(all_target_con_features, pseudo_label_t)
            # Loss Occ
            domain_similar_loss = DSH_loss(source_out, target_out)

            loss = cls_loss + 0.3 * lambd * lmmd_loss + contrastive_loss_s + contrastive_loss_t + domain_similar_loss

            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = source_outputs.data.max(1)[1]
            total_hit += pred.eq(source_label.data.cuda()).sum()
            size += source_label.data.size()[0]
            test_accuracy = 100. * float(total_hit) / size

            if epoch >= train_num:
                if i+1 % len_clean_loader == 0:
                    iter_clean = iter(clean_loader)
                clean_data, clean_label = iter_clean.next()
                clean_features, _, _, clean_outputs, _ = feature_encoder(clean_data.cuda())
                target_cls_loss = crossEntropy(clean_outputs, clean_label.cuda())
                optimizer.zero_grad()
                target_cls_loss.backward()
                optimizer.step()

        print('epoch {:>3d}:   cls loss: {:6.4f},lmmd loss:{:6f}, occ loss:{:6f} con_s loss:{:6f}, con_t loss:{:6f},acc {:6.4f}, total loss: {:6.4f}'
              .format(epoch , cls_loss.item(),lmmd_loss.item(),domain_similar_loss.item(),contrastive_loss_s.item(),contrastive_loss_t.item(),
               total_hit / size,loss.item()))

        train_end = time.time()
        if epoch % epochs == 0:
            print("Testing ...")
            feature_encoder.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)
            with torch.no_grad():
                for test_datas, test_labels in test_loader:
                    batch_size = test_labels.shape[0]

                    test_features, _, _, test_outputs, _ = feature_encoder(Variable(test_datas).cuda())

                    pred = test_outputs.data.max(1)[1]

                    test_labels = test_labels.numpy()
                    rewards = [1 if pred[j] == test_labels[j] else 0 for j in range(batch_size)]

                    total_rewards += np.sum(rewards)
                    counter += batch_size

                    predict = np.append(predict, pred.cpu().numpy())
                    labels = np.append(labels, test_labels)

                    accuracy = total_rewards / 1.0 / counter  #
                    accuracies.append(accuracy)

            test_accuracy = 100. * total_rewards / len(test_loader.dataset)
            acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
            OA = acc
            C = metrics.confusion_matrix(labels, predict)
            A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float)

            k[iDataSet] = metrics.cohen_kappa_score(labels, predict)
            print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format(total_rewards, len(test_loader.dataset),
                                                           100. * total_rewards / len(test_loader.dataset)))
            test_end = time.time()

            # Training mode

            if test_accuracy > last_accuracy:
                # save networks
                # torch.save(feature_encoder.state_dict(),str("../checkpoints/DFSL_feature_encoder_" + "houston_cl_lmmd_dis_attention" +str(iDataSet) +".pkl"))
                print("save networks for epoch:", epoch + 1)
                last_accuracy = test_accuracy
                best_episdoe = epoch
                best_predict_all = predict
                best_G, best_RandPerm, best_Row, best_Column = G, RandPerm, Row, Column
                print('best epoch:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))

            print('iter:{} best epoch:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))
            print('***********************************************************************************')

AA = np.mean(A, 1)
AAMean = np.mean(AA,0)
AAStd = np.std(AA)
AMean = np.mean(A, 0)
AStd = np.std(A, 0)
OAMean = np.mean(acc)
OAStd = np.std(acc)
kMean = np.mean(k)
kStd = np.std(k)
print ("train time per DataSet(s): " + "{:.5f}".format(train_end-train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end-train_end))
print ("average OA: " + "{:.2f}".format( OAMean) + " +- " + "{:.2f}".format( OAStd))
print ("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print ("average kappa: " + "{:.4f}".format(100 *kMean) + " +- " + "{:.4f}".format(100 *kStd))
print ("accuracy for each class: ")
for i in range(CLASS_NUM):
    print ("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))


best_iDataset = 0
for i in range(len(acc)):
    print('{}:{}'.format(i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print('best acc all={}'.format(acc[best_iDataset]))

#################classification map################################

for i in range(len(best_predict_all)):  # predict ndarray <class 'tuple'>: (9729,)
    best_G[best_Row[best_RandPerm[ i]]][best_Column[best_RandPerm[ i]]] = best_predict_all[i] + 1

hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
for i in range(best_G.shape[0]):
    for j in range(best_G.shape[1]):
        if best_G[i][j] == 0:
            hsi_pic[i, j, :] = [0, 0, 0]
        if best_G[i][j] == 1:
            hsi_pic[i, j, :] = [0, 0, 1]
        if best_G[i][j] == 2:
            hsi_pic[i, j, :] = [0, 1, 0]
        if best_G[i][j] == 3:
            hsi_pic[i, j, :] = [0, 1, 1]
        if best_G[i][j] == 4:
            hsi_pic[i, j, :] = [1, 0, 0]
        if best_G[i][j] == 5:
            hsi_pic[i, j, :] = [1, 0, 1]
        if best_G[i][j] == 6:
            hsi_pic[i, j, :] = [1, 1, 0]
        if best_G[i][j] == 7:
            hsi_pic[i, j, :] = [0.5, 0.5, 1]


# utils.classification_map(hsi_pic[4:-4, 4:-4, :], best_G[4:-4, 4:-4], 24,  "classificationMap/housotn18.png")
