from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim.lr_scheduler import StepLR
import os
import sys
import json
from scipy.sparse import csr_matrix
import infer_gbi
import torch.nn.functional as F
from statistics import mean

common_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../.."))
sys.path.insert(0, common_path)

from utils import auc_score, to_binary_labels

class SceneGraphTrainer:
    def __init__(self, model, train_data_loader, val_data_loader, n_epochs, lr, device, model_dir, type, test_data_loader=None, topk=5, meta_info_list=None):
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.is_train = True
        if test_data_loader is not None:
            self.val_data_loader = test_data_loader
            self.is_train = False
        self.n_epochs = n_epochs
        self.device = device
        self.model1, self.model2, self.model3, self.model4 = model[0].to(device),model[1].to(device), model[2].to(device), model[3].to(device)
        self.model_dir = model_dir
        self.type = type
        self.topk = topk
        self.meta_info = meta_info_list

        if self.is_train:
            self.optim1 = optim.Adam(self.model1.parameters(), lr=lr)
            self.optim2 = optim.Adam(self.model2.parameters(), lr=lr)
            self.optim3 = optim.Adam(self.model3.parameters(), lr=lr)
            self.optim4 = optim.Adam(self.model4.parameters(), lr=lr)
            self.scheduler1 = StepLR(self.optim1, step_size=15, gamma=0.3)
            self.scheduler2 = StepLR(self.optim2, step_size=15, gamma=0.3)
            self.scheduler3 = StepLR(self.optim3, step_size=15, gamma=0.3)
            self.scheduler4 = StepLR(self.optim4, step_size=15, gamma=0.3)
        self.loss = CrossEntropyLoss()
        if type == 'attribute':
            self.loss = BCEWithLogitsLoss()
        
        self.eval_f = open('sg_test_eval.csv', 'w', buffering=1)

        self.eval_f.write('index,level1,level2,level3,level4,pred_type\n')

        self.eval_f_idx = 0

        concept_path = "/home/ericwallace/alexwan/VQAR-launcher/hossein_data/concepts1.json"
        kb_path = "/home/ericwallace/alexwan/VQAR-launcher/hossein_data/hiereachy1.json"
        with open(concept_path) as f1, open(kb_path) as f2:
            self.concepts = json.load(f1)
            self.kb = json.load(f2)
        self.CONCEPT_NUM = 500
        self.concept_dict = {}
        for id, item in enumerate(self.concepts):
            self.concept_dict[item] = id
        self.adj_matrix = self.build_matirx().cuda()
        print(self.adj_matrix)

    def _pass(self, data, train=True, return_log_probs=False):
        data = [d.to(self.device) for d in data]

        feats, labels, labels1, labels2, labels3, labels4 = data

        # print(feats.shape)

        logits1 = self.model1(feats)
        logits2 = self.model2(feats)
        logits3 = self.model3(feats)
        logits4 = self.model4(feats)
 
        if not len(labels1.shape) == 1:
            labels1 = labels1.float()
        if not len(labels2.shape) == 1:
            labels2 = labels2.float()
        if not len(labels3.shape) == 1:
            labels3 = labels3.float()
        if not len(labels4.shape) == 1:
            labels4 = labels4.float()

        valid_indices1 = torch.nonzero(labels1 != -1).squeeze(1)
        valid_indices2 = torch.nonzero(labels2 != -1).squeeze(1)
        valid_indices3 = torch.nonzero(labels3 != -1).squeeze(1)
        valid_indices4 = torch.nonzero(labels4 != -1).squeeze(1)

        logprobs1 = F.log_softmax(logits1, dim=-1)
        logprobs2 = F.log_softmax(logits2, dim=-1)[:, :-1]
        logprobs3 = F.log_softmax(logits3, dim=-1)[:, :-1]
        logprobs4 = F.log_softmax(logits4, dim=-1)[:, :-1]

        # print(logprobs1.shape)
        # print(logprobs2.shape)
        # print(logprobs3.shape)
        # print(logprobs4.shape)

        # (batch_size, 500)
        logprobs_cat = torch.cat([logprobs1, logprobs2, logprobs3, logprobs4], dim=-1)

        # print(logprobs_cat.shape)

        loss1 = self.loss(logits1[valid_indices1,:], labels1[valid_indices1])
        loss2 = self.loss(logits2[valid_indices2,:], labels2[valid_indices2])
        loss3 = self.loss(logits3[valid_indices3,:], labels3[valid_indices3])
        loss4 = self.loss(logits4[valid_indices4,:], labels4[valid_indices4])

        #binary_labels = to_binary_labels(labels, logits.shape[-1])
        # accuracy = auc_score(binary_labels, logits)
        '''
        if self.type != 'attribute':
            if logits1.shape[-1] < self.topk:
                self.topk = logits1.shape[-1]

            _, pred = logits.topk(self.topk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))

            correct_k = correct[:self.topk].reshape(-1).float().sum(0, keepdim=True)
            accuracy = correct_k.mul_(100.0 / logits.shape[0]).item()

        else:
            if logits.shape[-1] < self.topk:
                self.topk = logits.shape[-1]

            _, pred = logits.topk(self.topk, 1, True, True)
            pred = pred.t()
            correct = torch.sum(labels.gather(1, pred.t()), dim=1)
            correct_label = torch.clamp(torch.sum(labels, dim = 1), 0, self.topk)
            accuracy = torch.mean(correct / correct_label).item()
        '''
        _, predicted1 = torch.max(logits1, 1)
        _, predicted2 = torch.max(logits2, 1)
        _, predicted3 = torch.max(logits3, 1)
        _, predicted4 = torch.max(logits4, 1)
        # if self.type == 'attribute':
        #     labels = labels.max(dim=-1)[1]

        oh1 = torch.zeros((274,))
        oh2 = torch.zeros((157,))
        oh3 = torch.zeros((62,))
        oh4 = torch.zeros((7,))

        oh1[predicted1[0]] = 1
        
        if predicted2[0] != 157:
            oh2[predicted2[0]] = 1
        
        if predicted3[0] != 62:
            oh3[predicted3[0]] = 1
        
        if predicted4[0] != 7:
            oh4[predicted4[0]] = 1

        oh_cat = torch.cat([oh1, oh2, oh3, oh4]).unsqueeze(0).to('cuda')

        # print(predicted1.item(), predicted2.item(), predicted3.item(), predicted4.item())
        # print(labels1.item(), labels2.item(), labels3.item(), labels4.item())
        # print(oh_cat.shape, oh_cat)

        #### check the accuracy between the original label and the predicted label
        """
        new_predict1 = []
        new_predict2 = []
        new_predict3 = []
        new_predict4 = []

        
        total_index = self.meta_info[0]['name']['idx']
        for i1 in predicted1:
            i1 = str(i1.cpu().tolist())
            tmp_label1 = total_index[self.meta_info[1][i1]]
            new_predict1.append(tmp_label1)
        for i2 in predicted2:
            try:
                i2 = str(i2.cpu().tolist())
                tmp_label2 = total_index[self.meta_info[2][i2]]
            except KeyError:
                tmp_label2 = -1
            new_predict2.append(tmp_label2)
        for id3, i3 in enumerate(predicted3):
            if predicted2[id3] == -1:
                tmp_label3 = -1
            else:
                try:
                    i3 = str(i3.cpu().tolist())
                    tmp_label3 = total_index[self.meta_info[3][i3]]
                except KeyError:
                    tmp_label3 = -1
            new_predict3.append(tmp_label3)
        for id4, i4 in enumerate(predicted4):
            if predicted3[id4] == -1:
                tmp_label4 = -1
            else:
                try:
                    i4 = str(i4.cpu().tolist())
                    tmp_label4 = total_index[self.meta_info[4][i4]]
                except KeyError:
                    tmp_label4 = -1
            new_predict4.append(tmp_label4)
        new_resulsts = list(map(list, zip(*[new_predict1, new_predict2, new_predict3, new_predict4])))
        correct = 0
        for id, item in enumerate(new_resulsts):
            if labels[id] in item:
                correct += 1
        """
        correct_count = 0
        count1, count2, count3, count4 = 0,0,0,0

        def l2s(lst):
            return [str(x.item()) if torch.is_tensor(x) else str(x) for x in lst]

        for each_example in range(predicted1.shape[0]):
            self.eval_f.write(','.join(l2s([self.eval_f_idx, labels1[each_example], labels2[each_example], labels3[each_example], labels4[each_example], 'label'])) + '\n')
            self.eval_f.write(','.join(l2s([self.eval_f_idx, predicted1[each_example], predicted2[each_example], predicted3[each_example], predicted4[each_example], 'argmax'])) + '\n')

            self.eval_f_idx += 1

            if predicted1[each_example] == labels1[each_example]:
                correct_count += 1
            count1 += 1
            if predicted2[each_example] == labels2[each_example]:
                correct_count += 1
            count2 += 1
            if predicted2[each_example] == len(self.meta_info[2]):
                continue

            if predicted3[each_example]  == labels3[each_example]:
                correct_count += 1
            count3 += 1
            if predicted3[each_example] == len(self.meta_info[3]):
                continue

            if predicted4[each_example] == labels4[each_example]:
                correct_count += 1
            count4 += 1
            


        # predicted2 = predicted2[torch.where(labels2 == len(self.meta_info[2]))[0]]
        # labels2 = labels2[torch.where(labels2 == len(self.meta_info[2]))[0]]

        # predicted3 = predicted3[torch.where(labels3 == len(self.meta_info[3]))[0]]
        # labels3 = labels3[torch.where(labels3 == len(self.meta_info[3]))[0]]

        # predicted4 = predicted4[torch.where(labels4 == len(self.meta_info[4]))[0]]
        # labels4 = labels4[torch.where(labels4 == len(self.meta_info[4]))[0]]

        correct1 = (predicted1 == labels1).sum().item()
        correct2 = ((predicted2 == labels2) or (predicted2 == 157 and labels2 == -1)).sum().item()
        correct3 = ((predicted3 == labels3) or (predicted3 == 62 and labels3 == -1)).sum().item()
        correct4 = ((predicted4 == labels4) or (predicted4 == 7 and labels4 == -1)).sum().item()
        # correct1 = (predicted1 == labels1).sum().item()
        # correct2 = (predicted2 == labels2).sum().item()
        # correct3 = (predicted3 == labels3).sum().item()
        # correct4 = (predicted4 == labels4).sum().item()
        accuracy1 = correct1 * 100. / len(labels1)
        accuracy2 = correct2 * 100. / len(labels2)
        accuracy3 = correct3 * 100. / len(labels3)
        accuracy4 = correct4 * 100. / len(labels4)
        #accuracy = correct * 100. / len(labels)

        # print(res, accuracy)

        if train:
            self.optim1.zero_grad()
            loss1.backward()
            self.optim1.step()

            self.optim2.zero_grad()
            loss2.backward()
            self.optim2.step()

            self.optim3.zero_grad()
            loss3.backward()
            self.optim3.step()

            self.optim4.zero_grad()
            loss4.backward()
            self.optim4.step()
        
        if return_log_probs:
            return logprobs_cat, oh_cat, (loss1, loss2, loss3, loss4, accuracy1, accuracy2, accuracy3, accuracy4, \
               correct_count, count1+count2+count3+count4)

        return loss1, loss2, loss3, loss4, accuracy1, accuracy2, accuracy3, accuracy4, \
               correct_count, count1+count2+count3+count4
      


    def _train_epoch(self):
        self.model1.train()
        self.model2.train()
        self.model3.train()
        self.model3.train()

        losses1, losses2, losses3, losses4 = [],[],[],[]
        
        acc1, acc2, acc3, acc4 = [], [], [], []
        
        pbar1 = tqdm(self.train_data_loader)
        pbar2 = tqdm(self.train_data_loader)
        pbar3 = tqdm(self.train_data_loader)
        pbar4 = tqdm(self.train_data_loader)

        for ct, data in enumerate(pbar1):

            #loss, accuracy = self._pass(data)
            loss1, loss2, loss3, loss4, accuracy1, accuracy2, accuracy3, accuracy4 =self._pass(data)
            losses1.append(loss1)
            losses2.append(loss2)
            losses3.append(loss3)
            losses4.append(loss4)
            acc1.append(accuracy1)
            acc2.append(accuracy2)
            acc3.append(accuracy3)
            acc4.append(accuracy4)
            pbar1.set_description('[loss1: %f]' % loss1)
            pbar2.set_description('[loss2: %f]' % loss2)
            pbar3.set_description('[loss3: %f]' % loss3)
            pbar4.set_description('[loss4: %f]' % loss4)

        return np.mean(losses1), np.mean(losses2), np.mean(losses3), np.mean(losses4),\
                 np.mean(acc1), np.mean(acc2), np.mean(acc3), np.mean(acc4)

    def _val_epoch(self, limit_samples=10000):
        # self.model1.eval()
        # self.model2.eval()
        # self.model3.eval()
        # self.model4.eval()

        violations_before, violations_after = [], []
        accuracy_before, accuracy_after = [], []
        losses1, losses2, losses3, losses4 = [],[],[],[]
        acc1, acc2, acc3, acc4, acc = [], [], [], [], []
        pbar1 = tqdm(self.val_data_loader, total=min(len(self.val_data_loader), limit_samples))
        total_correct_count = 0
        total_count = 0
        for eval_idx, data in enumerate(pbar1):
            if eval_idx >= limit_samples:
                break

            def _infer():
                return self._pass(data, train=False, return_log_probs=True)
            
            loss1, loss2, loss3, loss4, accuracy1, accuracy2, accuracy3, accuracy4, correct_count, each_total, (v_before, v_after), (acc_before, acc_after) = infer_gbi.run_gbi(
                _infer,
                {'name1': self.model1, 'name2': self.model2, 'name3': self.model3, 'name4': self.model4},
                self.adj_matrix,
                pbar1
            )
            
            accuracy_before.append(acc_before)
            accuracy_after.append(acc_after)
            violations_before.append(v_before.item() == 0)
            violations_after.append(v_after.item() == 0)

            losses1.append(loss1.detach().cpu())
            losses2.append(loss2.detach().cpu())
            losses3.append(loss3.detach().cpu())
            losses4.append(loss4.detach().cpu())
            acc1.append(accuracy1)
            acc2.append(accuracy2)
            acc3.append(accuracy3)
            acc4.append(accuracy4)
            pbar1.set_description('[loss1: %f; loss2: %f; loss3: %f; loss4: %f]' % (loss1, loss2, loss3, loss4))
            total_count += each_total
            total_correct_count += correct_count

        accuracy_before = np.array(accuracy_before)
        accuracy_after = np.array(accuracy_after)

        result = total_correct_count/total_count
        print(result)
        print('Violations:', mean(violations_before), mean(violations_after))
        print('Accuracy:', np.mean(accuracy_before, axis=0), np.mean(accuracy_after, axis=0))
        return np.mean(losses1), np.mean(losses2), np.mean(losses3), np.mean(losses4),\
                 np.mean(acc1), np.mean(acc2), np.mean(acc3), np.mean(acc4)

    def train(self):
        assert self.is_train
        for epoch in range(self.n_epochs):
            train_loss1, train_loss2, train_loss3, train_loss4, train_acc1,  \
                train_acc2, train_acc3,  train_acc4 = self._train_epoch()
            val_loss1, val_loss2, val_loss3, val_loss4, val_acc1,  \
                val_acc2, val_acc3, val_acc4 = self._val_epoch()
            self.scheduler1.step()
            self.scheduler2.step()
            self.scheduler3.step()
            self.scheduler4.step()
            print(
                '[Epoch %d/%d] [training loss1: %.5f, acc1: %.2f] [validation loss1: %.5f, acc1: %.2f] \n \
                               [training loss2: %.5f, acc2: %.2f] [validation loss2: %.5f, acc2: %.2f] \n \
                               [training loss3: %.5f, acc3: %.2f] [validation loss3: %.5f, acc3: %.2f] \n \
                               [training loss4: %.5f, acc4: %.2f] [validation loss4: %.5f, acc4: %.2f] ' %
                (epoch, self.n_epochs, train_loss1, train_acc1, val_loss1, val_acc1, \
                                train_loss2, train_acc2, val_loss2, val_acc2, \
                                train_loss3, train_acc3, val_loss3, val_acc3, \
                                train_loss4, train_acc4, val_loss4, val_acc4,)
            )

            # save model

            if epoch == self.n_epochs-1:
                save_f1 = self.model_dir + '/class1/name_best_epoch.pt'
                print('saving model to %s' % (save_f1))
                torch.save(self.model1.state_dict(), save_f1)

                save_f2 = self.model_dir + '/class2/name_best_epoch.pt'
                print('saving model to %s' % (save_f2))
                torch.save(self.model2.state_dict(), save_f2)

                save_f3 = self.model_dir + '/class3/name_best_epoch.pt'
                print('saving model to %s' % (save_f3))
                torch.save(self.model3.state_dict(), save_f3)

                save_f4 = self.model_dir + '/class4/name_best_epoch.pt'
                print('saving model to %s' % (save_f4))
                torch.save(self.model4.state_dict(), save_f4)



            # if val_acc1 < best_val_loss1:
            #     best_val_loss1 = val_loss1
            #     save_f1 = self.model_dir + '/class1/%s_best_epoch.pt' % self.type
            #     print('saving %s model to %s' % (self.type, save_f1))
            #     torch.save(self.model1.state_dict(), save_f1)

            # # save model2
            # if val_loss2 < best_val_loss2:
            #     best_val_loss2 = val_loss2
            #     save_f2 = self.model_dir + '/class2/%s_best_epoch.pt' % self.type
            #     print('saving %s model to %s' % (self.type, save_f2))
            #     torch.save(self.model2.state_dict(), save_f2)
            
            # # save model3
            # if val_loss3 < best_val_loss3:
            #     best_val_loss3 = val_loss3
            #     save_f3 = self.model_dir + '/class3/%s_best_epoch.pt' % self.type
            #     print('saving %s model to %s' % (self.type, save_f3))
            #     torch.save(self.model3.state_dict(), save_f3)
            # # savle model4
            # if val_loss4 < best_val_loss4:
            #     best_val_loss4 = val_loss4
            #     save_f4 = self.model_dir + '/class4/%s_best_epoch.pt' % self.type
            #     print('saving %s model to %s' % (self.type, save_f4))
            #     torch.save(self.model4.state_dict(), save_f4)

    def test(self):
        assert not self.is_train
        #test_loss, test_acc = self._val_epoch()
        test_loss1, test_loss2, test_loss3, test_loss4, test_acc1,  \
                test_acc2, test_acc3,  test_acc4 = self._val_epoch()
        print('[test loss1: %.2f, acc1: %.2f] \n \
               [test loss2: %.2f, acc2: %.2f]\n \
              [test loss3: %.2f, acc3: %.2f] \n \
               [test loss4: %.2f, acc4: %.2f] \n ' \
        % (test_loss1, test_acc1,test_loss2, test_acc2,test_loss3, test_acc3,test_loss4, test_acc4))

    def build_matirx(self):
        print('build_matrix')
        def adj_format(adj_mat):
        	# adj_mat is V*V tensor, which V is number of image labels.
            E = torch.count_nonzero(torch.tensor(adj_mat))
            V = adj_mat.shape[0]
            # Creating a E * V sparse matrix
            adj_mat_S = csr_matrix(adj_mat)
            EV_mat = torch.from_numpy(csr_matrix((V, E),dtype = np.float32).toarray())
            indices = adj_mat_S.nonzero()
            rows = indices[0]
            cols = indices[1]
            data = adj_mat_S.data
            for i in range(E):
                EV_mat[rows[i]][i]=data[i]
                EV_mat[cols[i]][i]=-data[i]
            return EV_mat

        matrix = np.zeros((self.CONCEPT_NUM, self.CONCEPT_NUM))
        for i in range(len(self.concepts)):
            tmp_c = self.concepts[i]
            child = []
            for key, value in self.kb.items():
                if tmp_c in value:
                    child = value[tmp_c]
                    break
            for j in child:
                matrix[i][self.concept_dict[j]] = 1
        return adj_format(matrix)