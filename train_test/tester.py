import os
import torch
from collections import Counter
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, roc_curve
from transformers import ElectraConfig
from tqdm import tqdm
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import gc
import time
import sys


from data.dataset import HitAnomalyDataset, LogBertDataset, ElectraDataset, DeepLogDataset, LogAnomalyDataset, PLELogDataset
from models.electra import ElectraForLanguageModelingModel
from models.HitAnomaly import HitAnomaly
from models.bert_pytorch.model import BERT
from models.lstm import DeepLog, LogAnomaly
# from models.PLELog import PLELog
from data.vocab import WordVocab

class Tester:
    def __init__(self, options):
        self.options = options
        self.__dict__.update(options)
        self.parallel = False
        if self.device:
            if 'cuda' in self.device:
                self.device = torch.device(self.device) if torch.cuda.is_available() else torch.device('cpu')
            else:
                self.device = torch.device(self.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.device.type == 'cuda':
            self.parallel = options.get('parallel', False)
        
        print("Using device {}".format(self.device))
    
    def test(self):
        pass






class HitAnomalyTester(Tester):
    def __init__(self, options):
        self.hitAnomaly = HitAnomaly(options['embed_dim'], options['hidden_dim'], options['dff'], options['heads'])
        test_dataset = HitAnomalyDataset(options['dataset'], options['data_dir'], 'test')
        self.test_loader = DataLoader(test_dataset, batch_size= 1, num_workers=4)
        del test_dataset
        super().__init__(options)
        self.hitAnomaly.eval()
        self.hitAnomaly.to(self.device)
        self.hitAnomaly.load_state_dict(torch.load(os.path.join(self.save_dir, 'best_model.pt')))


    def test(self):
        print("Testing...")
        y_true = []
        y_pred = []
        with torch.no_grad():
            for data in tqdm(self.test_loader):
                x, y = data['EventTemplate'], data['ParameterList']
                label = data['Label']
                x = x.squeeze().to(self.device)
                y = y.squeeze().to(self.device)
                output = self.model(x, y)
                output = torch.sigmoid(output)
                y_true.append(label.cpu().numpy())
                y_pred.append(output.cpu().numpy())
        pred = np.concatenate(y_pred)
        gt = np.concatenate(y_true)
        auroc = roc_auc_score(gt, pred)
        pred = np.where(pred > 0.5, 1, 0)
        print("Accuracy: ", accuracy_score(gt, pred))
        print("Precision: ", precision_score(gt, pred))
        print("Recall: ", recall_score(gt, pred))
        print("F1: ", f1_score(gt, pred))
        print("ROC AUC: ", auroc)
        print("Testing Done!")





class LogBertTester(Tester):
    def __init__(self, options):
        super().__init__(options)
        self.lower_bound = self.gaussian_mean - 3 * self.gaussian_std
        self.upper_bound = self.gaussian_mean + 3 * self.gaussian_std

        self.center = None
        self.radius = None

        dataset = LogBertDataset(options['dataset'], options['data_dir'], mode = 'test', mask_ratio=options['mask_ratio'])
        self.test_loader = DataLoader(dataset, batch_size = options['batch_size'], num_workers=4, collate_fn=dataset.collate_fn, shuffle = False)
        self.normal_len = dataset.normal_len
        self.anomaly_len = dataset.anomaly_len
        assert self.normal_len + self.anomaly_len == len(dataset)
        del dataset
        gc.collect()
        
    def predict(self):
        model = torch.load(self.model_path)
        model.to(self.device)
        model.eval()

        start_time = time.time()
        
        scale = None
        error_dict = None

        if self.hypersphere_loss:
            center_dict = torch.load(self.model_dir + "best_center.pt")
            self.center = center_dict["center"]
            self.radius = center_dict["radius"]
        
        output_cls = []
        total_dist = []
        total_results = []
       
        for idx, data in enumerate(self.test_loader):
            data = {key: value.to(self.device) for key, value in data.items()}
            result = model.forward(data["bert_input"], data["time_input"])

            mask_lm_output = result['logkey_output']
            output_cls += result['cls_output'].tolist()

            
            for i in range(len(data['bert_label'])):
                seq_results = {"num_error": 0, 
                               "undetected_tokens": 0,
                               "masked_tokens": 0,
                               "total_logkey": torch.sum(data["bert_input"][i] > 0).item(),
                               "deepSVDD_label": 0
                               }
                
                mask_index = data['bert_label'][i] > 0
                num_masked = torch.sum(mask_index).tolist()
                seq_results['masked_tokens'] = num_masked

                if self.is_logkey:
                    num_undetected = self.detect_logkey_anomaly(
                        mask_lm_output[i][mask_index], data["bert_label"][i][mask_index]
                    )
                    seq_results["undetected_tokens"] = num_undetected



                if self.hypersphere_loss_test:
                    assert result['cls_output'][i].size() == self.center.size()

                    dist = torch.sqrt(torch.sum((result['cls_output'][i] - self.center) ** 2))
                    total_dist.append(dist.item())

                    seq_results['deepSVDD_label'] = int(dist.item() > self.radius)
                

                # if idx < 10 or idx % 1000 == 0:
                #     print(
                #         "#time anomaly: {} # of undetected_tokens: {}, # of masked_tokens: {} , "
                #         "# of total logkey {}, deepSVDD_label: {} \n".format(
                #             seq_results["num_error"],
                #             seq_results["undetected_tokens"],
                #             seq_results["masked_tokens"],
                #             seq_results["total_logkey"],
                #             seq_results['deepSVDD_label']
                #         )
                #     )

                total_results.append(seq_results)

        inference_time = time.time() - start_time
        with open('./result.txt', 'a+') as f:
            f.write(f'LogBERT inference time on {self.dataset}: {inference_time}\n')
        params = {"is_logkey": self.is_logkey, "is_time": self.is_time, "hypersphere_loss": self.hypersphere_loss,
                  "hypersphere_loss_test": self.hypersphere_loss_test}
        
        
        best_result = self.find_best_threshold(total_results, params, np.arange(0, 1, 0.1))
        print("Best threshold: ", best_result[1])
        print("TP: ", best_result[3])
        print("FP: ", best_result[2])
        print("TN: ", best_result[4])
        print("FN: ", best_result[5])
        print(f'F1-measure: {best_result[-1]:.4f}')
        print(f'Precision: {best_result[-3]:.4f}')
        print(f'Recall: {best_result[-2]:.4f}')
        elapsed_time = time.time() - start_time
        print("elapsed_time: ", elapsed_time)
        
        with open('./logbert_test.txt', 'a+') as f:
            f.write(f'Testing on {self.dataset} F1: {best_result[-1]:.4f} Precision: {best_result[-3]:.4f} Recall: {best_result[-2]:.4f} Time: {elapsed_time}\n\n')





    def detect_logkey_anomaly(self, masked_output, masked_label):
        num_undetected_tokens = 0
        for i, token in enumerate(masked_label):
            if token not in torch.argsort(-masked_output[i])[:self.num_candidates]:
                num_undetected_tokens += 1
        
        return num_undetected_tokens
    

    def compute_anomaly(self, results, params, seq_threshold = 0.5):
        FP = 0
        TP = 0
        is_logkey = params["is_logkey"]
        is_time = params["is_time"]
        num_error = 0
        for i, seq_res in enumerate(results):
            if (is_logkey and seq_res["undetected_tokens"] > seq_res["masked_tokens"] * seq_threshold) or \
                (is_time and seq_res["num_error"]> seq_res["masked_tokens"] * seq_threshold) or \
                (params["hypersphere_loss_test"] and seq_res["deepSVDD_label"]):

                if i < self.normal_len:
                    FP += 1
                else:
                    TP += 1
        
        TN = self.normal_len - FP
        FN = self.anomaly_len - TP
        try:
            P = 100 * TP / (TP + FP)
        except ZeroDivisionError:
            P = 0
            
        try:
            R = 100 * TP / (TP + FN)
        except ZeroDivisionError:
            R = 0

        try:
            F1 = 2 * P * R / (P + R)
        except ZeroDivisionError:
            F1 = 0

        result = [0, seq_threshold, FP, TP, TN, FN, P, R, F1]
        return result


    def find_best_threshold(self, results, params, seq_range):
        best_result = [0] * 9
        for seq_th in seq_range:
            result = self.compute_anomaly(results, params, seq_th)
            
            if result[-1] > best_result[-1]:
                best_result = result
        
        return best_result
    


class ElectraTester(Tester):
    def __init__(self, model_configs, options):
        super().__init__(options)
        test_dataset = ElectraDataset(options['dataset'], options['data_dir'], mode = 'test', seq_len = options['seq_len'], mask_num = options['mask_num'], mask_ratio=options['mask_ratio'])
        self.test_loader = DataLoader(test_dataset, batch_size = options['batch_size'], num_workers=4, collate_fn=test_dataset.collate_fn, shuffle = False)
        vocab_size = len(test_dataset.vocab)
        model_configs = ElectraConfig(vocab_size = vocab_size, **model_configs)
        model = ElectraForLanguageModelingModel(model_configs, test_dataset.vocab, options['mask_num'], random_generator = options['random_generator'])
        model.tie_generator_and_discriminator_embeddings()
        model.to(self.device)
        model.load_state_dict(torch.load(self.model_path))
        self.model = model.discriminator_model
        self.model.eval()

        self.thre = torch.load(self.model_dir + "best_thre.pt").numpy()


    def find_Optimal_Cutoff(self, target, predicted):
        """ Find the optimal probability cutoff point for a classification model related to event rate
        Parameters
        ----------
        target : Matrix with dependent or target data, where rows are observations

        predicted : Matrix with predicted data, where rows are observations

        Returns
        -------     
        list type, with optimal cutoff value
            
        """
        fpr, tpr, threshold = roc_curve(target, predicted)
        i = np.arange(len(tpr)) 
        # roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
        # roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
        roc_t = sorted(list(zip(np.abs(tpr - (1 - fpr)), threshold)), key=lambda i: i[0], reverse=False)[0][1]

        precision, recall, threshold = precision_recall_curve(target, predicted)
        i = np.arange(len(recall))
        pr_t = sorted(list(zip(np.abs(precision - recall), threshold)), key=lambda i: i[0], reverse=False)[0][1]

        return roc_t, pr_t
    
    def displot(self, gt, norm):
        label = ['normal  '] * len(gt)
        label = np.array(label)
        label[gt == 1] = 'abnormal'
        data = pd.DataFrame({'label': label, 'output norm': norm})
        plt.figure(figsize = (10, 10))
        sns.set(font_scale=2, style = 'white')
        fig = sns.displot(data, x = 'output norm', hue = 'label', fill = True, palette = 'pastel', height = 5, aspect = 2, multiple = 'stack', stat = 'probability', bins = 40, common_norm = False)
        fig.set(xlim = (norm.min(), norm.max()), title = f'\n{self.dataset}')
        # remove legend
        fig._legend.remove()
        # plt.tight_layout()
        fig.figure.savefig(os.path.join(self.model_dir, "distplot.png"), bbox_inches="tight", dpi = 900)
        return fig
    
    def test(self):
        start = time.time()
        print("Testing...")
        self.model.eval()
        gt = []
        pred = []
        norm = []
        # token_labels = []
        # last_attentions = []
        with torch.no_grad():
            for data in tqdm(self.test_loader):
                log_seq = data['k'].to(self.device)
                attn_mask = data['attention_mask'].to(self.device)
                label = data['gt']
                seq_label = torch.max(label, dim = -1)[0]
                output = self.model(log_seq, attention_mask = attn_mask)
                rmd_out = output[0]
                rmd_loss = torch.linalg.vector_norm(rmd_out, dim = -1)
                num_seq = data['num_seq']
                agg_loss = []
                i = 0
                for j in num_seq:
                    agg_loss.append(rmd_loss[i : j].max().item())
                    i = j
                rmd_loss = torch.tensor(agg_loss)
                # rmd_loss = torch.sigmoid(rmd_output)
                # last_attention = output[-1].mean(dim = 1)[:, 0, :].cpu().numpy()
                # last_attention = np.split(last_attention, last_attention.shape[0], axis = 0)
                # label = label.cpu().numpy()
                # label = np.split(label, label.shape[0], axis = 0)

                rmd_loss = rmd_loss.cpu().numpy()
                seq_label = seq_label.cpu().numpy()
                pred.append(rmd_loss)  
                gt.append(seq_label)
                norm.append(rmd_loss)
                # token_labels.extend(label)
                # last_attentions.extend(last_attention)

        end = time.time()
                
        gt = np.concatenate(gt)
        pred = np.concatenate(pred)
        norm = np.concatenate(norm)

        print("plot displot...")
        fig = self.displot(gt, norm)

        roc_auc = roc_auc_score(gt, pred)
        print('AUROC: {:.4f}'.format(roc_auc))
        precision, recall, thresholds = precision_recall_curve(gt, pred)
        pr_auc = auc(recall, precision)
        print('AUPR: {:.4f}'.format(pr_auc))
        # loss_normal = pred[gt == 0]
        # loss_anomaly = pred[gt == 1]
        # print('Average loss normal: {:.4f}'.format(loss_normal.mean()))
        # print('Average loss anomaly: {:.4f}'.format(loss_anomaly.mean()))
        # print('Max loss normal: {:.4f}'.format(loss_normal.max()))
        # print('Max loss anomaly: {:.4f}'.format(loss_anomaly.max()))


        roc_thre, pr_thre = self.find_Optimal_Cutoff(gt, pred)
        print('ROC threshold: {:.6f}'.format(roc_thre))
        roc_pred = np.where(pred > roc_thre, 1, 0)
        accuracy = accuracy_score(gt, roc_pred)
        precision = precision_score(gt, roc_pred)
        recall = recall_score(gt, roc_pred)
        f1 = f1_score(gt, roc_pred)
        print('Accuracy: {:.4f}'.format(accuracy))
        print('Precision: {:.4f}'.format(precision))
        print('Recall: {:.4f}'.format(recall))
        print('F1: {:.4f}'.format(f1))

        print('PR threshold: {:.6f}'.format(pr_thre))
        pr_pred = np.where(pred > pr_thre, 1, 0)
        accuracy = accuracy_score(gt, pr_pred)
        precision = precision_score(gt, pr_pred)
        recall = recall_score(gt, pr_pred)
        f1 = f1_score(gt, pr_pred)
        print('Accuracy: {:.4f}'.format(accuracy))
        print('Precision: {:.4f}'.format(precision))
        print('Recall: {:.4f}'.format(recall))
        print('F1: {:.4f}'.format(f1))
        
        print('Threshold: {:.6f}'.format(self.thre))
        my_pred = np.where(pred > self.thre, 1, 0)
        accuracy = accuracy_score(gt, my_pred)
        precision = precision_score(gt, my_pred)
        recall = recall_score(gt, my_pred)
        f1 = f1_score(gt, my_pred)
        print('Accuracy: {:.4f}'.format(accuracy))
        print('Precision: {:.4f}'.format(precision))
        print('Recall: {:.4f}'.format(recall))
        print('F1: {:.4f}'.format(f1))
        with open(os.path.join(self.model_dir, "test_result.txt"), 'w+') as f:
            f.write('Threshold: {:.6f}\n'.format(self.thre))
            f.write('Accuracy: {:.4f}\n'.format(accuracy))
            f.write('Precision: {:.4f}\n'.format(precision))
            f.write('Recall: {:.4f}\n'.format(recall))
            f.write('F1: {:.4f}\n'.format(f1))
        
        with open('./result2.txt', 'a+') as f:
            f.write(f'Testing roc_auc: {roc_auc:.4f} pr_auc: {pr_auc:.4f} accuracy: {accuracy:.4f} precision: {precision:.4f} recall: {recall:.4f} f1: {f1:.4f} time: {end - start}\n\n')

        return {'roc_auc': roc_auc, 'pr_auc': pr_auc, 'thre': self.thre, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
        

                


class DeepLogTester(Tester):
    def __init__(self, options):
        super().__init__(options)
        self.test_dataset = DeepLogDataset(options['dataset'], options['data_dir'], mode = 'test')
        self.test_loader = DataLoader(self.test_dataset, batch_size = 128, num_workers=4, collate_fn=self.test_dataset.collate_fn, shuffle = False)
        vocab_size = len(self.test_dataset.vocab)
        self.model = DeepLog(options['hidden_size'], options['num_layers'], vocab_size)
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        self.window_size = self.test_dataset.window_size

    
    def predict_by_candidates(self, next_log, output, num_candidates):
        pred = []
        for i in range(len(output)):
            candidates = np.argsort(-output[i])[:num_candidates]
            if next_log[i] in candidates:
                pred.append(0)
            else:
                pred.append(1)
        
        return np.array(pred)

    
    def predict_sample(self, output, next_log):
        if torch.is_tensor(next_log):
            next_log = next_log.numpy()
        
        for i in range(len(output)):
            if next_log[i] not in output[i]:
                return 1

        return 0

    
    def test(self):
        gt_list = []
        pred_list = []
        # with torch.no_grad():
        #     for i in tqdm(range(len(self.test_dataset))):
        #         k, gt = self.test_dataset[i]
        #         gt_list.append(gt)
        #         pred_list.append(0)
        #         if len(k) < 2:
        #             continue
        #         for j in range(max(len(k) - self.window_size, 1)):
        #             if len(k) <= self.window_size:
        #                 x = k[:-1]
        #                 next_log = k[-1].item()
        #             else:
        #                 x = k[j:j+self.window_size]
        #                 next_log = k[j+self.window_size].item()
        #             output = self.model(x.unsqueeze(0).to(self.device)).squeeze().cpu().numpy()
        #             output = np.argsort(-output)[: self.k]
        #             if next_log not in output:
        #                 pred_list[-1] = 1
        #                 # if gt == 0:
        #                 #     print(output)
        #                 #     print(next_log)
        #                 #     print(next_log in output)
        #                 #     sys.exit()
        #                 break
        start_time = time.time()
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                x_seq, next_log, num_windows, gt = batch
                gt_list.extend(gt)
                x_seq = x_seq.to(self.device)
                output = self.model(x_seq).cpu().numpy()
                output = np.argsort(-output)[:, : self.k]
                i = 0
                for j in num_windows:
                    pred_list.append(self.predict_sample(output[i : i + j], next_log[i : i + j]))
                    i += j
        
        end_time = time.time()
        print("Testing time: ", end_time - start_time)

                
                
        
        gt = np.array(gt_list)
        pred = np.array(pred_list)
        print("Accuracy: ", accuracy_score(gt, pred))
        print("Precision: ", precision_score(gt, pred))
        print("Recall: ", recall_score(gt, pred))
        print("F1: ", f1_score(gt, pred))
        
        with open('./deeplog_test.txt', 'a+') as f:
            f.write(f'Accuracy: {accuracy_score(gt, pred):.4f} Precision: {precision_score(gt, pred):.4f} Recall: {recall_score(gt, pred):.4f} F1: {f1_score(gt, pred):.4f} Time: {end_time - start_time}\n\n')
        
        
                    

        # num_candidate_range = np.arange(1, 11)
        # best_f1 = -1
        # best_precision = 0
        # best_recall = 0
        # best_num_candidate = 0

        # for num_candidate in num_candidate_range:
        #     pred = self.predict_by_candidates(next_log, output, num_candidate)
        #     precision = precision_score(gt, pred)
        #     recall = recall_score(gt, pred)
        #     f1 = f1_score(gt, pred)
        #     print("Num Candidate: ", num_candidate)
        #     print("F1: ", f1)
        #     if f1 > best_f1:
        #         best_f1 = f1
        #         best_precision = precision
        #         best_recall = recall
        #         best_num_candidate = num_candidate

        # print("Best F1: ", best_f1)
        # print("Best Precision: ", best_precision)
        # print("Best Recall: ", best_recall)
        # print("Best Num Candidate: ", best_num_candidate)
            

class LogAnomalyTester(Tester):
    def __init__(self, options):
        super().__init__(options)
        test_dataset = LogAnomalyDataset(options['dataset'], options['data_dir'], mode = 'test', embedding_strategy = options['embedding_strategy'])
        self.test_loader = DataLoader(test_dataset, batch_size = min(128, len(test_dataset)), num_workers=4, collate_fn=test_dataset.collate_fn, shuffle = False, pin_memory=True)
        self.vocab_size = len(test_dataset.vocab)
        self.window_size = test_dataset.window_size
        self.model = LogAnomaly(options['hidden_size'], options['num_layers'], self.vocab_size, options['embedding_dim'], options['dropout'])
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        del test_dataset
        gc.collect()

    def predict_sample(self, output, next_log):
        if torch.is_tensor(next_log):
            next_log = next_log.numpy()
        
        for i in range(len(output)):
            if next_log[i] not in output[i]:
                return 1

        return 0
        
    def predict_by_candidates(self, next_log, output, num_candidates):
        pred = []
        for i in range(len(output)):
            candidates = np.argsort(-output[i])[:num_candidates]
            if next_log[i] in candidates:
                pred.append(0)
            else:
                pred.append(1)
        
        return np.array(pred)


    def test(self):

        gt_list = []
        pred_list = []
        start_time = time.time()
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                x_sem, x_quan, next_log, num_windows, gt = batch
                gt_list.extend(gt.tolist())
                x_sem = x_sem.to(self.device)
                x_quan = x_quan.to(self.device)
                output = self.model(x_sem, x_quan).cpu().numpy()
                output = np.argsort(-output)[:, : self.k]
                i = 0
                for j in num_windows:
                    pred_list.append(self.predict_sample(output[i : i + j], next_log[i : i + j]))
                    i += j
                    # print(f'\rprogress {i} / {sum(num_windows)}')
                
                # if k % 10 == 0:
                #     print(objgraph.show_growth())

        end_time = time.time()
        print("Testing time: ", end_time - start_time)
        
        gt = np.array(gt_list)
        pred = np.array(pred_list)
        print("Accuracy: ", accuracy_score(gt, pred))
        print("Precision: ", precision_score(gt, pred))
        print("Recall: ", recall_score(gt, pred))
        print("F1: ", f1_score(gt, pred))

        with open('./loganomaly_test.txt', 'a+') as f:
            f.write(f'Accuracy: {accuracy_score(gt, pred):.4f} Precision: {precision_score(gt, pred):.4f} Recall: {recall_score(gt, pred):.4f} F1: {f1_score(gt, pred):.4f} Time: {end_time - start_time}\n\n')
                




# class PLELogTester(Tester):
    
#     def __init__(self, options):
#         super().__init__(options)
#         test_dataset = PLELogDataset(options['dataset'], options['data_dir'], mode = 'test', embedding_strategy = options['embedding_strategy'])
#         self.test_loader = DataLoader(test_dataset, batch_size = min(128, len(test_dataset)), num_workers=4, collate_fn=test_dataset.collate_fn, shuffle = False)
#         self.model = PLELog(options['num_layers'], options['hidden_size'], options['embedding_dim'], options['dropout'])
#         self.model.to(self.device)
#         self.model.load_state_dict(torch.load(self.model_path))
#         self.model.eval()
    
#     def test(self):
#         gt_list = []
#         output_list = []
#         with torch.no_grad():
#             for data in tqdm(self.test_loader):
#                 x_sem, mask, gt = data
#                 gt_list.append(gt)
#                 x_sem = x_sem.to(self.device)
#                 mask = mask.to(self.device)
#                 output = self.model(x_sem, mask)
#                 prob = torch.sigmoid(output)
#                 output_list.append(prob)
        
#         gt_list = torch.cat(gt_list).cpu().numpy()
#         output_list = torch.cat(output_list).cpu().numpy()

#         auroc = roc_auc_score(gt_list, output_list)

#         thre_range = (0, 1, 0.05)
#         best_f1 = -1
#         best_precision = 0
#         best_recall = 0
#         best_thre = 0

#         for thre in np.arange(*thre_range):
#             pred = np.where(output_list > thre, 1, 0)
#             precision = precision_score(gt_list, pred)
#             recall = recall_score(gt_list, pred)
#             f1 = f1_score(gt_list, pred)
#             if f1 > best_f1:
#                 best_f1 = f1
#                 best_precision = precision
#                 best_recall = recall
#                 best_thre = thre
        
#         print("Best F1: ", best_f1)
#         print("Best Precision: ", best_precision)
#         print("Best Recall: ", best_recall)
#         print("Best Threshold: ", best_thre)
#         print("AUROC: ", auroc)
#         precision, recall, thresholds = precision_recall_curve(gt_list, output_list)
#         pr_auc = auc(recall, precision)
#         print('AUPR: {:.4f}'.format(pr_auc))

    
