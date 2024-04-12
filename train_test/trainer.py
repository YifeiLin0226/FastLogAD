import gc
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, auc
from transformers import ElectraConfig
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import sys
import time
import yaml
from torch_optimizer.lamb import Lamb
import warnings
warnings.filterwarnings("ignore")

from data.dataset import HitAnomalyDataset, LogBertDataset, ElectraDataset, DeepLogDataset, LogAnomalyDataset, PLELogDataset
from data.vocab import WordVocab
from models.HitAnomaly import HitAnomaly
from models.bert_pytorch.model import BERT
from models.bert_pytorch.trainer import BERTTrainer
from models.electra import ElectraForLanguageModelingModel
from models.lstm import DeepLog, LogAnomaly
# from models.PLELog import PLELog




class Trainer:
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

    def train(self):
        pass

    def validate(self):
        pass



class HitAnomalyTrainer(Trainer):
    def __init__(self, options):
        self.model = HitAnomaly(options['embed_dim'], options['hidden_dim'], options['dff'], options['heads'])
        train_dataset = HitAnomalyDataset(options['dataset'], options['data_dir'], 'train')
        valid_dataset = HitAnomalyDataset(options['dataset'], options['data_dir'], 'valid')
        self.train_loader = DataLoader(train_dataset, batch_size= 1, shuffle=True, num_workers=4, pin_memory=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size= 1, num_workers=4)
        del train_dataset
        del valid_dataset
        gc.collect()
        super().__init__(options)
        self.best_f1 = 0
        self.best_model = None

    
    def train(self):
        os.makedirs(self.save_dir, exist_ok=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-5)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        self.model.to(self.device)
        loss = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            print('Epoch: {}'.format(epoch + 1))
            for batch_idx, data in enumerate(tqdm(self.train_loader)):
                log_seq = data['EventTemplate'].squeeze().to(self.device)
                param = data['ParameterList'].squeeze().to(self.device)
                label = data['Label'].to(self.device)
                output = self.model(log_seq, param)
                

                single_loss = criterion(output, label)
                loss += single_loss
                total_loss += single_loss

                if (batch_idx + 1) % 32 == 0 or batch_idx == len(self.train_loader) - 1:
                    optimizer.zero_grad()
                    (loss / 32).backward()
                    optimizer.step()
                    loss = 0
            
            
            print('Train Epoch: {}\tLoss: {:.6f}'.format(
                    epoch + 1, total_loss.item()))
            self.validate()
        
        
        
        


    
    def validate(self):
        self.model.eval()
        criterion = torch.nn.BCEWithLogitsLoss()
        test_loss = 0
        
        pred = []
        gt = []
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_loader):
                x = data['EventTemplate'].squeeze().to(self.device)
                y = data['ParameterList'].squeeze().to(self.device)
                label = data['Label'].to(self.device)
                output = self.model(x, y)
                test_loss += criterion(output, label).cpu().item()
                output = torch.sigmoid(output)
                pred.append(output.cpu().numpy())
                gt.append(label.cpu().numpy())
        pred = np.concatenate(pred)
        gt = np.concatenate(gt)
        roc_auc = roc_auc_score(gt, pred)
        pred = np.where(pred > 0.5, 1, 0)
        accuracy = accuracy_score(gt, pred)
        precision = precision_score(gt, pred)
        recall = recall_score(gt, pred)
        f1 = f1_score(gt, pred)
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_model = self.model.state_dict()
            torch.save(self.best_model, os.path.join(self.save_dir, 'best_model.pt'))
            yaml.dump(self.options, open(os.path.join(self.save_dir, 'options.yaml'), 'w'))
            print('Best F1: {:.4f}'.format(self.best_f1))
        
        
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, AUROC: {:.4f}\n'.format(
            test_loss, accuracy, precision, recall, f1, roc_auc))
        


                



    
    
class LogBertTrainer(Trainer):
    def __init__(self, options):
        super().__init__(options)
        train_dataset = LogBertDataset(options['dataset'], options['data_dir'], mask_ratio=options['mask_ratio'], mode = 'train')
        valid_dataset = LogBertDataset(options['dataset'], options['data_dir'], mode = 'valid', mask_ratio=options['mask_ratio'])
        self.train_loader = DataLoader(train_dataset, batch_size = options['batch_size'], shuffle=True, num_workers=4, pin_memory=True, collate_fn=train_dataset.collate_fn)
        self.valid_loader = DataLoader(valid_dataset, batch_size = options['batch_size'], num_workers=4, collate_fn=valid_dataset.collate_fn)
        print("Building BERT model")
        self.bert = BERT(len(train_dataset.vocab), max_len=options['max_len'], hidden=options['hidden'], n_layers=options['layers'], attn_heads=options['attn_heads'])
        if 'cuda' in self.device.type:
            with_cuda = True
        else:
            with_cuda = False
        self.trainer = BERTTrainer(self.bert, len(train_dataset.vocab), train_dataloader=self.train_loader
                                   , valid_dataloader=self.valid_loader, lr=options['lr'], betas=(self.adam_beta1, self.adam_beta2), weight_decay=self.adam_weight_decay, with_cuda = with_cuda, cuda_device= self.device, hypersphere_loss=options['hypersphere_loss'])
        del train_dataset
        del valid_dataset
        gc.collect()
        os.makedirs(self.model_dir, exist_ok=True)
     

    def train(self):
        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            if self.hypersphere_loss:
                center = self.calculate_center([self.train_loader, self.valid_loader])
                self.trainer.hyper_center = center
            
            _, train_dist = self.trainer.train(epoch)
            avg_loss, valid_dist = self.trainer.valid(epoch)
            self.trainer.save_log(self.model_dir, surfix_log="log2")

            if self.hypersphere_loss:
                self.trainer.radius = self.trainer.get_radius(train_dist + valid_dist, self.trainer.nu)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.trainer.save(self.model_path)
                epochs_no_improve = 0

                if epoch > 10 and self.hypersphere_loss:
                    best_center = self.trainer.hyper_center
                    best_radius = self.trainer.radius
                    total_dist = train_dist + valid_dist

                    if best_center is None:
                        raise TypeError("center is None")
                

                    print("best radius", best_radius)
                    best_center_path = self.model_dir + "best_center.pt"
                    print("Save best center", best_center_path)
                    torch.save({"center": best_center, "radius": best_radius}, best_center_path)

                    total_dist_path = self.model_dir + "best_total_dist.pt"
                    print("Save best total dist", total_dist_path)
                    torch.save(total_dist, total_dist_path)
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve == self.n_epochs_stop:
                print("Early stopping!")
                break
    
    def calculate_center(self, data_loader_list):
        print("start calculating center")
        with torch.no_grad():
            outputs = 0
            total_samples = 0
            for data_loader in data_loader_list:
                total_length = len(data_loader)
                data_iter = tqdm(enumerate(data_loader), total=total_length)
                for i, data in data_iter:
                    data = {key: value.to(self.device) for key, value in data.items()}
                    output = self.trainer.model(data["bert_input"], data["time_input"])
                    cls_output = output["cls_output"]
                    outputs += torch.sum(cls_output.detach().clone(), dim=0)
                    total_samples += cls_output.shape[0]

        center = outputs / total_samples

        return center
    


class ElectraTrainer(Trainer):
    def __init__(self, model_configs, options):
        super().__init__(options)
        train_dataset = ElectraDataset(options['dataset'], options['data_dir'], mode = 'train', seq_len = options['seq_len'], mask_num = options['mask_num'], mask_ratio=options['mask_ratio'], rmd_loss_weight= options['rmd_loss_weight'])
        valid_dataset = ElectraDataset(options['dataset'], options['data_dir'], mode = 'valid', seq_len = options['seq_len'], mask_num = options['mask_num'], mask_ratio=options['mask_ratio'], rmd_loss_weight= options['rmd_loss_weight'])
        self.train_loader = DataLoader(train_dataset, batch_size = options['batch_size'], shuffle=True, num_workers=4, pin_memory=True, collate_fn=train_dataset.collate_fn)
        self.valid_loader = DataLoader(valid_dataset, batch_size = options['batch_size'], num_workers=4, collate_fn=valid_dataset.collate_fn)
        test_dataset = ElectraDataset(options['dataset'], options['data_dir'], mode = 'test', seq_len = options['seq_len'], mask_num = options['mask_num'], mask_ratio=options['mask_ratio'])
        self.test_loader = DataLoader(test_dataset, batch_size = options['batch_size'], num_workers=4, collate_fn=test_dataset.collate_fn, shuffle = False)
        vocab_size = len(train_dataset.vocab)
        model_configs = ElectraConfig(vocab_size = vocab_size, **model_configs)
        print("Building Electra model")
        self.model = ElectraForLanguageModelingModel(model_configs, train_dataset.vocab, model_configs.output_size, random_generator = options['random_generator'])
        # self.model.tie_generator_and_discriminator_embeddings()
        self.model.to(self.device)
        self.discriminator = self.model.discriminator_model

        del train_dataset
        del valid_dataset
        gc.collect()

        os.makedirs(self.model_dir, exist_ok=True)
        print("Model dir: ", self.model_dir)
    
    def train(self):
        self.best_loss = float('inf')
        self.best_thre = 0
        epochs_no_improve = 0
        optimizer = torch.optim.Adam(self.model.discriminator_model.parameters(), lr=self.lr, betas=(self.adam_beta1, self.adam_beta2), weight_decay=self.adam_weight_decay)
        optimizer_gen = torch.optim.Adam(self.model.generator_model.parameters(), lr=self.lr, betas=(self.adam_beta1, self.adam_beta2), weight_decay=self.adam_weight_decay)
        # optimizer = Lamb(self.model.parameters(), lr = self.lr)
        rmd_criterion = torch.nn.BCEWithLogitsLoss(reduction = 'none')
        start = time.time()
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            num_samples = 0
            rmd_losses = 0
            rtd_losses = 0
            mlm_losses = 0
            mlm_total = 0

            # if epoch == self.pretrain_epoch:
            #     self.c = self.initialize_c()

            for batch_idx, data in enumerate(tqdm(self.train_loader)):
                loss = 0
                log_seq = data['k'].to(self.device)
                attn_mask = data['attention_mask'].to(self.device)
                labels = data['labels'].to(self.device)
                gt = data['gt'].to(self.device)

                mlm_start = time.time()
                outputs = self.model(log_seq, labels, attention_mask = attn_mask)

                if epoch < self.pretrain_epoch:

                    
                    if not self.random_generator:
                        
                        g_loss = outputs[3]
                        mlm_loss = self.mlm_loss_weight * g_loss
                        optimizer_gen.zero_grad()
                        mlm_loss.backward()
                        optimizer_gen.step()
                        mlm_losses += mlm_loss.item() * len(log_seq)
                        mlm_end = time.time()
                        mlm_total += mlm_end - mlm_start
                        
                    
                    rtd_loss = outputs[1]
                    loss += self.rtd_loss_weight * rtd_loss
                    rtd_losses += self.rtd_loss_weight * rtd_loss.item() * len(log_seq)
                    
                if epoch >= self.pretrain_epoch:
                    self.model.generator_model.eval()
                    rmd_out = outputs[0]

                    rmd_loss = (1 - gt) * torch.linalg.vector_norm(0.1 * rmd_out, dim = -1) - 1 * gt * torch.log(1 - torch.exp(-torch.linalg.vector_norm(rmd_out, dim = -1)))
                    # rmd_loss = (1 - gt) * torch.sum((rmd_out - self.c)**2, dim = -1) + 0 * gt * 1 / torch.sum((rmd_out - self.c)**2, dim = -1)
                    rmd_loss = rmd_loss.mean()

                    # rmd_loss = rmd_criterion(rmd_output.squeeze(), gt.float())
                    loss += self.rmd_loss_weight * rmd_loss
                    rmd_losses += self.rmd_loss_weight * rmd_loss.item() * len(log_seq)

                

                

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(log_seq)
                num_samples += len(log_seq)
                del log_seq, attn_mask, labels, gt, outputs, loss
            print('Train Epoch: {}\tLoss: {:.6f}\tRTDLoss: {:.6f}\tRMDLoss:{:.6f}\tMLMLoss:{:.6f}'.format(
                    epoch + 1, total_loss / num_samples, rtd_losses / num_samples, rmd_losses / num_samples, mlm_losses / num_samples))
            torch.cuda.empty_cache()
            print('MLM Time: {:.4f}'.format(mlm_total))
            if epoch >= self.pretrain_epoch:
                avg_loss = self.validate()
                # if (epoch + 1) % 30 == 0:
                #     self.test()
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    epochs_no_improve = 0
                    torch.save(self.model.state_dict(), self.model_path)
                    self.best_thre = self.cal_dist()
                    torch.save(self.best_thre, os.path.join(self.model_dir, 'best_thre.pt'))
                else:
                    epochs_no_improve += 1

                if epochs_no_improve == self.n_epochs_stop:
                    print("Early stopping!")
                    # torch.save(self.model.state_dict(), self.model_path)
                    # self.best_thre = self.cal_dist()
                    # torch.save(self.best_thre, os.path.join(self.model_dir, 'best_thre.pt'))
                    self.test()
                    break
                    
                print('\n')
            torch.cuda.empty_cache()
            
            # if epoch % 10 == 0:
            #     self.c = self.initialize_c()
        end = time.time()
        with open('./result2.txt', 'a+') as f:
            f.write(f"Training Time: {end - start}\n")

    # def initialize_c(self, eps = 0.1):
    #     print("Initializing c...")
    #     c = 0
    #     num_samples = 0
    #     self.model.eval()
    #     with torch.no_grad():
    #         for data in self.train_loader:
    #             log_seq = data['k']
    #             attn_mask = data['attention_mask']
    #             gt = data['gt']
    #             log_seq = log_seq[gt == 0].to(self.device)
    #             attn_mask = attn_mask[gt == 0].to(self.device)
    #             output = self.discriminator(log_seq, attention_mask = attn_mask)
    #             rmd_out = output[0]
    #             c += torch.sum(rmd_out, dim = 0)
    #             num_samples += len(log_seq)

    #     c = c / num_samples
    #     c[(abs(c) < eps) & (c < 0)] = -eps
    #     c[(abs(c) < eps) & (c > 0)] = eps
    #     print("Shape of c: ", c.shape)
    #     return c

    def validate(self):
        torch.cuda.empty_cache()
        self.model.eval()
        test_loss = 0
        num_samples = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(self.valid_loader)):
                loss = 0
                log_seq = data['k'].to(self.device)
                attn_mask = data['attention_mask'].to(self.device)
                

                outputs = self.discriminator(log_seq, attention_mask = attn_mask)
                
                rmd_out = outputs[0]
                # rmd_loss = torch.sigmoid(rmd_output)
                loss += torch.linalg.vector_norm(rmd_out, dim = -1).sum()

                # if not self.random_generator:
                #     g_loss = outputs[3]
                #     loss += self.mlm_loss_weight * g_loss

                test_loss += loss.item()
                num_samples += len(log_seq)
        
        loss = test_loss / num_samples
        print('Validation set: Average loss: {:.4f}'.format(
            loss))
        return loss
    

    def cal_dist(self):
        dist = []
        with torch.no_grad():
        #     for data in self.train_loader:
        #         log_seq = data['k']
        #         attn_mask = data['attention_mask']
        #         gt = data['gt']
        #         attn_mask = attn_mask[gt == 0].to(self.device)
        #         log_seq = log_seq[gt == 0].to(self.device)
        #         output = self.discriminator(log_seq, attention_mask = attn_mask)
        #         rmd_out = output[0]
        #         dist.append(torch.linalg.vector_norm(rmd_out, dim = -1))
            
            for data in self.valid_loader:
                log_seq = data['k'].to(self.device)
                attn_mask = data['attention_mask'].to(self.device)
                output = self.discriminator(log_seq, attention_mask = attn_mask)
                rmd_out = output[0]
                dist.append(torch.linalg.vector_norm(rmd_out, dim = -1))
                
        dist = torch.cat(dist).cpu()
        return torch.quantile(dist, 0.99)

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

    def test(self):
        print("Testing...")
        self.model.eval()
        gt = []
        pred = []
        with torch.no_grad():
            for data in tqdm(self.test_loader):
                log_seq = data['k'].to(self.device)
                attn_mask = data['attention_mask'].to(self.device)
                label = torch.max(data['gt'], dim = -1)[0]
                output = self.discriminator(log_seq, attention_mask = attn_mask)
                rmd_out = output[0]
                rmd_loss = torch.linalg.vector_norm(rmd_out, dim = -1)
                # rmd_loss = torch.sigmoid(rmd_output)
                rmd_loss = rmd_loss.cpu().numpy()
                label = label.cpu().numpy()
                pred.append(rmd_loss)  
                gt.append(label)
        gt = np.concatenate(gt)
        pred = np.concatenate(pred)
        roc_auc = roc_auc_score(gt, pred)
        print('AUROC: {:.4f}'.format(roc_auc))
        precision, recall, thresholds = precision_recall_curve(gt, pred)
        pr_auc = auc(recall, precision)
        print('AUPR: {:.4f}'.format(pr_auc))
        loss_normal = pred[gt == 0]
        loss_anomaly = pred[gt == 1]
        print('Average loss normal: {:.4f}'.format(loss_normal.mean()))
        print('Average loss anomaly: {:.4f}'.format(loss_anomaly.mean()))
        print('Max loss normal: {:.4f}'.format(loss_normal.max()))
        print('Max loss anomaly: {:.4f}'.format(loss_anomaly.max()))


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
        
        thre= self.cal_dist()
        thre = thre.numpy()
        print('Threshold: {:.6f}'.format(thre))
        my_pred = np.where(pred > thre, 1, 0)
        accuracy = accuracy_score(gt, my_pred)
        precision = precision_score(gt, my_pred)
        recall = recall_score(gt, my_pred)
        f1 = f1_score(gt, my_pred)
        print('Accuracy: {:.4f}'.format(accuracy))
        print('Precision: {:.4f}'.format(precision))
        print('Recall: {:.4f}'.format(recall))
        print('F1: {:.4f}'.format(f1))


                
                

                
# class ElectraRawTrainer(Trainer):
#     def __init__(self, model_configs, options):
#         super().__init__(options)
#         train_dataset = ElectraRawDataset(options['dataset'], options['data_dir'], mode = 'train', mask_ratio = options['mask_ratio'])
#         valid_dataset = ElectraRawDataset(options['dataset'], options['data_dir'], mode = 'valid', mask_ratio = options['mask_ratio'])
#         test_dataset =  ElectraRawDataset(options['dataset'], options['data_dir'], mode = 'test', mask_ratio = options['mask_ratio'])
#         self.train_loader = DataLoader(train_dataset, batch_size = options['batch_size'], shuffle=True, num_workers=4, pin_memory=True, collate_fn=train_dataset.collate_fn)
#         self.valid_loader = DataLoader(valid_dataset, batch_size = options['batch_size'], num_workers=4, collate_fn=valid_dataset.collate_fn)
#         self.test_loader = DataLoader(test_dataset, batch_size = options['batch_size'], num_workers=4, collate_fn=test_dataset.collate_fn)
#         vocab_size = len(train_dataset.vocab)
#         model_configs = ElectraConfig(vocab_size = vocab_size, **model_configs)
#         print("Building Electra model")
#         self.model = ElectraForLanguageModelingModel(model_configs, train_dataset.vocab, model_configs.output_size, random_generator = options['random_generator'])
#         self.model.tie_generator_and_discriminator_embeddings()
#         self.model.to(self.device)
#         self.discriminator = self.model.discriminator_model

#         del train_dataset
#         del valid_dataset
#         gc.collect()

        
#         os.makedirs(self.model_dir, exist_ok=True)
    
#     def train(self):
#         self.best_loss = float('inf')
#         epochs_no_improve = 0
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.adam_beta1, self.adam_beta2), weight_decay=self.adam_weight_decay)
#         # optimizer = Lamb(self.model.parameters(), lr = self.lr)
#         # rmd_criterion = torch.nn.CrossEntropyLoss()
#         for epoch in range(self.epochs):
#             self.model.train()
#             total_loss = 0
#             num_samples = 0
#             rmd_losses = 0
#             rtd_losses = 0
#             for batch_idx, data in enumerate(tqdm(self.train_loader)):
#                 loss = 0
#                 log_seq = data['k'].to(self.device)
#                 attn_mask = data['attention_mask'].to(self.device)
#     


class DeepLogTrainer(Trainer):
    def __init__(self, options):
        super().__init__(options)
        train_dataset = DeepLogDataset(options['dataset'], options['data_dir'], mode = 'train')
        valid_dataset = DeepLogDataset(options['dataset'], options['data_dir'], mode = 'valid')
        self.train_loader = DataLoader(train_dataset, batch_size = options['batch_size'], shuffle=True, num_workers=4, pin_memory=True, collate_fn=train_dataset.collate_fn)
        self.valid_loader = DataLoader(valid_dataset, batch_size = options['batch_size'], num_workers=4, collate_fn=valid_dataset.collate_fn)
        print("Building DeepLog model")
        self.model = DeepLog(options['hidden_size'], options['num_layers'], len(train_dataset.vocab))
        self.model.to(self.device)
        self.best_loss = float('inf')
        os.makedirs(self.model_dir, exist_ok=True)
    
    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()
        epochs_no_improve = 0
        for epoch in range(self.epochs):
            if epoch == 0:
                optimizer.param_groups[0]['lr'] /= 32
            if epoch in [1, 2, 3, 4, 5]:
                optimizer.param_groups[0]['lr'] *= 2
            if epoch in self.lr_step:
                optimizer.param_groups[0]['lr'] *= self.lr_decay_ratio
            self.model.train()
            total_loss = 0
            num_samples = 0
            for batch_idx, data in enumerate(tqdm(self.train_loader)):
                loss = 0
                x, next_log = data
                x = x.to(self.device)
                next_log = next_log.to(self.device)
                output = self.model(x)
                output = output.view(-1, output.shape[-1])
                next_log = next_log.view(-1)
                loss = criterion(output, next_log)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(x)
                num_samples += len(x)
            print('Train Epoch: {}\tLoss: {:.6f}'.format(
                    epoch + 1, total_loss / num_samples))
            # val_loss = self.validate()
            # if val_loss < self.best_loss:
            #     self.best_loss = val_loss
            #     epochs_no_improve = 0
            #     torch.save(self.model.state_dict(), self.model_path)
            # else:
            #     epochs_no_improve += 1
            
            # if epochs_no_improve == self.n_epochs_stop:
            #     print("Early stopping!")
            #     break
            # print('\n')

        torch.save(self.model.state_dict(), self.model_path)
    
    def validate(self):
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        test_loss = 0
        num_samples = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(self.valid_loader)):
                loss = 0
                x, next_log = data
                x = x.to(self.device)
                next_log = next_log.to(self.device)
                output = self.model(x)
                output = output.view(-1, output.shape[-1])
                next_log = next_log.view(-1)
                loss = criterion(output, next_log)
                test_loss += loss.item() * len(x)
                num_samples += len(x)
        print('Validation set: Average loss: {:.4f}'.format(
            test_loss / num_samples))

        return test_loss / num_samples

                
                

class LogAnomalyTrainer(Trainer):
    def __init__(self, options):
        super().__init__(options)
        train_dataset = LogAnomalyDataset(options['dataset'], options['data_dir'], mode = 'train', embedding_strategy = options['embedding_strategy'])
        valid_dataset = LogAnomalyDataset(options['dataset'], options['data_dir'], mode = 'valid', embedding_strategy = options['embedding_strategy'])
        self.train_loader = DataLoader(train_dataset, batch_size = options['batch_size'], shuffle=True, num_workers=4, pin_memory=True, collate_fn=train_dataset.collate_fn)
        self.valid_loader = DataLoader(valid_dataset, batch_size = options['batch_size'], num_workers=4, collate_fn=valid_dataset.collate_fn)
        print("Building LogAnomaly model")
        self.model = LogAnomaly(options['hidden_size'], options['num_layers'], len(train_dataset.vocab), options['embedding_dim'], options['dropout'])
        self.model.to(self.device)
        self.best_loss = float('inf')
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()
        epochs_no_improve = 0
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            if epoch == 0:
                optimizer.param_groups[0]['lr'] /= 32
            if epoch in [1, 2, 3, 4, 5]:
                optimizer.param_groups[0]['lr'] *= 2
            if epoch in self.lr_step:
                optimizer.param_groups[0]['lr'] *= self.lr_decay_ratio
            num_samples = 0
            for batch_idx, data in enumerate(tqdm(self.train_loader)):
                loss = 0
                x_sem, x_quant, next_log = data
                x_sem = x_sem.to(self.device)
                x_quant = x_quant.to(self.device)
                next_log = next_log.to(self.device)
                output = self.model(x_sem, x_quant)
                output = output.view(-1, output.shape[-1])
                next_log = next_log.view(-1)
                loss = criterion(output, next_log)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(x_sem)
                num_samples += len(x_sem)
            print('Train Epoch: {}\tLoss: {:.6f}'.format(
                    epoch + 1, total_loss / num_samples))
            # val_loss = self.validate()
            # if val_loss < self.best_loss:
            #     self.best_loss = val_loss
            #     epochs_no_improve = 0
            #     torch.save(self.model.state_dict(), self.model_path)
            # else:
            #     epochs_no_improve += 1
            
            # if epochs_no_improve == self.n_epochs_stop:
            #     print("Early stopping!")
            #     break
            # print('\n')
        torch.save(self.model.state_dict(), self.model_path)
        
    def validate(self):
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        test_loss = 0
        num_samples = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(self.valid_loader)):
                loss = 0
                x_sem, x_quant, next_log = data
                x_sem = x_sem.to(self.device)
                x_quant = x_quant.to(self.device)
                next_log = next_log.to(self.device)
                output = self.model(x_sem, x_quant)
                output = output.view(-1, output.shape[-1])
                next_log = next_log.view(-1)
                loss = criterion(output, next_log)
                test_loss += loss.item() * len(x_sem)
                num_samples += len(x_sem)
        print('Validation set: Average loss: {:.4f}'.format(
            test_loss / num_samples))

        return test_loss / num_samples



# class PLELogTrainer(Trainer):
    
#     def __init__(self, options):
#         super().__init__(options)
#         train_dataset = PLELogDataset(options['dataset'], options['data_dir'], mode = 'train', embedding_strategy = options['embedding_strategy'], use_pseudo = options['use_pseudo'], tok_anomaly_ratio = options['tok_anomaly_ratio'])
#         valid_dataset = PLELogDataset(options['dataset'], options['data_dir'], mode = 'valid', embedding_strategy = options['embedding_strategy'])
#         self.train_loader = DataLoader(train_dataset, batch_size = options['batch_size'], shuffle=True, num_workers=4, pin_memory=True, collate_fn=train_dataset.collate_fn)
#         self.valid_loader = DataLoader(valid_dataset, batch_size = options['batch_size'], num_workers=4, collate_fn=valid_dataset.collate_fn)

#         test_dataset = PLELogDataset(options['dataset'], options['data_dir'], mode = 'test', embedding_strategy = options['embedding_strategy'])
#         self.test_loader = DataLoader(test_dataset, batch_size = min(128, len(test_dataset)), num_workers=4, collate_fn=test_dataset.collate_fn, shuffle = False)

#         print("Building PLELog model")
#         self.model = PLELog(options['num_layers'], options['hidden_size'], options['embedding_dim'], options['dropout'])
#         self.model.to(self.device)
#         self.best_loss = float('inf')
#         os.makedirs(self.model_dir, exist_ok=True)

#     def train(self):
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
#         epochs_no_improve = 0
#         for epoch in range(self.epochs):
#             self.model.train()
#             total_loss = 0
#             num_samples = 0
#             for batch_idx, data in enumerate(tqdm(self.train_loader)):
#                 loss = 0
#                 x_sem, mask, gt = data
#                 x_sem = x_sem.to(self.device)
#                 mask = mask.to(self.device)
#                 gt = gt.to(self.device)
#                 output = self.model(x_sem, mask)
#                 output = output.view(-1)
#                 criterion = torch.nn.BCEWithLogitsLoss()
#                 loss = criterion(output.squeeze(), gt)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item() * len(x_sem)
#                 num_samples += len(x_sem)
#             print('Train Epoch: {}\tLoss: {:.8f}'.format(
#                     epoch + 1, total_loss / num_samples))
#             val_loss = self.validate()
#             self.test()
#             if val_loss < self.best_loss:
#                 self.best_loss = val_loss
#                 epochs_no_improve = 0
#                 torch.save(self.model.state_dict(), self.model_path)
#             else:
#                 epochs_no_improve += 1
            
#             # self.test()
#             if epochs_no_improve == self.n_epochs_stop:
#                 print("Early stopping!")
#                 break
#             print('\n')
    

#     def validate(self):
#         self.model.eval()
#         criterion = torch.nn.BCEWithLogitsLoss()
#         test_loss = 0
#         num_samples = 0
#         with torch.no_grad():
#             for batch_idx, data in enumerate(tqdm(self.valid_loader)):
#                 loss = 0
#                 x_sem, mask, _ = data
#                 x_sem = x_sem.to(self.device)
#                 mask = mask.to(self.device)
#                 output = self.model(x_sem, mask)
#                 output = output.view(-1, output.shape[-1])
#                 label = torch.zeros_like(output)
#                 loss = criterion(output, label)
#                 test_loss += loss.item() * len(x_sem)
#                 num_samples += len(x_sem)
#         print('Validation set: Average loss: {:.8f}'.format(
#             test_loss / num_samples))

#         return test_loss / num_samples
    


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
#         print("AUROC: ", auroc)
#         precision, recall, thresholds = precision_recall_curve(gt_list, output_list)
#         aupr = auc(recall, precision)
#         print("AUPR: ", aupr)


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
#             # print("Threshold: ", thre)
#             # print("F1: ", f1)
#             if f1 > best_f1:
#                 best_f1 = f1
#                 best_precision = precision
#                 best_recall = recall
#                 best_thre = thre
        
#         print("Best F1: ", best_f1)
#         print("Best Precision: ", best_precision)
#         print("Best Recall: ", best_recall)
#         print("Best Threshold: ", best_thre)