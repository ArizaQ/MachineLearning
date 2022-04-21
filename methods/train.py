from backbone.model import PreResNet, BiasLayer
from methods.examplar import Exemplar
from methods.finetune import Finetune
from utils.train_utils import select_model, select_optimizer
from sklearn.model_selection import train_test_split
from utils.data_loader import ImageDataset
import random
from torch.utils.data import DataLoader
import pandas as pd
import logging
import torch.nn as nn
import torch
import torch.optim as optim
from methods.dataset import BatchData
from torch.optim.lr_scheduler import LambdaLR, StepLR
from copy import deepcopy
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import datetime
logger = logging.getLogger()



class Trainer(Finetune):
    def __init__(self,exempler, criterion, device, train_transform, test_transform, init_class, n_classes, **kwargs):
        super().__init__( criterion, device, train_transform, test_transform, init_class, n_classes, **kwargs)
        self.total_cls = kwargs['total_cls']
        self.seen_cls = 0
        self.model = PreResNet(32, 100).cuda()
        self.model = nn.DataParallel(self.model, device_ids=[0])  # 0,1
        # bias correction 使用 偏置层
        self.bias_layer1 = BiasLayer().cuda()
        self.bias_layer2 = BiasLayer().cuda()
        self.bias_layer3 = BiasLayer().cuda()
        self.bias_layer4 = BiasLayer().cuda()
        self.bias_layer5 = BiasLayer().cuda()
        self.bias_layers = [self.bias_layer1, self.bias_layer2, self.bias_layer3, self.bias_layer4, self.bias_layer5]

        self.lr=kwargs["lr"]
        self.max_size=kwargs["max_size"]
        self.batch_num=kwargs["batch_num"]
        self.criterion = nn.CrossEntropyLoss()
        self.test_accs = []
        self.exemplar = Exemplar(self.max_size, self.total_cls)
        self.test_s=[]
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Solver total trainable parameters : ", total_params)

    def get_train_and_val(self,cur_train_datalist):
        return cur_train_datalist[0:9000], cur_train_datalist[9000:10000]
    def set_current_dataset(self, train_datalist, test_datalist):
        random.shuffle(train_datalist)
        self.train_list, self.val_list= self.get_train_and_val(train_datalist)
        self.test_list = test_datalist
    def get_x_and_y(self,datalist):
        x_list=[]
        y_list=[]
        for item in datalist:
            x_list.append(item["file_name"])
            y_list.append(item["label"])
        return x_list,y_list

    def get_list(self,val_x,val_y):
        val_list=[]
        for i in range(len(val_x)):
            val_list.append({"file_name":val_x[i],"label":val_y[i]})
        return val_list

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def before_task(self, datalist):
        logger.info("Apply before_task")
        logger.info("asdasdasdas")

        incoming_classes = pd.DataFrame(datalist)["label"].unique().tolist()
        incoming_classes = list(set(incoming_classes))
        incoming_classes = 1 + max(incoming_classes)
        logger.info("Increasing fc layer:  {}  ---->  {}".format(self.num_learned_class, incoming_classes))
        self.num_learning_class = incoming_classes
    def train(self, inc_i):
        logger.info("#" * 10 + "Start Training" + "#" * 10)
        logger.info(f"Incremental num : {inc_i}")
        train_list = self.train_list + self.memory_list
        self.test_s.extend(self.test_list)
        train_data, test_data = self.get_dataloader(
            self.batch_size, self.n_woker, train_list, self.test_s
        )

        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=2e-4)
        scheduler = StepLR(optimizer, step_size=70, gamma=0.1)
        bias_optimizer = optim.Adam(self.bias_layers[inc_i].parameters(), lr=0.001)
        self.exemplar.update(self.total_cls // self.batch_num, (self.get_x_and_y(self.train_list)),
                             (self.get_x_and_y(self.val_list)))
        self.seen_cls = self.exemplar.get_cur_cls()
        val_xs, val_ys = self.exemplar.get_exemplar_val()
        # val_bias_data= self.get_dataloader(self.batch_size, self.n_woker, self.get_list(val_xs,val_ys), None)[0]
        val_bias_data= self.get_dataloader(self.batch_size, self.n_woker, self.memory_list, None)[0]
        self.seen_cls =self.seen_cls+20;
        test_acc = []
        eval_dict = dict()
        for epoch in range(self.n_epoch):
            logger.info("---" * 50)
            logger.info("Epoch"+" "+ str(epoch))
            logger.info("start stage1 in this epoch:")
            logger.info(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            scheduler.step()
            cur_lr = self.get_lr(optimizer)
            logger.info("Current Learning Rate : "+str(cur_lr))
            self.model.train()
            for _ in range(len(self.bias_layers)):
                self.bias_layers[_].eval()
            if inc_i > 0:
                self.stage1_distill(train_data, self.criterion, optimizer)
            else:
                self.stage1(train_data, self.criterion, optimizer)
            acc = self.test(test_data)
            logger.info("end stage1 in this epoch:")
            logger.info(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        logger.info("start stage2 in this epoch:")
        logger.info(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        if inc_i > 0:
            for epoch in range(self.n_epoch):
                # bias_scheduler.step()
                self.model.eval()
                for _ in range(len(self.bias_layers)):
                    self.bias_layers[_].train()
                self.stage2(val_bias_data, self.criterion, bias_optimizer)
                if epoch % 50 == 0:
                    acc = self.test(test_data)
                    test_acc.append(acc)
        for i, layer in enumerate(self.bias_layers):
            layer.printParam(i)
        self.previous_model = deepcopy(self.model)
        logger.info("end stage2 in this epoch:")
        logger.info(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        acc = self.test(test_data)
        test_acc.append(acc)
        self.test_accs.append(max(test_acc))
        logger.info("test_accs:")
        logger.info(self.test_accs)

        eval_dict = self.evaluation(test_loader=test_data, criterion=self.criterion)
        return test_acc,eval_dict

    def stage1_distill(self, train_data, criterion, optimizer):
        # 蒸馏old data
        logger.info("Training ... ")
        distill_losses = []
        ce_losses = []
        T = 2
        alpha = (self.seen_cls - 20)/ self.seen_cls
        logger.info("classification proportion 1-alpha = "+" "+ str(1-alpha))
        for i, data in enumerate(tqdm(train_data)):
            image = data['image'].to(self.device)
            label = data['label'].to(self.device)
            p = self.model(image)
            p = self.bias_forward(p)
            with torch.no_grad():
                pre_p = self.previous_model(image)
                pre_p = self.bias_forward(pre_p)
                pre_p = F.softmax(pre_p[:,:self.seen_cls-20]/T, dim=1)
            logp = F.log_softmax(p[:,:self.seen_cls-20]/T, dim=1)
            loss_soft_target = -torch.mean(torch.sum(pre_p * logp, dim=1))
            loss_hard_target = nn.CrossEntropyLoss()(p[:,:self.seen_cls], label)
            loss = loss_soft_target * T * T + (1-alpha) * loss_hard_target
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            distill_losses.append(loss_soft_target.item())
            ce_losses.append(loss_hard_target.item())
        logger.info("stage1 distill loss :"+str( np.mean(distill_losses))+ "ce loss :"+str( np.mean(ce_losses)))


    def stage1(self, train_data, criterion, optimizer):
        logger.info("Training ... ")
        losses = []
        for i, data in enumerate(tqdm(train_data)):
            image= data['image'].to(self.device)
            label = data['label'].view(-1).to(self.device)
            p = self.model(image)
            p = self.bias_forward(p)
            loss = criterion(p[:,:self.seen_cls], label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        logger.info("stage1 loss :"+ " "+str(np.mean(losses)))

    def stage2(self, val_bias_data, criterion, optimizer):
        logger.info("Evaluating ... ")
        losses = []
        for i, data in enumerate(tqdm(val_bias_data)):
            image = data['image'].to(self.device)
            label = data['label'].to(self.device)
            p = self.model(image)
            p = self.bias_forward(p)
            loss = criterion(p[:,:self.seen_cls], label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        logger.info("stage2 loss :"+" "+ str(np.mean(losses)))

    def bias_forward(self, input):
        in1 = input[:, :20]
        in2 = input[:, 20:40]
        in3 = input[:, 40:60]
        in4 = input[:, 60:80]
        in5 = input[:, 80:100]
        out1 = self.bias_layer1(in1)
        out2 = self.bias_layer2(in2)
        out3 = self.bias_layer3(in3)
        out4 = self.bias_layer4(in4)
        out5 = self.bias_layer5(in5)
        return torch.cat([out1, out2, out3, out4, out5], dim = 1)
    def test(self, testdata):
        logger.info("test data number : "+" "+ str(len(testdata)))
        self.model.eval()
        count = 0
        correct = 0
        wrong = 0
        for i, data in enumerate(testdata):
            image = data['image'].to(self.device)
            label = data['label'].to(self.device)
            p = self.model(image)
            p = self.bias_forward(p)
            pred = p[:,:self.seen_cls].argmax(dim=-1)
            correct += sum(pred == label).item()
            wrong += sum(pred != label).item()
        acc = correct / (wrong + correct)
        logger.info("Test Acc: {}".format(acc*100))
        self.model.train()
        print("---------------------------------------------")
        return acc
