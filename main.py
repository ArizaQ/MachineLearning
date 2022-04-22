import logging.config
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from MLwork.methods.train import Trainer
from configuration import config
from MLwork.utils.data_loader import get_statistics
from methods.finetune import Finetune
from collections import defaultdict
from dataset import get_datalist, get_classlist

def main():
    args = config.base_parser()


    logging.config.fileConfig("./configuration/logging.conf")
    logger = logging.getLogger()

    os.makedirs(f"logs/{args.dataset}", exist_ok=True)
    save_path = "tmp"
    fileHandler = logging.FileHandler("logs/{}.log".format(save_path), mode="w")
    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)


    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

        logger.info(f"Set the device ({device})")

    torch.manual_seed(113)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(113)
    random.seed(113)

    mean, std, n_classes, inp_size, _ = get_statistics(dataset=args.dataset)
    train_transform = transforms.Compose(
        [
            transforms.Resize((inp_size, inp_size)),
            transforms.RandomCrop(inp_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((inp_size, inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    criterion = nn.CrossEntropyLoss(reduction="mean")
    # exemplar = Exemplar(args.max_size, 2)

    # generate class order
    # split classes to args.n_task  tasks
    class_of_task = [20,20,20,20,20]   # the classes of each task
    class_order = get_classlist(args)# all classes
    random.shuffle(class_order)
    start_label = 0

    kwargs = vars(args)

    # create a instance of your method
    method = Trainer(criterion, device, train_transform, test_transform, class_of_task[0], n_classes, **kwargs)
    # method=Finetune(criterion, device, train_transform, test_transform, class_of_task[0], n_classes, **kwargs)
    task_records = defaultdict(list)
    batch_num=args.batch_num
    assert batch_num==len(class_of_task)
    for inc_i in range(batch_num):# 对每一个类别训练
        print("\n" + "#" * 50)
        print(f"# Task {inc_i} iteration")
        print("#" * 50 + "\n")

        logger.info("set train/test datalist for the current task!")
        # get datalist 
        cur_train_datalist = get_datalist(args, 'train', class_order[start_label:start_label+class_of_task[inc_i]], start_label)
        cur_test_datalist = get_datalist(args, 'test', class_order[:start_label+class_of_task[inc_i]], 0)
        method.set_current_dataset(cur_train_datalist, cur_test_datalist)
        # before task:  update some attribute about network(fc layer,  optimizer,  num_learning_class)
        # method.before_task(cur_train_datalist)
        method.before_task(cur_train_datalist)
        # start training
        task_acc, eval_dict = method.train(inc_i)

        method.after_task(inc_i)
        task_records["task_acc"].append(task_acc)
        # task_records['cls_acc'][k][j] = break down j-class accuracy from 'task_acc'
        task_records["cls_acc"].append(eval_dict["cls_acc"])
        start_label += class_of_task[inc_i]


    # Accuracy (A)
    A_avg = np.mean(task_records["task_acc"])
    A_last = task_records["task_acc"][len(class_of_task) - 1]


    logger.info(f"======== Summary =======")
    logger.info(f"A_last {A_last}")


if __name__ == "__main__":
    main()
