import torch.types
import os
from config import Config
from sklearn.model_selection import StratifiedShuffleSplit
from transform import Compose,ToTensor,RandomHorizontalFlip,RandomVerticalFlip,Resize,Normalize
from mydataset import Mydataset
from model import Alexnet
import torch.nn as nn
from torch.utils.data import DataLoader
from util import train,val,test
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
# import optuna

# TODO: 搭建模型(Alexnet,VGG,Resnet)
def createModel(device,pretrain=False):

    model = Alexnet(ch_in=3,cls_num=3)
    model = model.to(device=device)

    return model


# TODO: 构建dataset（train+val+test）
def readDatalist(root,className):
    nameDic = {}
    images = []
    labels = []
    with open(className, 'r') as files:
        names = files.read().strip().split()
        for i, name in enumerate(names):
            nameDic[str(name)] = i

    for case in sorted(os.listdir(root)):
        imges = sorted(os.listdir(os.path.join(root, case)))
        for img in imges:
            images.append(os.path.join(os.path.join(root, case), img))
            labels.append(nameDic[case])

    return images, labels

# def objective(trail):
#     params = {
#         "batch_size":trail.suggest_int('batchsize', 2,4),
#         "lr":trail.suggest_loguniform('lr', 1e-4, 1e-2),
#         "weight_decay": trail.suggest_loguniform("weight_devay",1e-4,1e-3),
#         "optimizer": trail.suggest_categorical("optimizer",["Adam","SGD"]),
#     }
#
#     loss = main(params)
#     return np.mean(loss)
# TODO: 数据预处理(数据增强，去噪等)

def main():
    arg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    imagesPath,labelsPath = readDatalist(arg.trainRoot,arg.className)
    testsImgPath,testsLabelPath = readDatalist(arg.testRoot,arg.className)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=arg.factor, random_state=42)
    train_indices, val_indices = next(sss.split(imagesPath, labelsPath))

    train_images = [imagesPath[i] for i in train_indices]
    train_labels = [labelsPath[i] for i in train_indices]

    val_images = [imagesPath[i] for i in val_indices]
    val_labels = [labelsPath[i] for i in val_indices]


    train_files = [{"image": image, "label": label} for image, label in zip(train_images, train_labels)]
    val_files = [{"image": image, "label": label} for image, label in zip(val_images, val_labels)]
    test_files = [{"image": image, "label": label} for image, label in zip(testsImgPath, testsLabelPath)]


    train_transform = Compose([
        Resize((227,227),interpolation='bilinear'),
        ToTensor(),
        RandomVerticalFlip(),
        RandomHorizontalFlip(),
        Normalize(),
    ])

    train_dataset = Mydataset(train_files,train_transform)
    val_dataset = Mydataset(val_files,train_transform)
    test_dataset = Mydataset(test_files,train_transform)

    loss_function = nn.CrossEntropyLoss()
    lr = arg.lr
    batch_size = arg.batch_size
    model = createModel(device,pretrain=False)
    optimizer = torch.optim.SGD(params=model.parameters(),)
    lr_schedular = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=1000,gamma=0.33)

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=2,pin_memory=True)
    val_loader = DataLoader(val_dataset,batch_size=1,shuffle=False,num_workers=2,pin_memory=True)
    test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=2,pin_memory=True)

    if arg.resume != " ":
        check_point = torch.load(arg.resume,map_location='cpu')
        model.load_state_dict(check_point["model"])
        optimizer.load_state_dict(check_point["optimizer"])
        lr_schedular.load_state_dict(check_point['lr_scheduler'])
        arg.start_epoch = check_point["epoch"] + 1

        print("load resume")


    train_loss = []
    val_loss = []
    acc_values = []
    best_value = -1
    best_acc_result_epoch = 0

    for epoch in range(arg.start_epoch,arg.max_epoch):
        print(f"epoch {epoch} / {arg.max_epoch}")
        loss,lr_now = train(model,optimizer,train_loader,device,epoch,arg,
                     loss_function,warmup=True)
        train_loss.append(loss)
        lr_schedular.step()

        if (epoch + 1) % arg.val_interval == 0:
            eval_loss,acc = val(model,val_loader,arg,epoch,device,loss_function)
            val_loss.append(eval_loss)
            acc_values.append(acc)
            save_file = {"model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": lr_schedular.state_dict(),
                         "opech": epoch}

            if arg.save_best:
                if acc > best_value:
                    best_value = acc
                    best_acc_result_epoch = epoch
                    torch.save(save_file, os.path.join(arg.saveWeights, "best_model.pth"))
                    print("saved new best metric model")
                    print(f"best acc: {best_value}")
                    print(f"at epoch: {best_acc_result_epoch},current_lr:{lr_now}")
                print(
                    f"\ncurrent acc: {np.mean(acc):.4f}"
                )
            torch.save(save_file, os.path.join(arg.saveWeights, "model.pth"))

    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d")
    plt.figure(figsize=(10,8))
    
    xt = [i for i in range(len(train_loss))]
    xv = [i*2 for i in range(len(val_loss))]
    plt.subplot(1,2,1)
    plt.plot(xt,train_loss)
    plt.plot(xv,val_loss)
    plt.title("loss function")
    plt.xlabel("epoch")
    plt.ylabel("loss")

    xacc = [i for i in range(len(acc_values))]
    plt.subplot(1,2,2)
    plt.plot(xacc,acc_values)
    plt.title("accuracy graph")
    plt.xlabel("epoch")
    plt.ylabel("acc")

    plt.tight_layout()

    save_path = os.path.join(arg.saveLogs, "log", formatted_time)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(arg.saveLogs,"log",formatted_time,"result.jpg"))



if __name__ == "__main__":
    main()