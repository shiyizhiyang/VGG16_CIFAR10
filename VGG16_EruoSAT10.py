#============================================================================
# ISSL learning: ResNet18 on EuroSAT

import ssl 
ssl._create_default_https_context = ssl._create_unverified_context
# 解释：导入 ssl 模块，并设置默认的 HTTPS 上下文为不验证证书的上下文。
# 这通常是为了在与某些服务器通信时忽略证书验证问题，可能是因为使用了自签名证书或其他原因。

# Single program simulation 
# ============================================================================
import torch, torchvision   # 解释：导入 PyTorch 和 PyTorch 的视觉库，用于深度学习和图像相关的操作。
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math
import os.path
import pandas as pd
# 解释：从 PyTorch 和 PyTorch 视觉库中进一步导入特定的模块，如神经网络模块 nn、图像变换模块 transforms、数据加载器和数据集类 DataLoader 和 Dataset，
# 以及一些常用的数学函数、文件路径判断函数和 pandas 库用于数据处理。
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
from pandas import DataFrame
# 解释：从 scikit-learn 的模型选择模块中导入 train_test_split 用于数据集分割，从 PIL 库中导入 Image 用于图像处理，从 glob 模块中导入函数用于文件路径匹配，
# 从 pandas 中导入 DataFrame 用于构建数据表格。
import random
import numpy as np
import os
# 解释：导入 random 用于生成随机数，导入 numpy 用于数值计算，导入 os 用于操作系统相关的操作。
import tensorboard
from torch.utils.tensorboard import SummaryWriter
# 解释：导入 tensorboard 库，并从 PyTorch 的 tensorboard 模块中导入 SummaryWriter 用于记录训练过程中的数据以便在 TensorBoard 中进行可视化。

from VGG16 import VGG16_client_side, VGG16_server_side
# 解释：从自定义的 VGG16 模块中导入特定的功能模块，可能是用于特定任务的客户端和服务器端的功能。

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
# 解释：导入 matplotlib 库，并设置使用 Agg 后端，然后导入 pyplot 用于绘图，导入 copy 用于对象的复制操作。


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))    
# 解释：设置一个固定的随机种子值为 1234。然后分别为 Python 的内置随机模块、NumPy 的随机模块、
# PyTorch 的 CPU 随机生成器和 PyTorch 的 GPU 随机生成器设置这个种子。如果 GPU 可用，
# 还设置 PyTorch 的 cuDNN 后端为确定性模式，并打印出第一个 GPU 的设备名称。

#===================================================================
program = "ISSL VGG-16 on CIFAR-10"
print(f"---------{program}----------")              # this is to identify the program in the slurm outputs files
# 解释：定义一个字符串表示程序名称，然后打印出来，可能是为了在输出文件中方便识别这个特定的程序。
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 解释：根据 GPU 是否可用，选择使用 GPU（'cuda'）或 CPU（'cpu'）作为计算设备。

# To print in color -------test/train of the client side
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))     
# 解释：在客户端的测试或训练过程中，使用prRed函数输出错误信息，使用prGreen函数输出成功信息。

#===================================================================
# No. of users
num_users = 20 # 2 * 6 orbits
epochs = 150
frac = 1        # participation of clients; if 1 then 100% clients participate in ISSL
lr = 3e-4
# 解释：定义了一些变量，包括用户数量、训练的轮数（epochs）、客户端参与比例（frac）和学习率（lr）。

# import pdb; pdb.set_trace()

#=====================================================================================================
#                           Client-side Model definition
#=====================================================================================================
# Model at client side

 
 
net_glob_client = VGG16_client_side()# 创建客户端模型实例
if torch.cuda.device_count() > 1:# 如果有多个 GPU
    print("We use",torch.cuda.device_count(), "GPUs")# 打印可用 GPU 数量
    net_glob_client = nn.DataParallel(net_glob_client)    # 使用多个 GPU 进行并行计算
#net_glob_client = nn.DataParallel(net_glob_client)具体做了以下事情：
#它将输入的模型net_glob_client作为参数传递给nn.DataParallel构造函数。
#nn.DataParallel创建了一个并行模型对象，该对象可以在多个 GPU 上同时处理数据。
#然后，这个并行模型对象被赋值回net_glob_client，替换了原来的单 GPU 模型。
net_glob_client.to(device)# 将模型移动到指定设备（GPU 或 CPU）
print(net_glob_client)# 打印客户端模型结构


net_glob_server = VGG16_server_side()
if torch.cuda.device_count() > 1:
    print("We use",torch.cuda.device_count(), "GPUs")
    net_glob_server = nn.DataParallel(net_glob_server)   # to use the multiple GPUs 

net_glob_server.to(device)
print(net_glob_server)      


criterion = nn.CrossEntropyLoss()#损失函数criterion衡量了模型预测的概率分布与真实标签之间的差异
# count1 = 0
# count2 = 0
#====================================================================================================
#                                  Server Side Program
#====================================================================================================
# Federated averaging: FedAvg
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])# 创建 w[0] 的深度拷贝作为初始平均权重 w_avg
    for k in w_avg.keys():# 遍历 w_avg 中的每个键（可能是神经网络各层的参数名等）
        for i in range(1, len(w)):# 从第二个元素开始遍历 w 列表中的每个元素（代表不同客户端的权重）
            w_avg[k] += w[i][k] # 将每个客户端的对应权重累加到 w_avg 中
        w_avg[k] = torch.div(w_avg[k], len(w))  # 计算平均权重，除以客户端数量 len(w)
    return w_avg


def calculate_accuracy(fx, y): #计算准确率，接收模型的输出 fx 和真实标签 y，通过比较预测值和真实标签计算出准确率并返回
    preds = fx.max(1, keepdim=True)[1]  # 获取 fx 在维度 1 上的最大值及对应的索引，keepdim=True 表示保持输出的维度与输入相同
    correct = preds.eq(y.view_as(preds)).sum()  # 计算预测值 preds 与真实标签 y（转换为与 preds 相同的形状）相等的数量总和
    acc = 100.00 *correct.float()/preds.shape[0]    # 计算准确率，将正确预测的数量转换为浮点数除以预测总数再乘以 100
    return acc

# to print train - test together in each round-- these are made global
# acc_avg_all_user_train = 0
# loss_avg_all_user_train = 0
# loss_train_collect_user = []
# acc_train_collect_user = []
# loss_test_collect_user = []
# acc_test_collect_user = []

# w_glob_server = net_glob_server.state_dict()
# w_locals_server = []

#client idx collector
# idx_collect = []
# l_epoch_check = False
# fed_check = False
# Initialization of net_model_server and net_server (server-side model)
# net_model_server = [net_glob_server for i in range(num_users)]
# net_server = copy.deepcopy(net_model_server[0]).to(device)
#optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)


def evaluate(net_glob_client, net_glob_server, dataset_test, iter): #用于评估模型在测试集上的性能
    with torch.no_grad():
        correct = 0
        samples = 0
        test_loader = DataLoader(dataset_test, batch_size=512, shuffle=True)
        # 创建一个数据加载器用于加载测试集数据，批量大小为 512，随机打乱数据
        for idx, (images, labels) in enumerate(test_loader):
            #enumerate(test_loader)函数用于遍历数据加载器，并为每个批次返回一个索引（从 0 开始）和对应的批次数据。
            #在每次循环迭代中，idx表示当前批次的索引，(images, labels)是一个包含当前批次图像数据和对应的标签数据的元组。
            # 将图像移动到指定设备上
            images = images.to(device=device)
            # 将标签移动到指定设备上
            labels = labels.to(device=device)
            # 首先使用客户端模型对图像进行处理，然后将结果传递给服务器端模型进行处理
            outputs = net_glob_server(net_glob_client(images))
           # 获取输出的最大值索引作为预测结果
            _, preds = outputs.max(1)
            # 将正确预测的数量累加到 correct 变量中
            correct += (preds == labels).sum()
            # 将总样本数量累加到 samples 变量中
            samples += preds.size(0)
        # 打印准确率和正确预测的数量与总样本数量的信息
        print(f"accuracy {float(correct) / float(samples) * 100:.2f} percentage || Correct {correct} out of {samples} samples")
        # 将准确率添加到记录器中，以便可视化，'acc/test'是记录的名称，iter 可能是迭代次数
        writer.add_scalar('acc/test', float(correct) / float(samples), iter)

#==============================================================================================================
#                                       Clients-side Program
#==============================================================================================================
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset 
        self.idxs = list(idxs)  

    def __len__(self):
        return len(self.idxs) 

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

# Client-side functions associated with Training and Testing
class Client(object):
    def __init__(self, idx, device, dataset_train = None, dataset_test = None, idxs = None, idxs_test = None):
        self.idx = idx
        self.local_ep = 1
        self.device = device
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size = 64, shuffle = True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size = 512, shuffle = True)
        

    def train(self, net, net_server, opt_stat, opt_stat_server):
        net.train()
        net_server.train()
        optimizer_client = torch.optim.Adam(net.parameters())
        optimizer_client.load_state_dict(opt_stat)
        optimizer_server = torch.optim.Adam(net_server.parameters())
        optimizer_server.load_state_dict(opt_stat_server)

        
        for iter in range(self.local_ep):
            loss_train = 0
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                optimizer_server.zero_grad()
                #---------forward prop-------------
                fx = net(images)
                activations = fx.clone().detach().requires_grad_(True)
                
                # Sending activations to server and receiving gradients from server
                scores = net_server(activations)    
                loss = criterion(scores,labels)
                loss.backward()
                loss_train += loss.item()
                activations_gradients = activations.grad.clone().detach()
                optimizer_server.step()
                
                #--------backward prop -------------

                fx.backward(activations_gradients)
                optimizer_client.step()
                
                if batch_idx%64==0:
                    print(f'Device {self.idx} || Local Epoch [{iter+1}/{self.local_ep}] || Step [{batch_idx+1}/{len(self.ldr_train)}] || Loss:{loss_train/(batch_idx+1)}')
            print(f"Loss at local epoch {iter+1} || {loss_train/(batch_idx+1)}")
           
        return net.state_dict(), net_server.state_dict(), optimizer_client.state_dict(), optimizer_server.state_dict(), loss_train/(batch_idx+1)
#=====================================================================================================
# dataset_iid() will create a dictionary to collect the indices of the data samples randomly for each client
def dataset_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def dataset_noniid(dataset_train, dataset_test, num_users):
    NUM_CLASS = 10
    ALPHA = 0.1
    prop = torch.tensor(np.random.dirichlet(np.ones(10) * ALPHA))
    users_shift = []
    for i in range(num_users):
        users_shift.append(i%10)
    random.shuffle(users_shift)

    all_idxs_per_class_train, all_idxs_per_class_test = [], []
    for j in range(NUM_CLASS):
        all_idxs_per_class_train.append([i for i in range(len(dataset_train)) if dataset_train[i][1]==j])
        all_idxs_per_class_test.append([i for i in range(len(dataset_test)) if dataset_test[i][1]==j]) 
    num_all_per_class_train = [len(all_idxs_per_class_train[i]) for i in range(NUM_CLASS)]
    num_all_per_class_test = [len(all_idxs_per_class_test[i]) for i in range(NUM_CLASS)]
    dict_users_train, dict_users_test = {}, {}

    for i in range(num_users):
        user_prop = torch.roll(prop, users_shift[i])
        dict_users_train[i], dict_users_test[i] = set(), set()
        for j in range(NUM_CLASS):
            num_take_train = int(num_all_per_class_train[j] / (num_users / NUM_CLASS) * user_prop[j])
            num_take_test = int(num_all_per_class_test[j] / (num_users / NUM_CLASS) * user_prop[j])
            dict_user_per_class_train = set(np.random.choice(all_idxs_per_class_train[j], num_take_train, replace = False))
            dict_user_per_class_test = set(np.random.choice(all_idxs_per_class_test[j], num_take_test, replace = False))
            dict_users_train[i].update(dict_user_per_class_train)
            dict_users_test[i].update(dict_user_per_class_test)
            all_idxs_per_class_train[j] = list(set(all_idxs_per_class_train[j]) - dict_user_per_class_train)
            all_idxs_per_class_test[j] = list(set(all_idxs_per_class_test[j]) - dict_user_per_class_test)

    return dict_users_train, dict_users_test
                          
#=============================================================================
#                         Data loading 
#============================================================================= 
# CIFAR10set = torchvision.datasets.CIFAR10(root='./dataset',download=True)

#==============================================================
# Custom dataset prepration in Pytorch format
# class CIFAR10Data(Dataset):
#     def __init__(self, datalist, transform = None):
        
#         self.datalist = datalist
#         self.transform = transform
        
#     def __len__(self):
        
#         return len(self.datalist)
    
#     def __getitem__(self, index):
        
#         X = self.datalist[index][0].resize((32, 32))
#         y = torch.tensor(int(self.datalist[index][1]))
        
#         if self.transform:
#             X = self.transform(X)
        
#         return X, y
#=============================================================================
# Train-test split          
# train, test = train_test_split(CIFAR10set, test_size = 0.2) # shuffled

#=============================================================================
#                         Data preprocessing
#=============================================================================  
# Data preprocessing: Transformation 

tranform_train = transforms.Compose([transforms.Resize((64,64)), transforms.RandomHorizontalFlip(p=0.7), transforms.Pad(3),transforms.RandomRotation(10),transforms.CenterCrop(64),transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
tranform_test = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# With augmentation
dataset_train = torchvision.datasets.EuroSAT("dataset/", download=True, transform=tranform_train) 
dataset_test = torchvision.datasets.EuroSAT("dataset/",  download=True, transform=tranform_test) 
# import pdb; pdb.set_trace()

#----------------------------------------------------------------
dict_users = dataset_iid(dataset_train, num_users)
dict_users_test = dataset_iid(dataset_test, num_users)
# dict_users, dict_users_test = dataset_noniid(dataset_train, dataset_test, num_users)

#------------ Training And Testing  -----------------
writer = SummaryWriter()
    
net_glob_client.train()
#copy weights
# w_glob_client = net_glob_client.state_dict()

# Federation takes place after certain local epochs in train() client-side
# this epoch is global epoch, also known as rounds

optimizer_client = torch.optim.Adam(net_glob_client.parameters(), lr = lr)
optimizer_status_client = [optimizer_client.state_dict() for _ in range(num_users)]
optimizer_server = torch.optim.Adam(net_glob_server.parameters(), lr = lr)
optimizer_status_server = [optimizer_server.state_dict() for _ in range(num_users)]

# import pdb; pdb.set_trace()


# train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
# test_loader = DataLoader(dataset_test, batch_size=512, shuffle=True)

train_loader = DataLoader(DatasetSplit(dataset_train, dict_users[0]), batch_size=64, shuffle=True)
test_loader = DataLoader(DatasetSplit(dataset_test, dict_users_test[0]), batch_size=512, shuffle=True)

for iter in range(epochs):
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace = False)
    w_locals_client = []
    w_locals_server = []
    loss_collect = []
    
    for idx in idxs_users:
        local = Client(idx, device, dataset_train = dataset_train, dataset_test = dataset_test, idxs = dict_users[idx], idxs_test = dict_users_test[idx])

        # Training ------------------
        w_client, w_server, optimizer_status_client[idx], optimizer_status_server[idx], loss = local.train(copy.deepcopy(net_glob_client).to(device), copy.deepcopy(net_glob_server).to(device), optimizer_status_client[idx], optimizer_status_server[idx])
        
        w_locals_client.append(copy.deepcopy(w_client))
        w_locals_server.append(copy.deepcopy(w_server))
        loss_collect.append(loss)
        # w_locals_client.append(copy.deepcopy(w_client))

    writer.add_scalar('loss/train', sum(loss_collect)/len(loss_collect), iter)
    
    w_glob_client = FedAvg(w_locals_client)
    w_glob_server = FedAvg(w_locals_server)  
    
    net_glob_client.load_state_dict(copy.deepcopy(w_glob_client))
    net_glob_server.load_state_dict(copy.deepcopy(w_glob_server))

    evaluate(net_glob_client, net_glob_server, dataset_test, iter)
        
       
    # Ater serving all clients for its local epochs------------
    # Fed  Server: Federation process at Client-Side-----------
    # print("-----------------------------------------------------------")
    # print("------ FedServer: Federation process at Client-Side ------- ")
    # print("-----------------------------------------------------------")
    # w_glob_client = FedAvg(w_locals_client)  

    
    # Update client-side global model 
    # net_glob_client.load_state_dict(w_glob_client) 
    
    
#===================================================================================     

print("Training and Evaluation completed!")    

#===============================================================================
# Save output data to .excel file (we use for comparision plots)
# round_process = [i for i in range(1, len(acc_train_collect)+1)]
# df = DataFrame({'round': round_process,'acc_train':acc_train_collect, 'acc_test':acc_test_collect})     
# file_name = program+".xlsx" 
# df.to_excel(file_name, sheet_name= "v1_test", index = False)     

#=============================================================================
#                         Program Completed
#=============================================================================