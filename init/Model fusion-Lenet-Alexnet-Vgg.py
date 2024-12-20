import torch
import torchvision
import torchvision.models
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

#  数据预处理
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(120),  # # 随机裁剪并调整到120x120
                                 transforms.RandomHorizontalFlip(),  # # 随机水平翻转
                                 transforms.ToTensor(),              # # 将图像从PIL格式转换为PyTorch中的Tensor格式
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),  # # 正则化，均值和标准差为0.5
    "val": transforms.Compose([transforms.Resize((120, 120)),  # 将验证集的所有图像缩放至120x120，不进行随机裁剪或翻转，以保持一致性。
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

def main():
    train_data = torchvision.datasets.ImageFolder(root = "./data/train" ,   transform = data_transform["train"])



    traindata = DataLoader(dataset= train_data , batch_size= 32 , shuffle= True , num_workers=0 )


    test_data = torchvision.datasets.ImageFolder(root = "./data/val" , transform = data_transform["val"])

    train_size = len(train_data)  #求出训练集的长度
    test_size = len(test_data)   #求出测试集的长度
    print(train_size)
    print(test_size)
    testdata = DataLoader(dataset = test_data , batch_size= 32 , shuffle= True , num_workers=0 )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  #如果有GPU就使用GPU，否则使用CPU
    print("using {} device.".format(device))


    class alexnet(nn.Module):      #alexnet神经网络  ，因为我的数据集是7种，因此如果你替换成自己的数据集，需要将这里的种类改成自己的
        def __init__(self):
            super(alexnet , self).__init__()
            self.model = nn.Sequential(

                nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 120, 120]  output[48, 55, 55]
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27]
                nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27]
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
                nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
                nn.Flatten(),
                nn.Dropout(p=0.5),
                nn.Linear(512, 2048),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(2048, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 5), #这里的7要修改成自己数据集的种类

            )
        def forward(self , x):
            x = self.model(x)
            return x

    class lenet(nn.Module): #Lenet神经网络
        def __init__(self):
            super(lenet , self).__init__()
            self.model = nn.Sequential(

                nn.Conv2d(3, 16,  kernel_size=5),  # input[3, 120, 120]  output[48, 55, 55]
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # output[48, 27, 27]

                nn.Conv2d(16, 32, kernel_size=5),  # output[128, 27, 27]
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # output[128, 13, 13]
                nn.Flatten(),
                nn.Linear(23328, 2048),
                nn.Linear(2048, 2048),
                nn.Linear(2048,5), #这里的7要修改成自己数据集的种类

            )
        def forward(self , x):
            x = self.model(x)
            return x

    class VGG(nn.Module):           #VGG神经网络
        def __init__(self, features, num_classes=5, init_weights=False):  #这里的7要修改成自己数据集的种类
            super(VGG, self).__init__()
            self.features = features
            self.classifier = nn.Sequential(
                nn.Linear(4608, 4096),
                nn.ReLU(True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, num_classes)
            )
            if init_weights:
                self._initialize_weights()  # 参数初始化

        def forward(self, x):
            # N x 3 x 224 x 224
            x = self.features(x)
            # N x 512 x 7 x 7
            x = torch.flatten(x, start_dim=1)
            # N x 512*7*7
            x = self.classifier(x)
            return x

        def _initialize_weights(self):
            for m in self.modules():  # 遍历各个层进行参数初始化
                if isinstance(m, nn.Conv2d):  # 如果是卷积层的话 进行下方初始化
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.xavier_uniform_(m.weight)  # 正态分布初始化
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)  # 如果偏置不是0 将偏置置成0  相当于对偏置进行初始化
                elif isinstance(m, nn.Linear):  # 如果是全连接层
                    nn.init.xavier_uniform_(m.weight)  # 也进行正态分布初始化
                    # nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)  # 将所有偏执置为0

    def make_features(cfg: list):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(True)]
                in_channels = v
        return nn.Sequential(*layers)

    cfgs = {
        'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                  'M'],
    }

    def vgg(model_name="vgg16", **kwargs):
        assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
        cfg = cfgs[model_name]

        model = VGG(make_features(cfg), **kwargs)
        return model



    VGGnet = vgg(num_classes=5, init_weights=True)    #这里的7要修改成自己数据集的种类

    lenet1 = lenet()

    alexnet1 = alexnet()


    mlps = [lenet1.to(device), alexnet1.to(device), VGGnet.to(device)] #建立一个数组，将三个模型放入

    epoch  = 1 #训练轮数
    LR = 0.0001  #学习率，我这里对于三个模型设置的是一样的学习率，事实上根据模型的不同设置成不一样的效果最好
    a = [{"params": mlp.parameters()} for mlp in mlps]  #依次读取三个模型的权重
    optimizer = torch.optim.Adam(a, lr=LR) #建立优化器
    loss_function = nn.CrossEntropyLoss()  #构建损失函数

    # 用于记录训练和测试过程中的损失和准确率，包括每个单独模型和融合模型（集成模型）

    train_loss_all = [[] , [] , []]
    train_accur_all = [[], [], []]
    ronghe_train_loss = []  #融合模型训练集的损失
    ronghe_train_accuracy = []  #融合模型训练集的准确率


    test_loss_all = [[] , [] , []]
    test_accur_all = [[] , [] , []]
    ronghe_test_loss = []  #融合模型测试集的损失
    ronghe_test_accuracy = [] #融合模型测试集的准确


    # 训练和测试过程
    
    for i in range(epoch): #遍历开始进行训练
        train_loss = [0 , 0 , 0]  #因为三个模型，初始化三个0存放模型的结果

        train_accuracy = [0.0 , 0.0 , 0.0]  #同上初始化三个0，存放模型的准确率
        for mlp in range(len(mlps)):
            mlps[mlp].train()    #遍历三个模型进行训练
        train_bar = tqdm(traindata)  #构建进度条，训练的时候有个进度条显示

        pre1 = [] #融合模型的损失
        vote1_correct = 0 #融合模型的准确率
        for step , data in enumerate(train_bar):  #遍历训练集

            img , target = data
            length = img.size(0)
            img, target = img.to(device), target.to(device)

            optimizer.zero_grad()

            for mlp in range(len(mlps)):  #对三个模型依次进行训练
                mlps[mlp].train()
                outputs = mlps[mlp](img)
                loss1 = loss_function(outputs, target)  # 求损失
                outputs = torch.argmax(outputs, 1)

                loss1.backward()#反向传播
                train_loss[mlp] += abs(loss1.item()) * img.size(0)
                accuracy = torch.sum(outputs == target)

                pre_num1 = outputs.cpu().numpy()
                # print(pre_num1.shape)
                train_accuracy[mlp] = train_accuracy[mlp] + accuracy

                pre1.append(pre_num1)

            # 融合模型的预测
            arr1 = np.array(pre1)
            pre1.clear()  # 将pre进行清空
            result1 = [Counter(arr1[:, i]).most_common(1)[0][0] for i in range(length)]  # 对于每张图片，统计三个模型其中，预测的那种情况最多，就取最多的值为融合模型预测的结果，即为投票
            #投票的意思，因为是三个模型，取结果最多的
            vote1_correct += (result1 == target.cpu().numpy()).sum()

            optimizer.step()  # 更新梯度

        losshe= 0
        # 输出每个模型和融合模型的训练结果
        for mlp in range(len(mlps)):
            print("epoch："+ str(i+1) , "模型" + str(mlp) + "的损失和准确率为：", "train-Loss：{} , train-accuracy：{}".format(train_loss[mlp]/train_size , train_accuracy[mlp]/train_size))
            train_loss_all[mlp].append(train_loss[mlp]/train_size)
            train_accur_all[mlp].append(train_accuracy[mlp].double().item()/train_size)
            losshe +=  train_loss[mlp]/train_size
        losshe /= 3
        print("epoch: " + str(i+1) + "集成模型训练集的正确率" + str(vote1_correct/train_size))
        print("epoch: " + str(i+1) + "集成模型训练集的损失" + str(losshe))
        ronghe_train_loss.append(losshe)
        ronghe_train_accuracy.append(vote1_correct/train_size)

        # 开始测试
        test_loss = [0 , 0 , 0]
        test_accuracy = [0.0 , 0.0 , 0.0]
        # 在测试时，所有模型的eval()模式用于禁用dropout等操作。
        for mlp in range(len(mlps)):
            mlps[mlp].eval()
        with torch.no_grad():
            pre = []
            vote_correct = 0
            test_bar = tqdm(testdata)
            vote_correct = 0
            for data in test_bar:

                img, target = data
                length1 = img.size(0)

                img, target = img.to(device), target.to(device)

                for mlp in range(len(mlps)):
                    outputs = mlps[mlp](img)

                    loss2 = loss_function(outputs, target)
                    outputs = torch.argmax(outputs, 1)


                    test_loss[mlp] += abs(loss2.item())*img.size(0)

                    accuracy = torch.sum(outputs == target)
                    pre_num = outputs.cpu().numpy()

                    test_accuracy[mlp] += accuracy

                    pre.append(pre_num)
                arr = np.array(pre)
                pre.clear()  # 将pre进行清空
                result = [Counter(arr[:, i]).most_common(1)[0][0] for i in range(length1)]  # 对于每张图片，统计三个模型其中，预测的那种情况最多，就取最多的值为融合模型预测的结果，
                vote_correct += (result == target.cpu().numpy()).sum()

        losshe1 = 0
        for mlp in range(len(mlps)):
            print("epoch："+ str(i+1), "模型" + str(mlp) + "的损失和准确率为：", "test-Loss：{} , test-accuracy：{}".format(test_loss[mlp] / test_size , test_accuracy[mlp] / test_size ))
            test_loss_all[mlp].append(test_loss[mlp]/test_size)
            test_accur_all[mlp].append(test_accuracy[mlp].double().item()/test_size )
            losshe1 += test_loss[mlp]/test_size
        losshe1 /= 3
        print("epoch: " + str(i+1) + "集成模型测试集的正确率" + str(vote_correct / test_size ))
        print("epoch: " + str(i+1) + "集成模型测试集的损失" + str(losshe1))
        ronghe_test_loss.append(losshe1)
        ronghe_test_accuracy.append(vote_correct/ test_size)

    #  绘制训练和测试结果
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    # for mlp in range(len(mlps)):
    plt.plot(range(epoch) , ronghe_train_loss,
                 "ro-",label = "Train loss")
    plt.plot(range(epoch), ronghe_test_loss,
                 "bs-",label = "test loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(range(epoch) , ronghe_train_accuracy,
                 "ro-",label = "Train accur")
    plt.plot(range(epoch) , ronghe_test_accuracy,
                 "bs-",label = "test accur")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()



    torch.save(alexnet1.state_dict(), "alexnet.pth")
    torch.save(lenet1.state_dict(), "lenet1.pth")
    torch.save(VGGnet.state_dict(), "VGGnet.pth")

    print("模型已保存")

if __name__ == '__main__':
    main()












