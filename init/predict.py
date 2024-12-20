import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import transforms
from collections import Counter
image_path = "./data/prt/2/LIDC-IDRI-0543_5_0094.jpg" #相对路径 导入图片
trans = transforms.Compose([transforms.Resize((120 , 120)),
                           transforms.ToTensor()])   #将图片缩放为跟训练集图片的大小一样 方便预测，且将图片转换为张量
image = Image.open(image_path)  #打开图片
#print(image)  #输出图片 看看图片格式
image = image.convert("RGB")  #将图片转换为RGB格式
image = trans(image)   #上述的缩放和转张量操作在这里实现
#print(image)   #查看转换后的样子
image = torch.unsqueeze(image, dim=0)  #将图片维度扩展一维

classes = ["1" , "2" ]  #预测种类


class alexnet(nn.Module):  # alexnet神经网络  ，因为我的数据集是7种，因此如果你替换成自己的数据集，需要将这里的种类改成自己的
    def __init__(self):
        super(alexnet, self).__init__()
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
            nn.Linear(1024, 2),  # 这里的7要修改成自己数据集的种类

        )

    def forward(self, x):
        x = self.model(x)
        return x


class lenet(nn.Module):  # Lenet神经网络
    def __init__(self):
        super(lenet, self).__init__()
        self.model = nn.Sequential(

            nn.Conv2d(3, 16, kernel_size=5),  # input[3, 120, 120]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output[48, 27, 27]

            nn.Conv2d(16, 32, kernel_size=5),  # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output[128, 13, 13]
            nn.Flatten(),
            nn.Linear(23328, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2),  # 这里的7要修改成自己数据集的种类

        )

    def forward(self, x):
        x = self.model(x)
        return x



class VGG(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=False):
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
            self._initialize_weights()   #参数初始化

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():         #遍历各个层进行参数初始化
            if isinstance(m, nn.Conv2d):   #如果是卷积层的话 进行下方初始化
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)  #正态分布初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)     #如果偏置不是0 将偏置置成0  相当于对偏置进行初始化
            elif isinstance(m, nn.Linear):        #如果是全连接层
                nn.init.xavier_uniform_(m.weight)    #也进行正态分布初始化
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)  #将所有偏执置为0


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
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)
    return model
#以上是神经网络结构，因为读取了模型之后代码还得知道神经网络的结构才能进行预测
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  #将代码放入GPU进行训练
print("using {} device.".format(device))

VGGnet = vgg(num_classes=2, init_weights=True)  # 这里的7要修改成自己数据集的种类

lenet1 = lenet()

alexnet1 = alexnet()

VGGnet.load_state_dict(torch.load("VGGnet.pth", map_location=device))#训练得到的VGGnet模型放入当前文件夹下
lenet1.load_state_dict(torch.load("lenet.pth", map_location=device))#训练得到的lenet模型放入当前文件夹下
alexnet1.load_state_dict(torch.load("alexnet.pth", map_location=device))#训练得到的alexnet模型放入当前文件夹下
mlps = [lenet1.to(device), alexnet1.to(device), VGGnet.to(device)] #建立一个数组，将三个模型放入

for mlp in range(len(mlps)):
    mlps[mlp].eval()#关闭梯度，将模型调整为测试模式

with torch.no_grad():  #梯度清零
    pre = []
    length1 = image.size(0)
    for mlp in range(len(mlps)):

        outputs = mlps[mlp](image.to(device))#将图片打入神经网络进行测试
        outputs = torch.argmax(outputs, 1)
        pre_num = outputs.cpu().numpy()
        pre.append(pre_num)
    arr = np.array(pre)
    print(arr)#查看三个模型输除的结果
    pre.clear()  # 将pre进行清空
    result = [Counter(arr[:, i]).most_common(1)[0][0] for i in range(length1)]  # 对于每张图片，统计三个模型其中，预测的那种情况最多，就取最多的值为融合模型预测的结果，
    print(result)  #选取最多的情况最为

    # 对应找其在种类中的序号即可然后输出即为其种类
    print(classes[result[0]-1])#因为他是输出的第几种情况，然后我们的列表是从0开始的因此，这里需要减一