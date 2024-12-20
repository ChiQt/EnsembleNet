'''
Created on Sat Dec 12 8:10:20 2024

@author: chiXJ

Topic: predict with three methods

Caculate prediction indexs based the file structure (identify the classes)
'''
#%%
import numpy as np
import torch
from PIL import Image
from torch import nn
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from collections import Counter
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

#%% 配置部分
class Config:
    IMAGE_DIR = "./data/prt"  # 存放待预测图片的文件夹
    TRANSFORM = transforms.Compose([
            transforms.RandomResizedCrop(120),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    CLASSES = [ "1","2"]  # 修改为您的类别名称
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL_PATHS = {
        "alexnet": "./model/alexnet_voting2.pth",
        "lenet": "./model/lenet_voting2.pth",
        "vggnet": "./model/vggnet_voting2.pth"
    }
    NUM_CLASSES = 2  # 根据您的数据集类别数修改

# # 数据处理部分
# def get_data_loaders(config):
#     data_transforms = {
#         "prt": transforms.Compose([
#             transforms.RandomResizedCrop(120),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ]),
#     }

#     prt_dataset = torchvision.datasets.ImageFolder(
#         root=config.VAL_DIR,
#         transform=data_transforms["prt"]
#     )

#     prt_loader = DataLoader(
#         dataset=prt_dataset,
#         batch_size=config.BATCH_SIZE,
#         shuffle=False,
#         num_workers=config.NUM_WORKERS
#     )

#     return prt_loader, len(prt_dataset)

# 其他模型定义（AlexNet, LeNet, VGG等）保持不变

#%% 定义神经网络模型
class AlexNet(nn.Module):
    def __init__(self, num_classes=7):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # [3, 120, 120] -> [48, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # [48, 27, 27] -> [48, 13, 13]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # [48, 13, 13] -> [128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # [128, 13, 13] -> [128, 6, 6]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # [128, 6, 6] -> [192, 6, 6]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # [192, 6, 6] -> [192, 6, 6]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # [192, 6, 6] -> [128, 6, 6]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # [128, 6, 6] -> [128, 2, 2]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(128 * 2 * 2, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class LeNet(nn.Module):
    def __init__(self, num_classes=7):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),  # [3, 120, 120] -> [16, 116, 116]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [16, 116, 116] -> [16, 58, 58]
            nn.Conv2d(16, 32, kernel_size=5),  # [16, 58, 58] -> [32, 54, 54]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # [32, 54, 54] -> [32, 27, 27]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 27 * 27, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class VGG(nn.Module):
    def __init__(self, features, num_classes=7, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 4096),  # 根据特征图大小调整
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

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
    assert model_name in cfgs, f"Warning: model number {model_name} not in cfgs dict!"
    cfg = cfgs[model_name]
    model = VGG(make_features(cfg), **kwargs)
    return model

# 加载模型并准备预测
def load_models(config):
    # 初始化模型
    VGGnet = vgg(model_name="vgg16", num_classes=config.NUM_CLASSES, init_weights=True)
    lenet1 = LeNet(num_classes=config.NUM_CLASSES)
    alexnet1 = AlexNet(num_classes=config.NUM_CLASSES)

    # 加载预训练模型权重
    VGGnet.load_state_dict(torch.load(config.MODEL_PATHS["vggnet"], map_location=config.DEVICE))
    lenet1.load_state_dict(torch.load(config.MODEL_PATHS["lenet"], map_location=config.DEVICE))
    alexnet1.load_state_dict(torch.load(config.MODEL_PATHS["alexnet"], map_location=config.DEVICE))
    
    # 打印模型结构
    print('VGGnet:\n',VGGnet)
    print('lenet:\n',lenet1)
    print('alexnet:\n',alexnet1)

    # 将模型移动到设备并设置为评估模式
    models = [lenet1.to(config.DEVICE), alexnet1.to(config.DEVICE), VGGnet.to(config.DEVICE)]
    for model in models:
        model.eval()

    return models
#%% 预测函数
# 预测单张图片
def predict_image(image_path, config, models):
    image = Image.open(image_path)
    image = image.convert("RGB")
    image = config.TRANSFORM(image)
    image = torch.unsqueeze(image, dim=0).to(config.DEVICE)

    with torch.no_grad():
        pre = []
        for model in models:
            model.eval()
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            pre_num = preds.cpu().numpy()
            pre.append(pre_num)
        arr = np.array(pre)
        # print('三个模型输出的结果:',arr)#查看三个基分类器输出的结果, ★ 输出的类别从 0 开始，后面计算准确率需要给类别+1
        # result = [Counter(arr[:, i]).most_common(1)[0][0] for i in range(len(arr[0]))]  # 对于每张图片，统计三个模型其中，预测的那种情况最多，就取最多的值为集成模型预测的结果
    return arr    # 直接输出一个array 如[[1][0][0]]（基分类器的结果）


# 预测多个图片--**1.多数投票**
def predict_images(image_dir, config, models):
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    all_results = []
    all_targets = []

    # 根据文件夹的类别进行预测
    for label in os.listdir(image_dir):
        label_dir = os.path.join(image_dir, label)
        if os.path.isdir(label_dir):
            for img_file in tqdm(os.listdir(label_dir), desc=f"Predicting for class {label}"):
                img_path = os.path.join(label_dir, img_file)

                # 进行多数投票
                arr = predict_image(img_path, config, models)+1  # ★ 输出的类别从 0 开始，如[[1][0][0]]（基分类器的结果）->[[2][1][1]]
                result = [Counter(arr[:, i]).most_common(1)[0][0] for i in range(len(arr[0]))] 

                all_results.append(result)
                all_targets.append(int(label))  # 标签即文件夹名称

    # 融合预测结果
    fused_results = [Counter(arr).most_common(1)[0][0] for arr in all_results]   

    # 计算评估指标
    accuracy = accuracy_score(all_targets, fused_results)
    precision = precision_score(all_targets, fused_results, average='weighted', zero_division=0)
    recall = recall_score(all_targets, fused_results, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, fused_results, average='weighted', zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 绘制混淆矩阵
    cm = confusion_matrix(all_targets, fused_results)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=config.CLASSES, yticklabels=config.CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Ensembel Model 1: Confusion Matrix')
    plt.show()

    # # 打印或返回预测结果
    # for img_file, pred in zip(image_files, fused_results):
    #     print(f"{img_file}: {config.CLASSES[pred-1]}")  # 假设pred从1开始

    return fused_results

## **2. 修改集成预测逻辑以使用加权投票**

# 预测多个图片 - 加权投票方式
def predict_images_weighted(image_dir, config, models, model_weights):
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    all_results = []
    all_targets = []  # 不再依赖 ground_truth，而是通过文件夹结构推断标签

    # 根据文件夹的类别进行预测
    for label in os.listdir(image_dir):
        label_dir = os.path.join(image_dir, label)
        if os.path.isdir(label_dir):
            for img_file in tqdm(os.listdir(label_dir), desc=f"Predicting for class {label}"):
                img_path = os.path.join(label_dir, img_file)
                all_targets.append(int(label))  # 使用文件夹名称作为真实标签

                # 获取该图片的每个基模型的预测结果
                result = predict_image(img_path, config, models)+1  # ★ 输出的类别从 0 开始，如[[1][0][0]]（基分类器的结果）->[[2][1][1]]
                all_results.append(result)

    # 加权投票方式：根据模型输出的类别进行加权
    fused_results = []
    for preds in all_results:
        class_scores = {}
    # 遍历每个预测结果和对应的权重
        for pred, weight in zip(preds.flatten(), model_weights):
            # 更新类别的累计得分
            class_scores[pred] = class_scores.get(pred, 0) + weight
            # 选择得分最高的类别作为最终预测
        fused_pred = max(class_scores.items(), key=lambda x: x[1])[0]
        fused_results.append(fused_pred)

    # 计算评估指标
    accuracy = accuracy_score(all_targets, fused_results)
    precision = precision_score(all_targets, fused_results, average='weighted', zero_division=0)
    recall = recall_score(all_targets, fused_results, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, fused_results, average='weighted', zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 绘制混淆矩阵
    cm = confusion_matrix(all_targets, fused_results)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=config.CLASSES, yticklabels=config.CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Ensembel Model 2:Confusion Matrix')
    plt.show()

    # # 打印或返回预测结果
    # for img_file, pred in zip(image_files, fused_results):
    #     print(f"{img_file}: {config.CLASSES[pred-1]}")  # 假设pred从1开始

    return fused_results


## **3.堆叠模型预测 **
# 加载堆叠模型
def load_meta_model():
    bst = xgb.Booster()
    bst.load_model("./model/xgboost_meta.model2")
    return bst

# 预测函数 - Stacking方式
def predict_images_stacking(image_dir, config, models, meta_model):
    image_files = []
    all_results = []
    all_targets = []

    # 根据文件夹结构推断类别
    for label in os.listdir(image_dir):
        label_dir = os.path.join(image_dir, label)
        if os.path.isdir(label_dir):
            for img_file in tqdm(os.listdir(label_dir), desc=f"Predicting for class {label}"):
                img_path = os.path.join(label_dir, img_file)
                image_files.append(img_file)
                all_targets.append(int(label))  # 文件夹名作为真实标签

                # 生成基分类器的预测
                preds = []
                for model in models:
                    model.eval()
                    with torch.no_grad():
                        image = Image.open(img_path).convert("RGB")
                        image = config.TRANSFORM(image)
                        image = torch.unsqueeze(image, dim=0).to(config.DEVICE)
                        output = model(image)
                        _, pred = torch.max(output, 1)
                        preds.append(pred.cpu().numpy()[0])
                all_results.append(preds)

    # 转换为堆叠模型的输入格式
    meta_X = np.array(all_results)
    dmatrix = xgb.DMatrix(meta_X)

    # 使用堆叠模型进行最终预测
    final_preds = meta_model.predict(dmatrix)+1  # ★ 输出的类别从 0 开始

    # 计算评估指标
    accuracy = accuracy_score(all_targets, final_preds)
    precision = precision_score(all_targets, final_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, final_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, final_preds, average='weighted', zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 绘制混淆矩阵
    cm = confusion_matrix(all_targets, final_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=config.CLASSES, yticklabels=config.CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Ensembel Model 3:Confusion Matrix')
    plt.show()

    # 打印预测结果
    # for img_file, pred in zip(image_files, final_preds):
    #     pred = int(pred)  # 转换为整数类型
    #     print(f"{img_file}: {config.CLASSES[pred-1]}")  # 假设pred从1开始

    return final_preds

#%% 主函数
def main():
    config = Config()
    models = load_models(config)

    # 方法一：多数投票法：预测并计算准确率等指标
    predict_images(config.IMAGE_DIR, config, models)

    # 方法二：加权投票法
    config.MODEL_PATHS = {
        "alexnet": "alexnet_weighted.pth",
        "lenet": "lenet_weighted.pth",
        "vggnet": "vggnet_weighted.pth"
    }
    # 读取模型权重
    model_weights_path= './history/model_weights.pth'
    model_weights = torch.load(model_weights_path)
    print(f"模型权重{model_weights}已成功加载")
    print("Predicting using weighted voting method:")
    predict_images_weighted(config.IMAGE_DIR, config, models, model_weights)

    # 方法三：堆叠 (Stacking) 方法
    config.MODEL_PATHS = {
        "alexnet": "./model/alexnet_xgboost2.pth",
        "lenet": "./model/lenet_xgboost2.pth",
        "vggnet": "./model/vggnet_xgboost2.pth"
    }
    meta_model = load_meta_model()    
    print("Predicting using Stacking method:")
    predict_images_stacking(config.IMAGE_DIR, config, models, meta_model)


if __name__ == '__main__':
    main()

#%% 绘制三种基模型和三种集成模型的混淆矩阵和AUC曲线