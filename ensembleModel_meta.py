"""
Created on Sat Dec 16 20:59:20 2024

@author: chiXJ

Topic: EnsembleModel__meta 

堆叠（Stacking）与 XGBoost 作为元分类器
堆叠（Stacking）方法的关键思想是利用基分类器的预测结果作为新的特征来训练元分类器（如 XGBoost）
这里是用在所有epoch迭代更新结束后得到的3个基分类器对于验证集的预测结果来训练元分类器

目标：
1.加强错误分类样本的重要性：通过元分类器学习哪些基分类器在不同样本上表现较好。
2.强化性能“好”的分类器，弱化性能“差”的分类器：XGBoost 可以自动学习基分类器的权重和组合方式，以提升整体性能。

增加内容：
1.新增记录验证集基分类器预测结果：在每个 epoch 的验证阶段，收集基分类器在验证集上的预测结果，用于训练元分类器。
2.训练 XGBoost 元分类器：使用基分类器在验证集上的预测结果作为特征，真实标签作为目标，训练 XGBoost 作为元分类器。

实现步骤：
1.train basic classifiers 2.train Meta classifier

details:
1.生成基分类器的预测结果：使用验证集生成基分类器的预测。
2.准备元分类器的训练数据：基分类器的预测作为特征，真实标签作为目标。
3.训练 XGBoost 作为元分类器。
4.在测试阶段，基分类器生成预测，并由 XGBoost 生成最终预测。

与 Adaboost 的对比：
Adaboost每一轮都会根据前一轮分类器的错误情况调整权重，并引入新的分类器。
这是一种 逐步迭代 的过程，不断调整每个分类器的权重，直到达到收敛。Adaboost 更强调 逐步修正 错误样本的偏差，以提高整体模型的性能。

相比之下，堆叠方法的元分类器（如 XGBoost）不是逐步修正错误分类器的预测，而是将多个基分类器的预测汇总（通常是通过交叉验证，得到不同基分类器的输出），然后学习如何根据这些输出做出最终的预测。
它主要是通过融合多个模型的预测来提升性能，而不是通过反复调整每个基分类器的权重。
总结来说，堆叠方法训练 XGBoost 的优点是它利用验证集上的基分类器输出进行元学习，以更好地组合多个模型的优点，而不会对训练集产生过拟合。Adaboost 的优点是它能 动态地调整基分类器的权重，集中改善难分类的样本，这种方式有助于改进集成模型对误分类样本的处理。

XGBoost 的训练过程:
初始化：首先初始化一个简单的预测模型（通常是均值或者其他简单的值）。
计算残差：对于每个样本，计算当前模型的预测值与真实值之间的 残差。
基于残差构建决策树：使用这些残差作为目标值，通过决策树来拟合误差（残差）。
更新模型：将新学到的树加入到现有模型中，调整预测结果。
重复步骤 2 至步骤 4，直到达到最大迭代次数或误差收敛。

"""
import xgboost as xgb

import torch
import torchvision
import torchvision.models as models
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
import json


# 配置部分
class Config:
    DATA_DIR = "./data"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "val")
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    NUM_CLASSES = 2  # 根据数据集的实际类别数进行修改
    EPOCHS = 100 # 增加训练轮数以充分展示早停效果
    LEARNING_RATE = 0.0001
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SAVE_PATHS = {
        "alexnet": "alexnet_xgboost3.pth",
        "lenet": "lenet_xgboost3.pth",
        "vggnet": "vggnet_xgboost3.pth"
    }
    EARLY_STOPPING_PATIENCE = 24  # 早停的耐心轮数

# 数据处理部分
def get_data_loaders(config):
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(120),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([
            transforms.Resize((120, 120)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    train_dataset = torchvision.datasets.ImageFolder(
        root=config.TRAIN_DIR,
        transform=data_transforms["train"]
    )
    val_dataset = torchvision.datasets.ImageFolder(
        root=config.VAL_DIR,
        transform=data_transforms["val"]
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    return train_loader, val_loader, len(train_dataset), len(val_dataset)

#%% 模型定义部分
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
            nn.Linear(512 * 3 * 3, 4096),  # 根据输入尺寸调整
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

def make_vgg_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

VGG_CONFIGS = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def build_vgg(model_name="vgg16", num_classes=7, init_weights=True):
    assert model_name in VGG_CONFIGS, f"Model {model_name} not found in VGG_CONFIGS."
    cfg = VGG_CONFIGS[model_name]
    features = make_vgg_features(cfg)
    model = VGG(features, num_classes=num_classes, init_weights=init_weights)
    return model

# 早停机制类 metric 是 验证损失
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        """
        Args:
            patience (int): 在多少个epoch内验证指标不再提升时，停止训练
            verbose (bool): 是否打印信息
            delta (float): 提升的最小变化
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_metric = None

    def __call__(self, metric):
        score = metric

        if self.best_score is None:
            self.best_score = score
            self.best_metric = score
        elif score > self.best_score - self.delta:  # 对于损失，较低更好
            self.best_score = score
            self.best_metric = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

#%%
# 训练函数
def train(models_list, optimizers, criterion, train_loader, device, config):
    models_train_loss = [0.0 for _ in models_list]
    models_train_accuracy = [0.0 for _ in models_list]

    for model in models_list:
        model.train()

    for imgs, targets in tqdm(train_loader, desc="Training", leave=False):
        imgs, targets = imgs.to(device), targets.to(device)

        batch_preds = []
        for idx, model in enumerate(models_list):
            optimizers[idx].zero_grad()  # 清零对应模型的梯度
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizers[idx].step()

            # 记录损失和准确率
            models_train_loss[idx] += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            models_train_accuracy[idx] += (preds == targets).sum().item()

            batch_preds.append(preds.cpu().numpy())

        # batch_preds 是一个包含所有模型对当前批次的预测结果的列表，每个元素形状为 [batch_size]
        batch_preds = np.array(batch_preds).T  # 转置为 [batch_size, num_models]

    # 计算平均损失和准确率
    avg_loss = [loss / config.TRAIN_SIZE for loss in models_train_loss]
    avg_acc = [acc / config.TRAIN_SIZE for acc in models_train_accuracy]

    return avg_loss, avg_acc  

# 评估函数
def evaluate(models_list, criterion, val_loader, device, config):
    models_test_loss = [0.0 for _ in models_list]
    models_test_accuracy = [0.0 for _ in models_list]

    for model in models_list:
        model.eval()

    with torch.no_grad():
        for imgs, targets in tqdm(val_loader, desc="Validation", leave=False):
            imgs, targets = imgs.to(device), targets.to(device)

            batch_preds = []
            for idx, model in enumerate(models_list):
                outputs = model(imgs)
                loss = criterion(outputs, targets)

                # 记录损失和准确率
                models_test_loss[idx] += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                models_test_accuracy[idx] += (preds == targets).sum().item()

                batch_preds.append(preds.cpu().numpy())

            # batch_preds 是一个包含所有模型对当前批次的预测结果的列表，每个元素形状为 [batch_size]
            batch_preds = np.array(batch_preds).T  # 转置为 [batch_size, num_models]

    # 计算平均损失和准确率
    avg_loss = [loss / config.VAL_SIZE for loss in models_test_loss]
    avg_acc = [acc / config.VAL_SIZE for acc in models_test_accuracy]

    return avg_loss, avg_acc

# 训练与评估管理部分
def train_and_evaluate(config, train_loader, val_loader, train_size, val_size):
    # 初始化模型
    alexnet = AlexNet(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    lenet = LeNet(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    vggnet = build_vgg(model_name="vgg16", num_classes=config.NUM_CLASSES, init_weights=True).to(config.DEVICE)

    models_list = [lenet, alexnet, vggnet]

    # 为每个模型单独创建优化器
    optimizers = [
        torch.optim.Adam(lenet.parameters(), lr=config.LEARNING_RATE),
        torch.optim.Adam(alexnet.parameters(), lr=config.LEARNING_RATE),
        torch.optim.Adam(vggnet.parameters(), lr=config.LEARNING_RATE)
    ]
    criterion = nn.CrossEntropyLoss()

    # 早停机制
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, verbose=True)

    # 记录变量
    history = {
        "train_loss_all": [[] for _ in models_list],
        "train_accur_all": [[] for _ in models_list],
        "test_loss_all": [[] for _ in models_list],
        "test_accur_all": [[] for _ in models_list],
        "meta_accuracy": [],
        "meta_precision": [],
        "meta_recall": [],
        "meta_f1": [],
        "meta_accuracy_fin": 0,
        "meta_precision_fin": 0,
        "meta_recall_fin": 0,
        "meta_f1_fin": 0,      
        # 新增
    }
    meta_hsitory = {
        "val_meta_features": [],  # 验证集基分类器预测
        "val_targets": []
    }

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.EPOCHS}")
        print("-" * 30)

        # 训练阶段
        train_loss, train_accuracy = train(
            models_list, optimizers, criterion, train_loader, config.DEVICE, config
        )

        # 记录和打印训练结果
        for idx, model in enumerate(models_list):
            history["train_loss_all"][idx].append(train_loss[idx])
            history["train_accur_all"][idx].append(train_accuracy[idx])
            print(f"Model {idx + 1} - Train Loss: {train_loss[idx]:.4f}, Train Acc: {train_accuracy[idx]:.4f}")
        # 验证阶段
        test_loss, test_accuracy = evaluate(
            models_list, criterion, val_loader, config.DEVICE, config
        )

        # 记录和打印验证结果
        for idx, model in enumerate(models_list):
            history["test_loss_all"][idx].append(test_loss[idx])
            history["test_accur_all"][idx].append(test_accuracy[idx])
            print(f"Model {idx + 1} - Test Loss: {test_loss[idx]:.4f}, Test Acc: {test_accuracy[idx]:.4f}")
        # 收集验证集基分类器的预测结果，每个epoch的3个基分类器都会被更新
        if epoch % 1 == 0:
            val_preds = []
            for model in models_list:  # 对于每个基分类器（在 models_list 中的每个模型），首先将模型切换到评估模式（model.eval()）
                model.eval()
                with torch.no_grad():
                    preds = []
                    for imgs, _ in val_loader:
                        imgs = imgs.to(config.DEVICE)
                        outputs = model(imgs)
                        _, pred = torch.max(outputs, 1)
                        preds.extend(pred.cpu().numpy())
                    val_preds.append(preds)             
                # 在不计算梯度的情况下 (torch.no_grad())，对验证集 (val_loader) 中的每个图像进行预测，并将每个模型的预测结果收集到 val_preds 中
            # 转置为 [num_samples, num_models]  预测结果 pred 被存储为每个样本的预测类别，并附加到 val_preds 中
            val_preds = np.array(val_preds).T
            meta_hsitory["val_meta_features"] = val_preds
            meta_hsitory["val_targets"] = []

            # 收集验证集的真实标签
            for _, targets in val_loader:
                meta_hsitory["val_targets"].extend(targets.numpy())

            meta_X = meta_hsitory["val_meta_features"]       # 这表示堆叠模型的训练数据特征，即之前各基分类器的预测结果（这些是先前模型在验证集上的输出）
            meta_y = np.array(meta_hsitory["val_targets"])   # 这是与这些特征对应的真实标签，表示验证集的真实类别

            # 训练 XGBoost 作为元分类器
            dtrain = xgb.DMatrix(meta_X, label=meta_y)  #  Dmatrix: 转换成 XGBoost 可以处理的数据格式，即基分类器的预测结果和真实标签
            params = {
                'objective': 'multi:softmax',
                'num_class': config.NUM_CLASSES,        # 指定了一个多分类任务 (multi:softmax)，并设定了评估指标为 mlogloss
                'eval_metric': 'mlogloss'
            }
            num_round = 100
            bst = xgb.train(params, dtrain, num_round)

            # 评估 XGBoost 元分类器
            # 使用训练好的 XGBoost 模型对训练数据进行预测，并计算模型的评估指标（准确率、精确度、召回率、F1 分数）
            meta_preds = bst.predict(dtrain)   # 在 Dmatrix 的 meta_X 上进行预测，得到预测的类别标签 meta_preds
            meta_accuracy = accuracy_score(meta_y, meta_preds)
            meta_precision = precision_score(meta_y, meta_preds, average='weighted', zero_division=0)
            meta_recall = recall_score(meta_y, meta_preds, average='weighted', zero_division=0)
            meta_f1 = f1_score(meta_y, meta_preds, average='weighted', zero_division=0)

            print(f"Meta-Classifier - Accuracy: {meta_accuracy:.4f}")
            print(f"Meta-Classifier - Precision: {meta_precision:.4f}, Recall: {meta_recall:.4f}, F1 Score: {meta_f1:.4f}")

            # 将元分类器的评估指标添加到 history 中
            history["meta_accuracy"].append(meta_accuracy)
            history["meta_precision"].append(meta_precision)
            history["meta_recall"].append(meta_recall)
            history["meta_f1"].append(meta_f1)            

        # # 早停机制检测
        # early_stopping(ensemble_test_acc)  # 使用验证准确率进行早停判断
        # if early_stopping.early_stop:
        #     print("早停触发，停止训练。")
        #     break

    # 准备堆叠模型的训练数据：
    # 将3个基分类器的预测结果（val_meta_features）和验证集的真实标签（val_targets）作为训练数据来训练元分类器（stacked model）
    
    meta_X = meta_hsitory["val_meta_features"]       # 这表示堆叠模型的训练数据特征，即之前各基分类器的预测结果（这些是先前模型在验证集上的输出）
    meta_y = np.array(meta_hsitory["val_targets"])   # 这是与这些特征对应的真实标签，表示验证集的真实类别

    # 训练 XGBoost 作为元分类器
    dtrain = xgb.DMatrix(meta_X, label=meta_y)  #  Dmatrix: 转换成 XGBoost 可以处理的数据格式，即基分类器的预测结果和真实标签
    params = {
        'objective': 'multi:softmax',
        'num_class': config.NUM_CLASSES,        # 指定了一个多分类任务 (multi:softmax)，并设定了评估指标为 mlogloss
        'eval_metric': 'mlogloss'
    }
    num_round = 100
    bst = xgb.train(params, dtrain, num_round)

    # 评估 XGBoost 元分类器
    # 使用训练好的 XGBoost 模型对训练数据进行预测，并计算模型的评估指标（准确率、精确度、召回率、F1 分数）
    meta_preds = bst.predict(dtrain)   # 在 Dmatrix 的 meta_X 上进行预测，得到预测的类别标签 meta_preds
    meta_accuracy = accuracy_score(meta_y, meta_preds)
    meta_precision = precision_score(meta_y, meta_preds, average='weighted', zero_division=0)
    meta_recall = recall_score(meta_y, meta_preds, average='weighted', zero_division=0)
    meta_f1 = f1_score(meta_y, meta_preds, average='weighted', zero_division=0)

    print(f"Meta-Classifier - Accuracy: {meta_accuracy:.4f}")
    print(f"Meta-Classifier - Precision: {meta_precision:.4f}, Recall: {meta_recall:.4f}, F1 Score: {meta_f1:.4f}")

    # 将元分类器的评估指标添加到 history 中
    history["meta_accuracy_fin"] = meta_accuracy
    history["meta_precision_fin"] = meta_precision
    history["meta_recall_fin"] = meta_recall
    history["meta_f1_fin"] = meta_f1

    # 保存 XGBoost 模型
    bst.save_model("xgboost_meta.model3")
    print("XGBoost 元分类器已保存")

    # # 绘制训练和测试结果
    # plot_history(history, epoch)

    # 保存基分类器模型
    save_models(models_list, config.SAVE_PATHS)
    print("基分类器模型已保存")
    
    return history

# 绘图函数
def plot_history(history, epochs):
    plt.figure(figsize=(18, 12))  # 增大画布尺寸以容纳更多子图

    # 1. 损失曲线
    plt.subplot(1, 3, 1)
    for idx, model_name in enumerate(["lenet", "alexnet", "vggnet"]):
        plt.plot(range(1, epochs + 1), history["train_loss_all"][idx], label=f"{model_name} Train Loss")
        plt.plot(range(1, epochs + 1), history["test_loss_all"][idx], label=f"{model_name} Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    # 2. 准确率曲线
    plt.subplot(1, 3, 2)
    for idx, model_name in enumerate(["lenet", "alexnet", "vggnet"]):
        plt.plot(range(1, epochs + 1), history["train_accur_all"][idx], label=f"{model_name} Train Acc")
        plt.plot(range(1, epochs + 1), history["test_accur_all"][idx], label=f"{model_name} Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    # 6. 元分类器（XGBoost）性能
    plt.subplot(1, 3, 3)
    # 使用条形图展示元分类器的评估指标
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    meta_scores = [
        history.get("meta_accuracy_fin", 0),
        history.get("meta_precision_fin", 0),
        history.get("meta_recall_fin", 0),
        history.get("meta_f1_fin", 0)
    ]
    plt.bar(metrics, meta_scores, color=['blue', 'green', 'red', 'cyan'])
    plt.ylim(0, 1)
    for i, v in enumerate(meta_scores):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title("Meta-Classifier (XGBoost) Performance")

    plt.tight_layout()
    plt.show()

# 保存模型函数
def save_models(models, save_paths):
    model_names = ["lenet", "alexnet", "vggnet"]
    for model, name in zip(models, model_names):
        path = save_paths.get(name, f"{name}.pth")
        torch.save(model.state_dict(), path)

def save_history(history, history_file_path):
    with open(history_file_path, 'w') as f:
        json.dump(history, f, indent=4)

    print(f"History saved to {history_file_path}")

def read_history(history_file_path):
    with open(history_file_path, 'r') as f:
        '''with能省去最后close file的步骤'''
        loaded_history = json.load(f)
    return loaded_history

# 主函数部分
def main():
    config = Config()
    print(f"Using device: {config.DEVICE}")

    train_loader, val_loader, train_size, val_size = get_data_loaders(config)
    config.TRAIN_SIZE = train_size
    config.VAL_SIZE = val_size
    print(f"训练集大小: {train_size}")
    print(f"验证集大小: {val_size}")

    history= train_and_evaluate(config, train_loader, val_loader, train_size, val_size)

    # 确保目录存在，存储训练和测试的历史记录
    os.makedirs(r'.\history', exist_ok=True)
    history_file_path = r'.\history\history_stacking3.json'
    
    save_history(history, history_file_path)
    print("训练和验证历史已保存")

    # 读取历史记录，绘制训练和测试结果
    # read_history(history_file_path)
    # plot_history(history, config.EPOCHS)

    
    # 绘制训练和测试结果
    plot_history(history, config.EPOCHS)


if __name__ == '__main__':
    main()

#%%
history_file_path = r'.\history\history_stacking3.json'
# 读取历史记录，绘制训练和测试结果
history = read_history(history_file_path)
epochnum  = len(history["train_loss_all"][0])
plot_history(history, epochnum)
# %%
