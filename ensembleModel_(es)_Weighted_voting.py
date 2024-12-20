"""
Created on Sat Dec 16 8:10:20 2024

@author: chiXJ

Topic: EnsembleModel_weighted_voting

改动如下：
在EnsembleModel_voting现有的投票法基础上进行改进，加权投票，以体现集成学习的优势

目标：
1.强化性能“好”的分类器，弱化性能“差”的分类器：根据每个分类器在验证集上的表现，为其分配不同的权重，性能越好的分类器权重越大。

实现步骤：
1.计算每个模型的权重：根据每个模型在验证集上的准确率，为其分配权重(由train_and_evaluate返回)
2.加权投票：在集成预测时，根据模型权重对预测结果进行加权，最终选择得分最高的类别(对predict_images进行修改)
"""
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
    EPOCHS = 100  # 增加训练轮数以充分展示早停效果
    LEARNING_RATE = 0.0001
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SAVE_PATHS = {
        "alexnet": "alexnet_weighted.pth",
        "lenet": "lenet_weighted.pth",
        "vggnet": "vggnet_weighted.pth"
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

# 模型定义部分
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

        早停机制的目的是为了防止模型在训练过程中过拟合，同时节省计算资源。为了实现这一目标，选择合适的监控指标至关重要。
            通常，早停机制应基于验证集（Validation Set）的性能指标，而不是训练集（Training Set）的损失。
        
        原因如下：
            训练损失：持续优化训练损失会导致模型在训练集上表现越来越好，但可能在未见过的数据（验证集或测试集）上表现变差，这是过拟合的表现。
            验证损失或验证指标：通过监控验证集上的损失或其他性能指标，可以更好地判断模型在未见数据上的泛化能力。当验证损失不再下降或验证指标（如准确率）不再提升时，说明模型可能已经达到了最佳的泛化性能，此时停止训练可以避免过拟合。
        
        最科学的方法是：
            监控验证损失（Validation Loss）：当验证损失在连续若干个epoch中不再下降时，触发早停。
            或监控验证指标（如验证准确率 Validation Accuracy）：当验证准确率在连续若干个epoch中不再提升时，触发早停。

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


# 训练函数
def train(models_list, optimizers, criterion, train_loader, device, config):
    models_train_loss = [0.0 for _ in models_list]
    models_train_accuracy = [0.0 for _ in models_list]
    all_targets = []
    all_preds = []

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

        # 对每个样本进行投票
        for sample_preds in batch_preds:
            ensemble_pred = Counter(sample_preds).most_common(1)[0][0]
            all_preds.append(ensemble_pred)
        all_targets.extend(targets.cpu().numpy())

    # 计算平均损失和准确率
    avg_loss = [loss / config.TRAIN_SIZE for loss in models_train_loss]
    avg_acc = [acc / config.TRAIN_SIZE for acc in models_train_accuracy]

    # 计算集成模型的评估指标
    ensemble_acc = accuracy_score(all_targets, all_preds)
    ensemble_precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    ensemble_recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    ensemble_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

    return avg_loss, avg_acc, ensemble_acc, ensemble_precision, ensemble_recall, ensemble_f1

# 评估函数
def evaluate(models_list, criterion, val_loader, device, config):
    models_test_loss = [0.0 for _ in models_list]
    models_test_accuracy = [0.0 for _ in models_list]
    all_targets = []
    all_preds = []

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

            # 对每个样本进行投票
            for sample_preds in batch_preds:
                ensemble_pred = Counter(sample_preds).most_common(1)[0][0]
                all_preds.append(ensemble_pred)
            all_targets.extend(targets.cpu().numpy())

    # 计算平均损失和准确率
    avg_loss = [loss / config.VAL_SIZE for loss in models_test_loss]
    avg_acc = [acc / config.VAL_SIZE for acc in models_test_accuracy]

    # 计算集成模型的评估指标
    ensemble_acc = accuracy_score(all_targets, all_preds)
    ensemble_precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    ensemble_recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    ensemble_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

    return avg_loss, avg_acc, ensemble_acc, ensemble_precision, ensemble_recall, ensemble_f1

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
        "ensemble_train_accuracy": [],
        "ensemble_train_precision": [],
        "ensemble_train_recall": [],
        "ensemble_train_f1": [],
        "test_loss_all": [[] for _ in models_list],
        "test_accur_all": [[] for _ in models_list],
        "ensemble_test_accuracy": [],
        "ensemble_test_precision": [],
        "ensemble_test_recall": [],
        "ensemble_test_f1": []
    }

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.EPOCHS}")
        print("-" * 30)

        # 训练阶段
        train_loss, train_accuracy, ensemble_train_acc, ensemble_train_precision, ensemble_train_recall, ensemble_train_f1 = train(
            models_list, optimizers, criterion, train_loader, config.DEVICE, config
        )

        # 记录和打印训练结果
        for idx, model in enumerate(models_list):
            history["train_loss_all"][idx].append(train_loss[idx])
            history["train_accur_all"][idx].append(train_accuracy[idx])
            print(f"Model {idx + 1} - Train Loss: {train_loss[idx]:.4f}, Train Acc: {train_accuracy[idx]:.4f}")

        history["ensemble_train_accuracy"].append(ensemble_train_acc)
        history["ensemble_train_precision"].append(ensemble_train_precision)
        history["ensemble_train_recall"].append(ensemble_train_recall)
        history["ensemble_train_f1"].append(ensemble_train_f1)
        print(f"Ensemble - Train Acc: {ensemble_train_acc:.4f}")
        print(f"Ensemble - Train Precision: {ensemble_train_precision:.4f}, Train Recall: {ensemble_train_recall:.4f}, Train F1: {ensemble_train_f1:.4f}")

        # 验证阶段
        test_loss, test_accuracy, ensemble_test_acc, ensemble_test_precision, ensemble_test_recall, ensemble_test_f1 = evaluate(
            models_list, criterion, val_loader, config.DEVICE, config
        )

        # 记录和打印验证结果
        for idx, model in enumerate(models_list):
            history["test_loss_all"][idx].append(test_loss[idx])
            history["test_accur_all"][idx].append(test_accuracy[idx])
            print(f"Model {idx + 1} - Test Loss: {test_loss[idx]:.4f}, Test Acc: {test_accuracy[idx]:.4f}")

        history["ensemble_test_accuracy"].append(ensemble_test_acc)
        history["ensemble_test_precision"].append(ensemble_test_precision)
        history["ensemble_test_recall"].append(ensemble_test_recall)
        history["ensemble_test_f1"].append(ensemble_test_f1)
        print(f"Ensemble - Test Acc: {ensemble_test_acc:.4f}")
        print(f"Ensemble - Test Precision: {ensemble_test_precision:.4f}, Test Recall: {ensemble_test_recall:.4f}, Test F1: {ensemble_test_f1:.4f}")

        # 早停机制检测
        early_stopping(ensemble_test_acc)  # 使用验证准确率进行早停判断
        if early_stopping.early_stop:
            print("早停触发，停止训练。")
            break

    # 计算模型权重（基于验证集上的准确率）
    val_accuracies = [acc[-1] for acc in history["test_accur_all"]]
    total_acc = sum(val_accuracies)
    model_weights = [acc / total_acc for acc in val_accuracies]
    print(f"Model Weights: {model_weights}")

    # 保存模型和权重
    save_models(models_list, config.SAVE_PATHS)
    print("模型已保存")

    # 返回模型权重
    return history, model_weights


# 绘图函数
def plot_history(history, epochs):
    plt.figure(figsize=(14, 10))

    # 损失曲线
    plt.subplot(3, 2, 1)
    for idx, model_name in enumerate(["lenet", "alexnet", "vggnet"]):
        plt.plot(range(1, epochs + 1), history["train_loss_all"][idx], label=f"{model_name} Train Loss")
        plt.plot(range(1, epochs + 1), history["test_loss_all"][idx], label=f"{model_name} Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    # 准确率曲线
    plt.subplot(3, 2, 2)
    for idx, model_name in enumerate(["lenet", "alexnet", "vggnet"]):
        plt.plot(range(1, epochs + 1), history["train_accur_all"][idx], label=f"{model_name} Train Acc")
        plt.plot(range(1, epochs + 1), history["test_accur_all"][idx], label=f"{model_name} Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    # Ensemble 准确率曲线
    plt.subplot(3, 2, 3)
    plt.plot(range(1, epochs + 1), history["ensemble_train_accuracy"], "ro-", label="Train Accuracy")
    plt.plot(range(1, epochs + 1), history["ensemble_test_accuracy"], "bs-", label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Ensemble Accuracy Curve")
    plt.legend()

    # Ensemble 精确率曲线
    plt.subplot(3, 2, 4)
    plt.plot(range(1, epochs + 1), history["ensemble_train_precision"], "go-", label="Train Precision")
    plt.plot(range(1, epochs + 1), history["ensemble_test_precision"], "ms-", label="Test Precision")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Ensemble Precision Curve")
    plt.legend()

    # Ensemble 召回率和 F1 分数曲线
    plt.subplot(3, 2, 5)
    plt.plot(range(1, epochs + 1), history["ensemble_train_recall"], "c^-", label="Train Recall")
    plt.plot(range(1, epochs + 1), history["ensemble_test_recall"], "kv-", label="Test Recall")
    plt.plot(range(1, epochs + 1), history["ensemble_train_f1"], "y*-", label="Train F1")
    plt.plot(range(1, epochs + 1), history["ensemble_test_f1"], "b+-", label="Test F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Ensemble Recall and F1 Score Curve")
    plt.legend()

    # 隐藏第六个子图以保持布局整齐
    plt.subplot(3, 2, 6)
    plt.axis('off')

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

    history, model_weights = train_and_evaluate(config, train_loader, val_loader, train_size, val_size)

    # 绘制训练和测试结果
    plot_history(history, config.EPOCHS)

    # 确保目录存在，存储训练和测试的历史记录
    os.makedirs(r'.\history', exist_ok=True)
    history_file_path = r'.\history\history_weighted.json'
    
    save_history(history, history_file_path)
    print("训练和验证历史已保存")

    model_weights_path= '.\history\model_weights_weighted.pth'
    torch.save(model_weights, model_weights_path)
    print(f"模型权重:{model_weights}已保存到 {model_weights_path}")

    # 读取历史记录，绘制训练和测试结果
    # read_history(history_file_path)
    # plot_history(history, config.EPOCHS)

    # 读取模型权重
    loaded_weights = torch.load(model_weights_path)
    print(f"模型权重{loaded_weights}已成功加载")


if __name__ == '__main__':
    main()

#%%
history_file_path = r'.\history\history_weighted.json'
# 读取历史记录，绘制训练和测试结果
history = read_history(history_file_path)
epochnum  = len(history["ensemble_train_recall"])
plot_history(history, epochnum)