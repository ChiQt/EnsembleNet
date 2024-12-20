"""
Created on Sat Dec 13 20:10:55 2024

@author: chiXJ

This script is revealing the setup and training and testing processes of our mixture model based on the idea of ensemble learning
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

# 配置部分
class Config:
    DATA_DIR = "./data"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "val")
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    NUM_CLASSES = 5  # 根据数据集的实际类别数进行修改
    EPOCHS = 10       # 进行一轮
    LEARNING_RATE = 0.0001
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SAVE_PATHS = {
        "alexnet": "./model/alexnet.pth",
        "lenet": "./model/lenet.pth",
        "vggnet": "./model/vggnet.pth"
    }

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
    def __init__(self, num_classes=5):
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
    def __init__(self, num_classes=5):
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
    def __init__(self, features, num_classes=5, init_weights=False):
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

# 训练函数
def train(models_list, optimizer, criterion, train_loader, device, config):
    models_train_loss = [0.0 for _ in models_list]
    models_train_accuracy = [0.0 for _ in models_list]
    ensemble_correct = 0
    ensemble_loss = 0.0

    for model in models_list:
        model.train()

    train_bar = tqdm(train_loader, desc="Training", leave=False)
    for imgs, targets in train_bar:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs_list = []
        losses = []
        predictions = []

        for model in models_list:
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            losses.append(loss.item())
            _, preds = torch.max(outputs, 1)
            predictions.append(preds.cpu().numpy())

        optimizer.step()

        # 更新每个模型的损失和准确率
        for idx, model in enumerate(models_list):
            models_train_loss[idx] += losses[idx] * imgs.size(0)
            models_train_accuracy[idx] += (predictions[idx] == targets.cpu().numpy()).sum()

        # 集成模型的预测（投票机制）
        predictions = np.array(predictions).T  # 转置以便按样本处理
        ensemble_preds = [Counter(sample).most_common(1)[0][0] for sample in predictions]
        ensemble_correct += (np.array(ensemble_preds) == targets.cpu().numpy()).sum()
        ensemble_loss += sum(losses) / len(losses) * imgs.size(0)  # 平均损失

        # 更新进度条
        current_loss = ensemble_loss / ((train_bar.n + 1) * config.BATCH_SIZE)
        current_acc = ensemble_correct / ((train_bar.n + 1) * config.BATCH_SIZE)
        train_bar.set_postfix({
            "Ensemble Acc": f"{current_acc:.4f}",
            "Ensemble Loss": f"{current_loss:.4f}"
        })

    avg_loss = [loss / config.TRAIN_SIZE for loss in models_train_loss]
    avg_acc = [acc / config.TRAIN_SIZE for acc in models_train_accuracy]
    ensemble_avg_loss = ensemble_loss / config.TRAIN_SIZE
    ensemble_avg_acc = ensemble_correct / config.TRAIN_SIZE

    return avg_loss, avg_acc, ensemble_avg_loss, ensemble_avg_acc

# 评估函数
def evaluate(models_list, criterion, val_loader, device, config):
    models_test_loss = [0.0 for _ in models_list]
    models_test_accuracy = [0.0 for _ in models_list]
    ensemble_correct = 0
    ensemble_loss = 0.0

    for model in models_list:
        model.eval()

    with torch.no_grad():
        val_bar = tqdm(val_loader, desc="Validation", leave=False)
        for imgs, targets in val_bar:
            imgs, targets = imgs.to(device), targets.to(device)

            outputs_list = []
            losses = []
            predictions = []

            for model in models_list:
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                losses.append(loss.item())
                _, preds = torch.max(outputs, 1)
                predictions.append(preds.cpu().numpy())

            # 更新每个模型的损失和准确率
            for idx, model in enumerate(models_list):
                models_test_loss[idx] += losses[idx] * imgs.size(0)
                models_test_accuracy[idx] += (predictions[idx] == targets.cpu().numpy()).sum()

            # 集成模型的预测（投票机制）
            predictions = np.array(predictions).T  # 转置以便按样本处理
            ensemble_preds = [Counter(sample).most_common(1)[0][0] for sample in predictions]
            ensemble_correct += (np.array(ensemble_preds) == targets.cpu().numpy()).sum()
            ensemble_loss += sum(losses) / len(losses) * imgs.size(0)  # 平均损失

            # 更新进度条
            current_loss = ensemble_loss / ((val_bar.n + 1) * config.BATCH_SIZE)
            current_acc = ensemble_correct / ((val_bar.n + 1) * config.BATCH_SIZE)
            val_bar.set_postfix({
                "Ensemble Acc": f"{current_acc:.4f}",
                "Ensemble Loss": f"{current_loss:.4f}"
            })

    avg_loss = [loss / config.VAL_SIZE for loss in models_test_loss]
    avg_acc = [acc / config.VAL_SIZE for acc in models_test_accuracy]
    ensemble_avg_loss = ensemble_loss / config.VAL_SIZE
    ensemble_avg_acc = ensemble_correct / config.VAL_SIZE

    return avg_loss, avg_acc, ensemble_avg_loss, ensemble_avg_acc

# 训练与评估管理部分
def train_and_evaluate(config, train_loader, val_loader, train_size, val_size):
    # 初始化模型
    alexnet = AlexNet(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    lenet = LeNet(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    vggnet = build_vgg(model_name="vgg16", num_classes=config.NUM_CLASSES, init_weights=True).to(config.DEVICE)

    models_list = [lenet, alexnet, vggnet]

    # 优化器和损失函数
    optimizer = torch.optim.Adam([{"params": model.parameters()} for model in models_list], lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 记录变量
    history = {
        "train_loss_all": [[] for _ in models_list],
        "train_accur_all": [[] for _ in models_list],
        "ensemble_train_loss": [],
        "ensemble_train_accuracy": [],
        "test_loss_all": [[] for _ in models_list],
        "test_accur_all": [[] for _ in models_list],
        "ensemble_test_loss": [],
        "ensemble_test_accuracy": []
    }

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.EPOCHS}")
        print("-" * 30)

        # 训练阶段
        train_loss, train_accuracy, ensemble_train_loss, ensemble_train_acc = train(
            models_list, optimizer, criterion, train_loader, config.DEVICE, config
        )

        # 记录和打印训练结果
        for idx, model in enumerate(models_list):
            history["train_loss_all"][idx].append(train_loss[idx])
            history["train_accur_all"][idx].append(train_accuracy[idx])
            print(f"Model {idx + 1} - Train Loss: {train_loss[idx]:.4f}, Train Acc: {train_accuracy[idx]:.4f}")

        history["ensemble_train_loss"].append(ensemble_train_loss)
        history["ensemble_train_accuracy"].append(ensemble_train_acc)
        print(f"Ensemble - Train Loss: {ensemble_train_loss:.4f}, Train Acc: {ensemble_train_acc:.4f}")

        # 验证阶段
        test_loss, test_accuracy, ensemble_test_loss, ensemble_test_acc = evaluate(
            models_list, criterion, val_loader, config.DEVICE, config
        )

        # 记录和打印验证结果
        for idx, model in enumerate(models_list):
            history["test_loss_all"][idx].append(test_loss[idx])
            history["test_accur_all"][idx].append(test_accuracy[idx])
            print(f"Model {idx + 1} - Test Loss: {test_loss[idx]:.4f}, Test Acc: {test_accuracy[idx]:.4f}")

        history["ensemble_test_loss"].append(ensemble_test_loss)
        history["ensemble_test_accuracy"].append(ensemble_test_acc)
        print(f"Ensemble - Test Loss: {ensemble_test_loss:.4f}, Test Acc: {ensemble_test_acc:.4f}")

    # 绘制训练和测试结果
    plot_history(history, config.EPOCHS)

    # 保存模型
    save_models(models_list, config.SAVE_PATHS)
    print("模型已保存")

# 绘图函数
def plot_history(history, epochs):
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history["ensemble_train_loss"], "ro-", label="Train Loss")
    plt.plot(range(1, epochs + 1), history["ensemble_test_loss"], "bs-", label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history["ensemble_train_accuracy"], "ro-", label="Train Acc")
    plt.plot(range(1, epochs + 1), history["ensemble_test_accuracy"], "bs-", label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.show()

# 保存模型函数
def save_models(models, save_paths):
    model_names = ["lenet", "alexnet", "vggnet"]
    for model, name in zip(models, model_names):
        path = save_paths.get(name, f"{name}.pth")
        torch.save(model.state_dict(), path)

# 主函数部分
def main():
    config = Config()
    print(f"Using device: {config.DEVICE}")

    train_loader, val_loader, train_size, val_size = get_data_loaders(config)
    config.TRAIN_SIZE = train_size
    config.VAL_SIZE = val_size
    print(f"训练集大小: {train_size}")
    print(f"验证集大小: {val_size}")

    train_and_evaluate(config, train_loader, val_loader, train_size, val_size)

if __name__ == '__main__':
    main()
