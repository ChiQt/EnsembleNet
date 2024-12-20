import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import transforms
from collections import Counter
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from tqdm import tqdm
import matplotlib.pyplot as plt

# 配置部分
class Config:
    IMAGE_DIR = "./images"  # 存放待预测图片的文件夹
    TRANSFORM = transforms.Compose([
        transforms.Resize((120, 120)),
        transforms.ToTensor(),
    ])
    CLASSES = ["1", "2", "3", "4", "5"]  # 修改为您的类别名称
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL_PATHS = {
        "alexnet": ".\\file\\alexnet.pth",
        "lenet": ".\\file\\lenet1.pth",
        "vggnet": ".\\file\\VGGnet.pth"
    }
    NUM_CLASSES = 7  # 根据您的数据集类别数修改
    EARLY_STOPPING_PATIENCE = 30  # 早停耐心
    LEARNING_RATE = 0.0001
    EPOCHS = 50
    STACKING_INTERVAL = 10  # 每隔多少个epoch收集一次预测数据
    STACKING_MODEL_PATH = ".\\file\\xgboost_meta.model"  # 堆叠模型保存路径
    SAVE_PATHS = {
        "alexnet": ".\\file\\alexnet.pth",
        "lenet": ".\\file\\lenet1.pth",
        "vggnet": ".\\file\\VGGnet.pth"
    }

# 定义神经网络模型 (AlexNet, LeNet, VGG 同上)
class AlexNet(nn.Module):
    def __init__(self, num_classes=7):
        super(AlexNet, self).__init__()
        self.model = nn.Sequential(
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
            nn.MaxPool2d(kernel_size=3, stride=2),  # [128, 6, 6] -> [128, 2, 2]
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(128 * 2 * 2, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        return x

class LeNet(nn.Module):
    def __init__(self, num_classes=7):
        super(LeNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),  # [3, 120, 120] -> [16, 116, 116]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [16, 116, 116] -> [16, 58, 58]
            nn.Conv2d(16, 32, kernel_size=5),  # [16, 58, 58] -> [32, 54, 54]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [32, 54, 54] -> [32, 27, 27]
            nn.Flatten(),
            nn.Linear(32 * 27 * 27, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        return x

class VGG(nn.Module):
    def __init__(self, features, num_classes=7, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2 * 2, 4096),  # 根据特征图大小调整
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

# 早停机制类
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
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_metric = score
            self.counter = 0

# 定义堆叠数据收集函数
def collect_stacking_data(models_list, val_loader, config):
    stacking_preds = []
    stacking_targets = []

    for model in models_list:
        model.eval()
        preds = []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(config.DEVICE)
                outputs = model(imgs)
                _, pred = torch.max(outputs, 1)
                preds.extend(pred.cpu().numpy())
        stacking_preds.append(preds)
    
    # 转置为 [num_samples, num_models]
    stacking_preds = np.array(stacking_preds).T
    # 收集真实标签（假设所有模型的预测都是针对同一批次的标签）
    stacking_targets = []
    for _, targets in val_loader:
        stacking_targets.extend(targets.numpy())
    
    return stacking_preds, stacking_targets

# 修改训练函数
def train(models_list, optimizers, criterion, train_loader, device, config):
    models_train_loss = [0.0 for _ in models_list]
    models_train_accuracy = [0.0 for _ in models_list]
    all_preds = []
    all_targets = []

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

# 修改评估函数
def evaluate(models_list, criterion, val_loader, device, config):
    models_test_loss = [0.0 for _ in models_list]
    models_test_accuracy = [0.0 for _ in models_list]
    all_preds = []
    all_targets = []

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

def train_and_evaluate(config, train_loader, val_loader, train_size, val_size):
    # 初始化模型
    alexnet = AlexNet(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    lenet = LeNet(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    vggnet = vgg(model_name="vgg16", num_classes=config.NUM_CLASSES, init_weights=True).to(config.DEVICE)

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
        "ensemble_test_f1": [],
        # 新增，用于堆叠模型
        "stacking_meta_features": [],  # 堆叠模型的特征
        "stacking_meta_targets": []     # 堆叠模型的目标
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

        # 每隔 STACKING_INTERVAL 个epoch，收集基分类器在验证集上的预测结果
        if epoch % config.STACKING_INTERVAL == 0:
            stacking_meta_features, stacking_meta_targets = collect_stacking_data(models_list, val_loader, config)
            history["stacking_meta_features"].append(stacking_meta_features)
            history["stacking_meta_targets"].append(stacking_meta_targets)
            print(f"Collected stacking data at epoch {epoch}.")

        # 早停机制检测
        early_stopping(ensemble_test_acc)  # 使用验证准确率进行早停判断
        if early_stopping.early_stop:
            print("早停触发，停止训练。")
            break

    # 将所有收集到的堆叠数据合并
    if history["stacking_meta_features"]:
        meta_X = np.vstack(history["stacking_meta_features"])  # 合并特征
        meta_y = np.hstack(history["stacking_meta_targets"])    # 合并目标

        # 训练 XGBoost 作为元分类器
        dtrain = xgb.DMatrix(meta_X, label=meta_y)
        params = {
            'objective': 'multi:softmax',
            'num_class': config.NUM_CLASSES,
            'eval_metric': 'mlogloss'
        }
        num_round = 100
        bst = xgb.train(params, dtrain, num_round)

        # 评估 XGBoost 元分类器
        meta_preds = bst.predict(dtrain)
        meta_accuracy = accuracy_score(meta_y, meta_preds)
        meta_precision = precision_score(meta_y, meta_preds, average='weighted', zero_division=0)
        meta_recall = recall_score(meta_y, meta_preds, average='weighted', zero_division=0)
        meta_f1 = f1_score(meta_y, meta_preds, average='weighted', zero_division=0)

        print(f"Meta-Classifier - Accuracy: {meta_accuracy:.4f}")
        print(f"Meta-Classifier - Precision: {meta_precision:.4f}, Recall: {meta_recall:.4f}, F1 Score: {meta_f1:.4f}")

        # 将元分类器的评估指标添加到 history 中
        history["meta_accuracy"] = meta_accuracy
        history["meta_precision"] = meta_precision
        history["meta_recall"] = meta_recall
        history["meta_f1"] = meta_f1

        # 保存 XGBoost 元分类器
        os.makedirs(os.path.dirname(config.STACKING_MODEL_PATH), exist_ok=True)
        bst.save_model(config.STACKING_MODEL_PATH)
        print(f"XGBoost 元分类器已保存到 {config.STACKING_MODEL_PATH}")
    else:
        print("没有收集到堆叠数据，跳过堆叠模型训练。")

    # 绘制训练和测试结果
    plot_history(history, config.EPOCHS)

    # 保存基分类器模型
    save_models(models_list, config.SAVE_PATHS)
    print("基分类器模型已保存")

    # 返回堆叠模型的评估指标
    return history

#%%
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

    # 将模型移动到设备并设置为评估模式
    models = [lenet1.to(config.DEVICE), alexnet1.to(config.DEVICE), VGGnet.to(config.DEVICE)]
    for model in models:
        model.eval()

    return models

# 预测单张图片
def predict_image(image_path, config, models):
    image = Image.open(image_path)
    image = image.convert("RGB")
    image = config.TRANSFORM(image)
    image = torch.unsqueeze(image, dim=0).to(config.DEVICE)

    with torch.no_grad():
        pre = []
        for model in models:
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            pre.append(preds.cpu().numpy()[0])
        # 加权投票
        if hasattr(config, 'model_weights'):
            model_weights = config.model_weights
            class_scores = {}
            for pred, weight in zip(pre, model_weights):
                class_scores[pred] = class_scores.get(pred, 0) + weight
            fused_pred = max(class_scores.items(), key=lambda x: x[1])[0]
        else:
            # 如果没有权重，则使用多数投票
            fused_pred = Counter(pre).most_common(1)[0][0]
    return fused_pred

# 预测多个图片
def predict_images(image_dir, config, models, model_weights=None, meta_model=None, ground_truth=None):
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    all_results = []
    all_targets = [] if ground_truth else None

    for img_file in tqdm(image_files, desc="Predicting"):
        img_path = os.path.join(image_dir, img_file)
        if ground_truth and img_file in ground_truth:
            all_targets.append(ground_truth[img_file])

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

    # 使用元分类器进行最终预测
    if meta_model:
        meta_X = np.array(all_results)
        dmatrix = xgb.DMatrix(meta_X)
        final_preds = meta_model.predict(dmatrix)
    else:
        # 使用加权投票或多数投票
        if model_weights:
            final_preds = []
            for preds in all_results:
                class_scores = {}
                for pred, weight in zip(preds, model_weights):
                    class_scores[pred] = class_scores.get(pred, 0) + weight
                fused_pred = max(class_scores.items(), key=lambda x: x[1])[0]
                final_preds.append(fused_pred)
        else:
            # 多数投票
            final_preds = [Counter(preds).most_common(1)[0][0] for preds in all_results]

    # 如果有真实标签，计算评估指标
    if ground_truth:
        if meta_model:
            y_true = np.array(all_targets)
            y_pred = final_preds
        else:
            y_true = np.array(all_targets)
            y_pred = final_preds

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

    # 打印或返回预测结果
    for img_file, pred in zip(image_files, final_preds):
        print(f"{img_file}: {config.CLASSES[int(pred)-1]}")  # 假设pred从1开始

    return final_preds

# 加载堆叠模型
def load_meta_model(config):
    bst = xgb.Booster()
    bst.load_model(config.STACKING_MODEL_PATH)
    return bst

# 绘图函数 (同上)
def plot_history(history, epochs):
    plt.figure(figsize=(18, 12))  # 增大画布尺寸以容纳更多子图

    # 1. 损失曲线
    plt.subplot(3, 2, 1)
    for idx, model_name in enumerate(["Model 1", "Model 2", "Model 3"]):
        plt.plot(range(1, epochs + 1), history["train_loss_all"][idx], label=f"{model_name} Train Loss")
        plt.plot(range(1, epochs + 1), history["test_loss_all"][idx], label=f"{model_name} Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    # 2. 准确率曲线
    plt.subplot(3, 2, 2)
    for idx, model_name in enumerate(["Model 1", "Model 2", "Model 3"]):
        plt.plot(range(1, epochs + 1), history["train_accur_all"][idx], label=f"{model_name} Train Acc")
        plt.plot(range(1, epochs + 1), history["test_accur_all"][idx], label=f"{model_name} Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    # 3. Ensemble 准确率曲线
    plt.subplot(3, 2, 3)
    plt.plot(range(1, epochs + 1), history["ensemble_train_accuracy"], "ro-", label="Ensemble Train Acc")
    plt.plot(range(1, epochs + 1), history["ensemble_test_accuracy"], "bs-", label="Ensemble Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Ensemble Accuracy Curve")
    plt.legend()

    # 4. Ensemble 精确率曲线
    plt.subplot(3, 2, 4)
    plt.plot(range(1, epochs + 1), history["ensemble_train_precision"], "go-", label="Ensemble Train Precision")
    plt.plot(range(1, epochs + 1), history["ensemble_test_precision"], "ms-", label="Ensemble Test Precision")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Ensemble Precision Curve")
    plt.legend()

    # 5. Ensemble 召回率和 F1 分数曲线
    plt.subplot(3, 2, 5)
    plt.plot(range(1, epochs + 1), history["ensemble_train_recall"], "c^-", label="Ensemble Train Recall")
    plt.plot(range(1, epochs + 1), history["ensemble_test_recall"], "kv-", label="Ensemble Test Recall")
    plt.plot(range(1, epochs + 1), history["ensemble_train_f1"], "y*-", label="Ensemble Train F1")
    plt.plot(range(1, epochs + 1), history["ensemble_test_f1"], "b+-", label="Ensemble Test F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Ensemble Recall and F1 Score Curve")
    plt.legend()

    # 6. 元分类器（XGBoost）性能
    plt.subplot(3, 2, 6)
    if "meta_accuracy" in history:
        metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
        meta_scores = [
            history.get("meta_accuracy", 0),
            history.get("meta_precision", 0),
            history.get("meta_recall", 0),
            history.get("meta_f1", 0)
        ]
        plt.bar(metrics, meta_scores, color=['blue', 'green', 'red', 'cyan'])
        plt.ylim(0, 1)
        for i, v in enumerate(meta_scores):
            plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
        plt.xlabel("Metrics")
        plt.ylabel("Score")
        plt.title("Meta-Classifier (XGBoost) Performance")
    else:
        plt.text(0.5, 0.5, "No Meta-Classifier Data", horizontalalignment='center', verticalalignment='center')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    config = Config()
    models = load_models(config)
    
    # 定义训练和验证数据加载器
    # 这里需要根据您的数据集定义DataLoader
    # 例如：
    # train_dataset = YourTrainDataset(...)
    # val_dataset = YourValDataset(...)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 示例（请根据实际情况替换）
    train_loader = ...  # 定义您的训练数据加载器
    val_loader = ...    # 定义您的验证数据加载器
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    
    # 方法一：加权投票
    history = train_and_evaluate(config, train_loader, val_loader, train_size, val_size)
    
    # 加载元分类器
    if "meta_accuracy" in history:
        meta_model = load_meta_model(config)
    else:
        meta_model = None
    
    # 方法二：堆叠 (Stacking) 方法
    predict_images_stacking(config.IMAGE_DIR, config, models, meta_model, ground_truth=None)
    
    # 如果需要方法一的加权投票预测，可以调用如下：
    # predict_images(config.IMAGE_DIR, config, models, model_weights=config.model_weights, meta_model=None, ground_truth=None)
