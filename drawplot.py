"""
Created on Sat Dec 19 1:10:20 2024

@author: chiXJ

This script is for drawing the training and validation process of three basic classifiers(loss and accuarcy) and three bensembel classifiers(indexs: accuarcy, precision, recall, f1).
"""
#%%
import matplotlib.pyplot as plt
import json

def read_history(history_file_path):
    try:
        with open(history_file_path, 'r', encoding='utf-8') as f:
            loaded_history = json.load(f)
        return loaded_history
    except FileNotFoundError:
        print(f"Error: The file {history_file_path} does not exist.")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from {history_file_path}: {e}")
        # 可选：返回空字典或其他默认值，以便程序可以继续执行
        return {}

def plot_history_Meta(history, epochs):
    plt.figure(figsize=(18, 12))  # 增大画布尺寸以容纳更多子图

    # # 1. 损失曲线
    # plt.subplot(3, 1, 1)
    # for idx, model_name in enumerate(["lenet", "alexnet", "vggnet"]):
    #     plt.plot(range(1, epochs + 1), history["train_loss_all"][idx], label=f"{model_name} Train Loss")
    #     plt.plot(range(1, epochs + 1), history["test_loss_all"][idx], label=f"{model_name} Test Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Loss Curve")
    # plt.legend()

    # # 2. 准确率曲线
    # plt.subplot(3, 1, 2)
    # for idx, model_name in enumerate(["lenet", "alexnet", "vggnet"]):
    #     plt.plot(range(1, epochs + 1), history["train_accur_all"][idx], label=f"{model_name} Train Acc")
    #     plt.plot(range(1, epochs + 1), history["test_accur_all"][idx], label=f"{model_name} Test Acc")
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.title("Accuracy Curve")
    # plt.legend()

    # # 6. 元分类器（XGBoost）性能
    # plt.subplot(3, 1, 3)
    # # 使用条形图展示元分类器的评估指标
    # metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    # meta_scores = [
    #     history.get("meta_accuracy_fin", 0),
    #     history.get("meta_precision_fin", 0),
    #     history.get("meta_recall_fin", 0),
    #     history.get("meta_f1_fin", 0)
    # ]
    # plt.bar(metrics, meta_scores, color=['blue', 'green', 'red', 'cyan'])
    # plt.ylim(0, 1)
    # for i, v in enumerate(meta_scores):
    #     plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    # plt.xlabel("Metrics")
    # plt.ylabel("Score")
    # plt.title("Meta-Classifier (XGBoost) Performance")

    plt.subplot(2, 2, 1)
    plt.plot(range(1, epochs + 1), history["meta_accuracy"], label=f"meta_accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("meta_accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    
    plt.subplot(2, 2, 2)

    plt.plot(range(1, epochs + 1), history["meta_precision"], label=f"meta_precision")
    plt.xlabel("Epoch")
    plt.ylabel("meta_precision")
    plt.title("Precision Curve")
    plt.legend()

    
    plt.subplot(2, 2, 3)

    plt.plot(range(1, epochs + 1), history["meta_recall"], label=f"meta_recall")
    plt.xlabel("Epoch")
    plt.ylabel("meta_recall")
    plt.title("Recall Curve")
    plt.legend()

    
    plt.subplot(2, 2, 4)

    plt.plot(range(1, epochs + 1), history["meta_f1"], label=f"meta_f1")
    plt.xlabel("Epoch")
    plt.ylabel("meta_f1")
    plt.title("F1 Score Curve")
    plt.legend()

    plt.tight_layout()
    plt.show()


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

    # Ensemble 召回率
    plt.subplot(3, 2, 5)
    plt.plot(range(1, epochs + 1), history["ensemble_train_recall"], "c^-", label="Train Recall")
    plt.plot(range(1, epochs + 1), history["ensemble_test_recall"], "kv-", label="Test Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Ensemble Recall Curve")
    plt.legend()

    # Ensemble F1 分数曲线

    # 隐藏第六个子图以保持布局整齐
    plt.subplot(3, 2, 6)
    plt.plot(range(1, epochs + 1), history["ensemble_train_f1"], "y*-", label="Train F1")
    plt.plot(range(1, epochs + 1), history["ensemble_test_f1"], "b+-", label="Test F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Ensemble F1 Score Curve")
    plt.legend()

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt

def plot_comparing(history1, history2, history3, epochs):
    # 设置图表尺寸，增大整体绘图区域
    plt.figure(figsize=(16, 12))

    # Ensemble 准确率曲线
    plt.subplot(3, 2, 1)
    plt.plot(range(1, epochs + 1), history1["ensemble_train_accuracy"], "r-o", label="Majority Voting Accuracy", color="red", linewidth=2)
    plt.plot(range(1, epochs + 1), history2["ensemble_train_accuracy"], "b-s", label="Weighted Voting Accuracy", color="blue", linewidth=2)
    plt.plot(range(1, epochs + 1), history3["meta_accuracy"], "g-^", label="Meta Accuracy", color="green", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Ensemble Accuracy Curve")
    plt.legend()

    # Ensemble 精确率曲线
    plt.subplot(3, 2, 2)
    plt.plot(range(1, epochs + 1), history1["ensemble_train_precision"], "c-o", label="Majority Voting Precision", color="cyan", linewidth=2)
    plt.plot(range(1, epochs + 1), history2["ensemble_train_precision"], "m-s", label="Weighted Voting Precision", color="magenta", linewidth=2)
    plt.plot(range(1, epochs + 1), history3["meta_precision"], "y-^", label="Meta Precision", color="yellow", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Ensemble Precision Curve")
    plt.legend()

    # Ensemble 召回率曲线
    plt.subplot(3, 2, 3)
    plt.plot(range(1, epochs + 1), history1["ensemble_train_recall"], "g-o", label="Majority Voting Recall", color="lime", linewidth=2)
    plt.plot(range(1, epochs + 1), history2["ensemble_train_recall"], "b-s", label="Weighted Voting Recall", color="blue", linewidth=2)
    plt.plot(range(1, epochs + 1), history3["meta_recall"], "r-^", label="Meta Recall", color="red", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Ensemble Recall Curve")
    plt.legend()

    # Ensemble F1 分数曲线
    plt.subplot(3, 2, 4)
    plt.plot(range(1, epochs + 1), history1["ensemble_train_f1"], "k-o", label="Majority Voting F1", color="black", linewidth=2)
    plt.plot(range(1, epochs + 1), history2["ensemble_train_f1"], "g-s", label="Weighted Voting F1", color="green", linewidth=2)
    plt.plot(range(1, epochs + 1), history3["meta_f1"], "b-^", label="Meta F1", color="blue", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Ensemble F1 Score Curve")
    plt.legend()

    # 调整布局
    plt.tight_layout()
    plt.show()

    
##%% 训练过程绘制
#%%  绘制 stacking的meta-Classifier的训练验证历史 
history_file_path = r'.\history\history_stacking3.json'
# 读取历史记录，绘制训练和测试结果
history1 = read_history(history_file_path)
# print(history1["train_loss_all"])
epochnum1  = len(history1["train_loss_all"][0])
plot_history_Meta(history1, epochnum1)

#%% 绘制 weighted voting的Classifier的训练验证历史
history_file_path = r'.\history\history_weighted.json'
# 读取历史记录，绘制训练和测试结果
history2 = read_history(history_file_path)
epochnum2  = len(history2["ensemble_train_recall"])
plot_history(history2, epochnum2)

#%%
history_file_path = r'.\history\history_voting2.json'
# 读取历史记录，绘制训练和测试结果
history3 = read_history(history_file_path)
epochnum3  = len(history3["ensemble_train_recall"])
plot_history(history3, epochnum3)

#%% Comparing the index changing on validation dataset
epochs = 100
# 读取历史记录，绘制训练和测试结果
history1 = read_history(r'.\history\history_voting2.json')
history2 = read_history(r'.\history\history_weighted.json')
history3 = read_history(r'.\history\history_stacking3.json')

plot_comparing(history1,history2,history3, epochs)
# %%
