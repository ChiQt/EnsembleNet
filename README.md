# Ensemble NNetworks
This project aims to solve a binary or multi-class image recognition problem, for example, by analyzing slice images of pulmonary nodules to determine whether they are benign or malignant nodules. Multiple neural network models were used as base classifiers in this task. In addition, ensemble learning is introduced to reduce the risk of possible overfitting of a single model and improve the accuracy and robustness of ensemble models by combining multiple base classifiers.

This project is created based on Pytorch by ChiQt(chiXJ).

Notes:

1. The code comments and the slide in \readhelper\ are currently mostly in Chinese.
2. With limited time and computing resourses, this project uses only three network models as base classifiers, and only three methods are used to generate the ensemble model: majority voting method, weighted voting method, and stacking method. Whisper: The creator personally believes that the last type belongs to the true ensemble learning, which truly combines the advantages of all base classifiers.
3. Methods that use the network as a base classifier for stepwise iterative methods, such as AdaBoost, have yet to be implemented...

## base classifiers
![three classic network models](".\readhelper\base classifiers.png")

**LeNet**: Classical convolutional neural network, suitable for small scale data sets, fewer parameters.

**AlexNet**: Deep network structure with strong learning ability in image classification, capable of handling complex images.

**VGGNet**: With a deeper structure and a concise convolutional layer design, it is suitable for complex image data sets.

## voting method and stacking method
![voting method](".\readhelper\voting method.png")
![stacking method step1](".\readhelper\stacking step1.png")
![stacking method step2](".\readhelper\stacking step2.png")
step2 refer to: Yao, Xiaotong & Fu, Xiaoli & Zong, Chaofei. (2022). Short-Term Load Forecasting Method Based on Feature Preference Strategy and LightGBM-XGboost. IEEE Access.

## Introduction
running these scripts is very sample, you can bulid or revise your moodels based on them.

**drawplot.py**: You can run this file directly. It is revealing the history of training and validating process of base calssifiers and ensemble classifiers.

**ensembleModel_(es)_voting.py**: Majority voting method.

**ensembleModel_(es)_Weighted_voting.py**: weighted voting method.

**ensembleModel_meta**: stacking method.

**predict_file.py**: Predict images with three methods. If you want to caculate the
results, please note that image categories are determined by the folder structure.

**predict_image.py**: Predict images with three methods. You can directly predict that single or multiple images do not necessarily need to provide true labels.If you want to caculate the results, please note that labels are need to be determined by you.


## Bib

If you want to refer to this project, you can use the following format:

```bibtex
@misc{EnsembleNet,
  author = {ChiQt},
  title = {EnsembleNet},
  year = {2024},
  url = {https://github.com/ChiQt/EnsembleNet.git},
  note = {Accessed: 2024-12-20}
}
