import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score, average_precision_score, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

def compute_accuracy(output, target):
    """
     Calculates the classification accuracy.
    :param target: Tensor of correct labels of size [batch_size]
    :param output: Tensor of model predictions of size [batch_size, num_classes]
    :return: prediction accuracy
    """
    num_samples = target.size(0)
    num_correct = torch.sum(target == torch.argmax(output, dim=1))
    accuracy = num_correct.float() / num_samples
    return accuracy.detach().cpu().numpy()

def binary_accuracy(output, target):
    """
     Calculates the classification accuracy.
    :param target: Tensor of correct labels of size [batch_size]
    :param output: Tensor of model predictions of size [batch_size, num_classes]
    :return: prediction accuracy
    """
    num_samples = target.size(0)
    score = torch.sigmoid(output)
    preds = score > 0.5
    num_correct = torch.sum(target == preds)
    accuracy = num_correct.float() / num_samples
    return accuracy.detach().cpu().numpy()


def binary_cls_compute_metrics(output, target):
    fpr_, tpr_, _ = roc_curve(target.detach().cpu().numpy(), output[:, -1].detach().cpu().numpy())
    auc_ = auc(fpr_, tpr_)
    #precision_, recall_, _ = precision_recall_curve(target.detach().cpu().numpy(), output[:, -1].detach().cpu().numpy())
    score = torch.sigmoid(output)
    preds = score > 0.5
    precision_ = precision_score(target.detach().cpu().numpy(), preds.detach().cpu().numpy())
    recall_ = recall_score(target.detach().cpu().numpy(), preds.detach().cpu().numpy())
    f1_ = f1_score(target.detach().cpu().numpy(), preds.detach().cpu().numpy())
    metrics = {'f1': f1_,
               'auc': auc_,
               'precision': precision_,
               'recall': recall_,
               'acc': binary_accuracy(output, target)}

    print(confusion_matrix(target.detach().cpu().numpy(), preds.detach().cpu().numpy()))

    return metrics

def plot_dist_age(output, target, conf):
    score = torch.sigmoid(output)
    score = score.squeeze().detach().cpu().numpy()
    target = target.squeeze().detach().cpu().numpy()
    all_data = pd.DataFrame(data=zip(score, target), columns=["score", "target"])
    all_data = all_data.rename(index=lambda x: "all")
    #sns.violinplot(x=all_data.index, y="score", data=all_data, hue="target", palette="muted", split=True)
    #plt.show()


    score_young, target_young = score[conf <= 40], target[conf <= 40]
    young_data = pd.DataFrame(data=zip(score_young, target_young), columns=["score", "target"])
    young_data = young_data.rename(index=lambda x: "young")
    #sns.violinplot(x=young_data.index, y="score", data=young_data, hue="target", palette="muted", split=True)
    #plt.show()

    score_old, target_old = score[conf > 40], target[conf > 40]
    old_data = pd.DataFrame(data=zip(score_old, target_old), columns=["score", "target"])
    old_data = old_data.rename(index=lambda x: "old")
    #sns.violinplot(x=old_data.index, y="score", data=old_data, hue="target", palette="muted", split=True)
    #plt.show()

    df = pd.concat([all_data, young_data, old_data])

    sns.violinplot(x=df.index, y="score", data=df, hue="target", palette="pastel", split=True)
    plt.savefig("cIRM_age_scores.pdf", dpi=300)
    print("Done!")

def plot_dist_sex(output, target, conf):
    score = torch.sigmoid(output)
    score = score.squeeze().detach().cpu().numpy()
    target = target.squeeze().detach().cpu().numpy()
    all_data = pd.DataFrame(data=zip(score, target), columns=["score", "target"])
    all_data = all_data.rename(index=lambda x: "all")
    #sns.violinplot(x=all_data.index, y="score", data=all_data, hue="target", palette="muted", split=True)
    #plt.show()


    score_young, target_young = score[conf == "F"], target[conf == "F"]
    young_data = pd.DataFrame(data=zip(score_young, target_young), columns=["score", "target"])
    young_data = young_data.rename(index=lambda x: "Female")
    #sns.violinplot(x=young_data.index, y="score", data=young_data, hue="target", palette="muted", split=True)
    #plt.show()

    score_old, target_old = score[conf == "M"], target[conf == "M"]
    old_data = pd.DataFrame(data=zip(score_old, target_old), columns=["score", "target"])
    old_data = old_data.rename(index=lambda x: "Male")
    #sns.violinplot(x=old_data.index, y="score", data=old_data, hue="target", palette="muted", split=True)
    #plt.show()

    df = pd.concat([all_data, young_data, old_data])

    sns.violinplot(x=df.index, y="score", data=df, hue="target", palette="pastel", split=True)
    plt.savefig("cIRM_scores_sex.pdf", dpi=300)
    print("Done!")

def compute_metrics(outputs, targets):
    metrics = roc_auc_score(targets.detach().cpu().numpy(), outputs.detach().cpu().numpy())
    return metrics
