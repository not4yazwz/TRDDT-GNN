import numpy as np
import torch


def k_matrix(matrix, k=20):
    """k-近邻图：每个节点与其 k 个最相似（或最近的）节点相连"""
    num = matrix.shape[0]
    knn_graph = np.zeros_like(matrix)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        top_k_indices = idx_sort[i, :k]
        knn_graph[i, top_k_indices] = matrix[i, top_k_indices]
        knn_graph[top_k_indices, i] = matrix[top_k_indices, i]
    np.fill_diagonal(knn_graph, 1)
    return knn_graph


def get_sample(adj):
    """获取和正样本等量而负样本"""
    positive_samples = np.argwhere(adj == 1)
    num_positive_samples = len(positive_samples)
    negative_samples = np.argwhere(adj == 0)

    np.random.seed(42)
    random_indices = np.random.choice(len(negative_samples), num_positive_samples, replace=False)
    negative_samples_selected = negative_samples[random_indices]

    positive_samples_with_label = np.hstack((positive_samples, np.ones((num_positive_samples, 1))))
    negative_samples_with_label = np.hstack((negative_samples_selected, np.zeros((num_positive_samples, 1))))

    samples = np.vstack((positive_samples_with_label, negative_samples_with_label))

    return samples


def calculate_loss(predict, pos_edge_idx, neg_edge_idx):
    """Loss计算"""
    pos_predict = predict[pos_edge_idx[0], pos_edge_idx[1], pos_edge_idx[2]]
    neg_predict = predict[neg_edge_idx[0], neg_edge_idx[1], neg_edge_idx[2]]
    predict_scores = torch.hstack((pos_predict, neg_predict))
    true_labels = torch.hstack((torch.ones(pos_predict.shape[0]), torch.zeros(neg_predict.shape[0])))
    loss_fun = torch.nn.BCEWithLogitsLoss(reduction='mean')
    return loss_fun(predict_scores, true_labels)


def calculate_evaluation_metrics(predict, pos_edges, neg_edges):
    """结果评估"""
    pos_edges = pos_edges.astype(int)
    neg_edges = neg_edges.astype(int)
    pos_predict = predict[pos_edges[0], pos_edges[1], pos_edges[2]]
    neg_predict = predict[neg_edges[0], neg_edges[1], neg_edges[2]]
    predict_labels = np.hstack((pos_predict, neg_predict))
    true_labels = np.hstack((np.ones(pos_predict.shape[0]), np.zeros(neg_predict.shape[0])))
    return get_metrics(true_labels, predict_labels)


def get_metrics(real_score, predict_score):
    """混淆矩阵"""
    real_score, predict_score = real_score.flatten(), predict_score.flatten()

    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    thresholds = sorted_predict_score[np.int32(len(sorted_predict_score) * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)

    predict_score_matrix = np.tile(predict_score, (thresholds.shape[1], 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1

    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]

    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T

    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]

    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    # plt.plot(x_ROC, y_ROC)
    # plt.plot(x_PR, y_PR)
    # plt.show()

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return [auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision], x_ROC, y_ROC, x_PR, y_PR
