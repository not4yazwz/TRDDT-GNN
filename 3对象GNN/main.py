import pandas as pd
from sklearn.model_selection import KFold
from util import *
from model import GNNModel
from torch import optim

"""
Dataloader
1、node_num.csv：
    drug、disease、target分别对应的序号
2、*_sim_matrix.txt：
    自相似矩阵
3、*_*_num.csv：
    单独的drug-target、drug-disease、disease-target关联
4、triple_net_num.csv：
    三维矩阵对应系数（三者同时相关）
"""

# 不同节点编号
node_name = pd.read_csv("dataset/node_num.csv", names=["name", "index"])
drug = node_name[:124]
disease = node_name[124:301]
target = node_name[301:404]

# 自相似矩阵
drug_similarity_matrix = np.loadtxt("dataset/drug_sim_matrix.txt")
disease_similarity_matrix = np.loadtxt("dataset/disease_sim_matrix.txt")
target_similarity_matrix = np.loadtxt("dataset/target_sim_matrix.txt")

# 三者同时相关，构建三维adj
triple_ass = pd.read_csv("dataset/triple_net_num.csv", names=["drug", "disease", "target"])
adj_matrix = np.zeros((124, 177, 104), dtype=int)
for index, row in triple_ass.iterrows():
    a_index = row['drug']
    b_index = row['disease'] - 124
    c_index = row['target'] - 301
    adj_matrix[a_index, b_index, c_index] = 1

"""
Method
1、输入数据：  
   三个【自相似矩阵】 和 【三维邻接矩阵】
2、处理方式：  
   ① 分别取对应【三维邻接矩阵】的截面为每个节点的【初始特征】
   ② 【自相似矩阵】通过K近邻得到各自的【自相似图】
   ③ 三个节点分别输入【自相似图】和【初始特征】进入<GNN>得到强化特征
   ④ 外积还原 原【三维邻接矩阵】
   ⑤ 计算损失并优化
"""
metric_list = []
metrics = np.zeros(7)
x_ROC_list = []
x_PR_list = []
y_ROC_list = []
y_PR_list = []

# 获取正负样本
samples = get_sample(adj_matrix)

# 五倍交叉验证开始
kf = KFold(n_splits=5, shuffle=True)
fold = 0
for train_index, val_index in kf.split(samples):
    fold += 1

    # 提出训练集和测试集的正负样本
    train_samples = samples[train_index, :]
    val_samples = samples[val_index, :]
    pos_train = train_samples[train_samples[:, 3] == 1][:, :-1]
    neg_train = train_samples[train_samples[:, 3] == 0][:, :-1]
    pos_val = val_samples[val_samples[:, 3] == 1][:, :-1]
    neg_val = val_samples[val_samples[:, 3] == 0][:, :-1]

    # 删除测试集中的已知关系
    new_adj_matrix = adj_matrix.copy()
    val_samples_positions = val_samples[:, :3].astype(int)
    for i in val_samples_positions:
        new_adj_matrix[i[0], i[1], i[2]] = 0

    # K近邻形成图
    drug_graph = k_matrix(drug_similarity_matrix, 30)
    disease_graph = k_matrix(disease_similarity_matrix, 30)
    target_graph = k_matrix(target_similarity_matrix, 30)

    # 初始化特征从三维邻接矩阵中获取
    X_drug = new_adj_matrix.copy()
    X_disease = new_adj_matrix.transpose(1, 0, 2)
    X_target = new_adj_matrix.transpose(2, 0, 1)

    # 将输入数据集转化为tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    drug_net_device = torch.tensor(np.array(np.where(drug_graph == 1)), dtype=torch.long, device=device)
    disease_net_device = torch.tensor(np.array(np.where(disease_graph == 1)), dtype=torch.long, device=device)
    target_net_device = torch.tensor(np.array(np.where(target_graph == 1)), dtype=torch.long, device=device)
    drug_x_device = torch.tensor(X_drug, dtype=torch.float32, device=device)
    disease_x_device = torch.tensor(X_disease, dtype=torch.float32, device=device)
    target_x_device = torch.tensor(X_target, dtype=torch.float32, device=device)

    """
    消融实验
    1、模型选择：GCN、GAT
    2、初始特征：随即特征、邻接矩阵切面、随机游走特征
    """
    conv_type = "GCN"
    initial_type = "random"
    if initial_type == "random_generate":
        drug_ini_x = torch.randn(124, 177, 104, dtype=torch.float32, device=device)
        disease_ini_x = torch.randn(177, 124, 104, dtype=torch.float32, device=device)
        target_ini_x = torch.randn(104, 124, 177, dtype=torch.float32, device=device)
    elif initial_type == "random_walk":
        drug_ini_x = drug_x_device
        disease_ini_x = disease_x_device
        target_ini_x = target_x_device
    else:
        drug_ini_x = drug_x_device
        disease_ini_x = disease_x_device
        target_ini_x = target_x_device

    # 模型定义
    model = GNNModel(124, 177, 104, 128, conv_type).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-3)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=5e-5, max_lr=1e-3, step_size_up=200,
                                            step_size_down=200, mode='exp_range', gamma=0.99, scale_fn=None,
                                            cycle_momentum=False, last_epoch=-1)

    model.train()
    for epoch in range(1000):
        model.zero_grad()
        z = model(drug_net_device, drug_ini_x, disease_net_device, disease_ini_x, target_net_device,
                  target_ini_x)
        z = z.cpu().reshape(124, 177, 104)
        loss = calculate_loss(z, pos_train.transpose(), neg_train.transpose())
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        z = model(drug_net_device, drug_x_device, disease_net_device, disease_x_device, target_net_device,
                  target_x_device)
        z = z.cpu().detach().reshape(124, 177, 104)
        metric, x_ROC, y_ROC, x_PR, y_PR = calculate_evaluation_metrics(z, pos_val.transpose(), neg_val.transpose())
        x_ROC_list.append(x_ROC)
        x_PR_list.append(x_PR)
        y_ROC_list.append(y_ROC)
        y_PR_list.append(y_PR)
        metric_list.append(metric)
        metrics = [m + n for m, n in zip(metric, metrics)]
        print(
            'fold_{}-auc:{:.4f},aupr:{:.4f},f1_score:{:.4f},accuracy:{:.4f},recall:{:.4f},specificity:{:.4f},precision:{:.4f}'.format(
                fold, metric[0],
                metric[1],
                metric[2],
                metric[3],
                metric[4],
                metric[5],
                metric[6]))

metrics = [value / 5 for value in metrics]
print(
    'auc:{:.4f},aupr:{:.4f},f1_score:{:.4f},accuracy:{:.4f},recall:{:.4f},specificity:{:.4f},precision:{:.4f}'.format(
        metrics[0],
        metrics[1],
        metrics[2],
        metrics[3],
        metrics[4],
        metrics[5],
        metrics[6]))
