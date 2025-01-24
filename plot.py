import matplotlib.pyplot as plt
import numpy as np
import re

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.interpolate import interp1d


def parse_complex_string_to_array(data_series):
    arrays = []
    for item in data_series:
        cleaned_item = item.strip("[]")
        split_items = cleaned_item.split("\n")
        array = [float(re.sub(r"[^\d.]", "", x)) for x in split_items]
        arrays.append(np.array(array))

    return arrays


def convert_string_to_array(data_str):
    # Removing the outermost brackets and splitting by newline
    data_str = re.sub(r'\[\[|\]\]', '', data_str)
    data_list = data_str.strip().split('\n')
    # Converting each element to float and removing brackets
    return np.array([float(re.sub(r'\[|\]', '', element.strip())) for element in data_list])


plt.rcParams['font.family'] = 'Times New Roman'
"""PR曲线绘制"""
# pr_data = pd.read_csv('pr_data.csv')
# pr_x_parsed = parse_complex_string_to_array(pr_data['PR_X'])
# pr_y_parsed = parse_complex_string_to_array(pr_data['PR_Y'])
#
# common_recall = np.linspace(0, 1, 500)
# interpolated_precisions = []
#
# for x, y in zip(pr_x_parsed, pr_y_parsed):
#     interp_func = interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
#     interpolated_precisions.append(interp_func(common_recall))
#
# average_precision = np.mean(interpolated_precisions, axis=0)
#
# # Create the main figure and axis
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(pr_x_parsed[0], pr_y_parsed[0], label=f'PR fold 1(AUPR=0.9378)', alpha=0.5, lw=1)
# ax.plot(pr_x_parsed[1], pr_y_parsed[1], label=f'PR fold 2(AUPR=0.9679)', alpha=0.5, lw=1)
# ax.plot(pr_x_parsed[2], pr_y_parsed[2], label=f'PR fold 3(AUPR=0.9476)', alpha=0.5, lw=1)
# ax.plot(pr_x_parsed[3], pr_y_parsed[3], label=f'PR fold 4(AUPR=0.9529)', alpha=0.5, lw=1)
# ax.plot(pr_x_parsed[4], pr_y_parsed[4], label=f'PR fold 5(AUPR=0.9595)', alpha=0.5, lw=1)
# ax.plot(common_recall, average_precision, label='Average PR(AUPR=0.9531)', color='black', linestyle='--', lw=1)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# ax.set_title('PR Curves', fontsize=14)
# ax.set_xlabel('Recall', fontsize=14)
# ax.set_ylabel('Precision', fontsize=14)
# ax.legend()
# ax_inset = inset_axes(ax, width="30%", height="30%", loc='center', borderpad=3)
# ax_inset.set_xlim(0.9, 1.0)
# ax_inset.set_ylim(0.8, 1.0)
# for i in range(len(pr_x_parsed)):
#     ax_inset.plot(pr_x_parsed[i], pr_y_parsed[i], alpha=0.5, lw=0.5)
# ax_inset.plot(common_recall, average_precision, color='black', linestyle='--', lw=0.5)
# mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")
# plt.show()

"""AUC曲线绘制"""
pr_data = pd.read_csv('roc_data.csv')
pr_x_parsed = pr_data['ROC_X'].apply(convert_string_to_array)
pr_y_parsed = pr_data['ROC_Y'].apply(convert_string_to_array)

common_recall = np.linspace(0, 1, 500)
interpolated_precisions = []

for x, y in zip(pr_x_parsed, pr_y_parsed):
    interp_func = interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
    interpolated_precisions.append(interp_func(common_recall))

average_precision = np.mean(interpolated_precisions, axis=0)

# Create the main figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(pr_x_parsed[0], pr_y_parsed[0], label=f'ROC fold 1(AUC=0.9156)', alpha=0.5, lw=1)
ax.plot(pr_x_parsed[1], pr_y_parsed[1], label=f'ROC fold 2(AUC=0.9545)', alpha=0.5, lw=1)
ax.plot(pr_x_parsed[2], pr_y_parsed[2], label=f'ROC fold 3(AUC=0.9228)', alpha=0.5, lw=1)
ax.plot(pr_x_parsed[3], pr_y_parsed[3], label=f'ROC fold 4(AUC=0.9374)', alpha=0.5, lw=1)
ax.plot(pr_x_parsed[4], pr_y_parsed[4], label=f'ROC fold 5(AUC=0.9422)', alpha=0.5, lw=1)
ax.plot(common_recall, average_precision, label='Average ROC(AUC=0.9345)', color='black', linestyle='--', lw=1)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.set_title('ROC Curves', fontsize=14)
ax.set_xlabel('FPR', fontsize=14)
ax.set_ylabel('TPR', fontsize=14)
ax.legend()
ax_inset = inset_axes(ax, width="30%", height="30%", loc='center', borderpad=3)

ax_inset.set_xlim(0.0, 0.1)
ax_inset.set_ylim(0.8, 1.0)
for i in range(len(pr_x_parsed)):
    ax_inset.plot(pr_x_parsed[i], pr_y_parsed[i], alpha=0.5, lw=0.5)
ax_inset.plot(common_recall, average_precision, color='black', linestyle='--', lw=0.5)
mark_inset(ax, ax_inset, loc1=1, loc2=2, fc="none", ec="0.5")
plt.show()

# import seaborn as sns
# def ablation_net():
#     N = 3
#     model = (0.9345, 0.9531, 0.9057)
#     model_rif = (0.5000, 0.7500, 0.6667)
#     model_ff = (0.8792, 0.9110, 0.8594)
#     model_gat = (0.9029, 0.9304, 0.8789)
#     ind = np.arange(N)
#     width = 0.2
#
#     # 创建图表并指定图表大小
#     fig, ax = plt.subplots(figsize=(16, 9))
#
#     palette = sns.color_palette("Set2", 4)  # "Set2" has distinct, soft colors
#     color_model = palette[0]  # e.g., 'Set2' first color
#     color_model_rif = palette[1]
#     color_model_ff = palette[2]
#     color_model_gat = palette[3]
#
#     # 绘制条形图
#     rects1 = ax.bar(ind - 1.5 * width, model, width, color=color_model, label='Model')
#     rects2 = ax.bar(ind - 0.5 * width, model_rif, width, color=color_model_rif, label='Model-RIF')
#     rects3 = ax.bar(ind + 0.5 * width, model_ff, width, color=color_model_ff, label='Model-FF')
#     rects4 = ax.bar(ind + 1.5 * width, model_gat, width, color=color_model_gat, label='Model-GAT')
#
#     # 设置y轴的显示范围
#     ax.set_ylim(0.45, 1.0)
#
#     # 设置x轴的刻度和标签
#     ax.set_xticks(ind)
#     ax.set_xticklabels(('AUROC', 'AUPR', 'F1-score'), fontsize=16)
#
#     # 设置图例
#     ax.legend(fontsize=16)
#
#     # 设置坐标轴字体大小
#     ax.tick_params(axis='both', labelsize=16)
#
#     # 显示图表
#     plt.show()
#     return
#
#
# ablation_net()