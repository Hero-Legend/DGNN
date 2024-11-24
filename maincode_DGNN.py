##############这个代码的主要思想是BioBERT+BiLSTM+REGCN+lossattention
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.nn.dense import Linear as DenseLinear
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AdamW
from torchvision.transforms import transforms
from tqdm import tqdm
import torch.cuda
import xml.etree.ElementTree as ET
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import dgl
import numpy as np
import torch as th
from dgl.nn import RelGraphConv
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN
from torch.nn import MultiheadAttention
from scipy.stats import uniform, randint
import torch.optim as optim
import matplotlib.pyplot as plt



#############################################定义 BERT 模型和 tokenizer##############################################

#导入Biobert
model_path = './model_path/biobert'                     #这个要用相对路径，不要用绝对路径
biobert_tokenizer = AutoTokenizer.from_pretrained(model_path)
biobert_model = AutoModel.from_pretrained(model_path)


####################################################################################################################

#############################################读取数据################################################################

df_train = pd.read_csv('./data/ddi2013ms/augmented_train_data.tsv', sep='\t')
df_dev = pd.read_csv('./data/ddi2013ms/dev.tsv', sep='\t')
df_test = pd.read_csv('./data/ddi2013ms/test.tsv', sep='\t')
print("read")

# print("训练集数据量：", df_train.shape)
# print("验证集数据量：", df_dev.shape)
# print("测试集数据量：", df_test.shape)

####################################################################################################################

#######################################################定义模型参数##################################################
#定义训练设备，默认为GPU，若没有GPU则在CPU上训练
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备

num_classes=5

# 定义模型参数
max_length = 300
batch_size = 32


# #############################################定义数据集和数据加载器###################################################
# # 定义数据集类
# 定义标签到整数的映射字典
label_map = {
    'DDI-false': 0,
    'DDI-effect': 1,
    'DDI-mechanism': 2,
    'DDI-advise': 3,
    'DDI-int': 4
    # 可以根据你的实际标签情况添加更多映射关系
}

# 定义数据集类
class DDIDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def construct_txt_intra_matrix(self, word_num):
        """构建文本模态内的矩阵"""
        mat = np.zeros((max_length, max_length), dtype=np.float32)
        mat[:word_num, :word_num] = 1.0
        return mat

    def __getitem__(self, idx):
        sentence = str(self.data['sentence'][idx])
        label_str = self.data['label'][idx]
        label = label_map[label_str]

        encoding = self.tokenizer(sentence, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
 
        # 使用 attention_mask 来确定有效的 token 数量
        word_num = encoding['attention_mask'].sum().item()
        txt_intra_matrix = self.construct_txt_intra_matrix(word_num)

        # # 输出检查语句
        # print(f"txt_intra_matrix shape: {txt_intra_matrix.shape}")

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'txt_intra_matrix': torch.tensor(txt_intra_matrix, dtype=torch.long)
        }

# 定义数据加载器
def create_data_loader(df, tokenizer, max_length, batch_size):
    dataset = DDIDataset(
        dataframe=df,
        tokenizer=tokenizer,

        max_length=max_length
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True  # 设置 drop_last=True 来丢弃最后一个不满足批次大小的批次,因为我们在LSTM和GCN维度转换时，出现了维度不匹配问题，找了很久原因，发现是在最后batch时，数据只有4条，导致维度出错
    )



# # 加载数据集和数据加载器
train_data_loader = create_data_loader(df_train, biobert_tokenizer, max_length, batch_size)
dev_data_loader = create_data_loader(df_dev, biobert_tokenizer, max_length, batch_size)
test_data_loader = create_data_loader(df_test, biobert_tokenizer, max_length, batch_size)

# for batch in test_data_loader:
#     print(batch)
#     break  # 这将打印第一批数据并中断循环。


# BioBERT + BiLSTM 特征提取器
class BioBERTBiLSTMFeatureExtractor(nn.Module):
    def __init__(self, freeze_bert=True, lstm_hidden_dim=256, lstm_layers=1, bidirectional=True):
        super(BioBERTBiLSTMFeatureExtractor, self).__init__()
        self.bert = biobert_model
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=768,  # BioBERT 输出的特征维度
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.8)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # 获取 BioBERT 输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        # 通过 BiLSTM
        lstm_output, _ = self.lstm(sequence_output)

        # 使用 dropout 层
        output_features = self.dropout(lstm_output)
        return output_features



# 定义双图神经网络模型
class DualGraphNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DualGraphNN, self).__init__()
        self.interaction_conv1 = GCNConv(input_dim, hidden_dim)
        self.interaction_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.similarity_conv1 = GCNConv(input_dim, hidden_dim)
        self.similarity_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, interaction_data, similarity_data):
        x_inter = F.relu(self.interaction_conv1(interaction_data.x, interaction_data.edge_index))
        x_inter = F.relu(self.interaction_conv2(x_inter, interaction_data.edge_index))
        x_sim = F.relu(self.similarity_conv1(similarity_data.x, similarity_data.edge_index))
        x_sim = F.relu(self.similarity_conv2(x_sim, similarity_data.edge_index))
        x = torch.cat([x_inter, x_sim], dim=1)
        out = self.fc(x)
        return out

# 创建双图数据
interaction_edge_index = torch.tensor([
    [0, 1, 2],
    [1, 2, 0]
], dtype=torch.long)
num_nodes = 5  # 假设有5个节点
interaction_graph = Data(edge_index=interaction_edge_index, num_nodes=num_nodes).to(device)

similarity_edge_index = torch.tensor([
    [0, 1],
    [1, 2]
], dtype=torch.long)
similarity_graph = Data(edge_index=similarity_edge_index, num_nodes=num_nodes).to(device)


# 双图关系提取模型
class Classifier(nn.Module):
    def __init__(self, hidden_dim=768, num_classes=5):  # 修改 hidden_dim 为 768
        super(Classifier, self).__init__()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, features):
        logits = self.fc(features)  
        return logits

# 双图关系提取模型
class DualGraphRelationModel(nn.Module):
    def __init__(self, d_model=512, d_hidden=256, dropout=0.8):
        super().__init__()
        self.dp = dropout
        self.d_model = d_model
        self.hid = d_hidden
        self.BioBERTBiLSTMFeatureExtractor = BioBERTBiLSTMFeatureExtractor(freeze_bert=True)
        self.dual_graph_nn = DualGraphNN(input_dim=512, hidden_dim=128, output_dim=256)  # 输入为 BiLSTM 输出维度 512
        self.gcn_projection = nn.Linear(256, 512)  # 将 GCN 输出从 256 转换为 512
        self.dropout = nn.Dropout(dropout)
        self.attention = MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.8)
        self.interactive_attention = MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.8)
        self.MLP = nn.Linear(512 * 300, 256)

    def forward(self, input_ids, attention_mask, labels, mat, interaction_graph, similarity_graph):
        device = input_ids.device

        # 获取 BERT + BiLSTM 的输出
        text_features = self.BioBERTBiLSTMFeatureExtractor(input_ids, attention_mask)

        # GCN部分
        gcn_features_list = []
        for i in range(text_features.shape[0]):
            single_fea = text_features[i]
            interaction_graph.x = single_fea
            similarity_graph.x = single_fea
            gcn_output = self.dual_graph_nn(interaction_graph, similarity_graph)
            gcn_features_list.append(gcn_output.unsqueeze(0))

        gcn_features = torch.cat(gcn_features_list, dim=0)

        # 重新调整 gcn_features 的形状
        batch_size, sequence_length, hidden_dim = gcn_features.shape
        gcn_features = gcn_features.view(batch_size * sequence_length, hidden_dim)

        # 使用线性层将 gcn_features 调整到与 text_features 一致的维度
        gcn_features = self.gcn_projection(gcn_features)


        # 恢复 gcn_features 的形状
        gcn_features = gcn_features.view(batch_size, sequence_length, -1)


        # 应用交互注意力
        interactive_output, _ = self.interactive_attention(gcn_features, text_features, text_features)


        combined_features = gcn_features + interactive_output


        # 将 combined_features 的维度变换为适合全连接层输入
        batch_size, seq_len, hidden_size = combined_features.shape
        combined_features = combined_features.view(batch_size, seq_len * hidden_size)


        combined_features = self.MLP(combined_features)


        return combined_features, None





#FocalLoss定义
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Convert targets to one-hot encoding
        targets = F.one_hot(targets, num_classes=inputs.size(1)).float()
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt) * targets
        focal_loss = -self.alpha * ((1 - pt) ** self.gamma) * logpt * targets
        focal_loss = focal_loss.sum(dim=1)  # Sum over classes

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss




#分类器
class Classifier(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=5):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, features):
        logits = self.fc(features)  
        return logits


# 整体模型
class BioMedRelationExtractor(nn.Module):
    def __init__(self):
        super(BioMedRelationExtractor, self).__init__()
        self.DualGraphRelationModel = DualGraphRelationModel()
        self.classifier = Classifier()

    def forward(self, input_ids, attention_mask, labels, mat, interaction_graph, similarity_graph):
        gcn_features, _ = self.DualGraphRelationModel(input_ids, attention_mask, labels, mat, interaction_graph, similarity_graph)
        logits = self.classifier(gcn_features)
        return logits

# 在训练和测试之前定义 true_labels 和 predicted_probs
true_labels = []
predicted_probs = []


# 训练代码
def train_model(model, train_data_loader, optimizer, criterion, device):

    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    epoch_true_labels = []
    epoch_pred_labels = []

    for batch in train_data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        mat = batch['txt_intra_matrix'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, labels, mat, interaction_graph, similarity_graph)   # 从模型中获取 outputs 和 attn_scores

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total_preds += labels.size(0)
        correct_preds += (predicted == labels).sum().item()
        
        epoch_true_labels.extend(labels.cpu().numpy())
        epoch_pred_labels.extend(predicted.cpu().numpy())

        # 记录每个 batch 的真实标签和预测概率
        true_labels.extend(labels.cpu().numpy())
        predicted_probs.extend(F.softmax(logits, dim=1).detach().cpu().numpy())  # Use detach() here

    train_loss = running_loss / len(train_data_loader)
    train_acc = correct_preds / total_preds
    
    # 计算混淆矩阵和 F1 值
    conf_matrix = confusion_matrix(epoch_true_labels, epoch_pred_labels)  # Use epoch_pred_labels here
    accuracy = accuracy_score(epoch_true_labels, epoch_pred_labels)
    precision = precision_score(epoch_true_labels, epoch_pred_labels, average='weighted', zero_division=1)
    recall = recall_score(epoch_true_labels, epoch_pred_labels, average='weighted', zero_division=1)
    f1 = 2*precision*recall/(precision+recall)
    
    return train_loss, train_acc, conf_matrix, f1


# 测试代码
def test_model(model, test_data_loader, criterion, device):

    model.eval()
    epoch_true_labels = []
    epoch_pred_labels = []

    with torch.no_grad():
        for batch in test_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            mat = batch['txt_intra_matrix'].to(device)

            logits = model(input_ids, attention_mask, labels, mat, interaction_graph, similarity_graph)
            _, predicted = torch.max(logits, 1)

            epoch_true_labels.extend(labels.cpu().numpy())
            epoch_pred_labels.extend(predicted.cpu().numpy())

    # 计算混淆矩阵、准确率、精确率、召回率和 F1 值
    conf_matrix = confusion_matrix(epoch_true_labels, epoch_pred_labels)
    accuracy = accuracy_score(epoch_true_labels, epoch_pred_labels)
    precision = precision_score(epoch_true_labels, epoch_pred_labels, average='weighted', zero_division=1)
    recall = recall_score(epoch_true_labels, epoch_pred_labels, average='weighted', zero_division=1)
    f1 = 2*precision*recall/(precision+recall)

    # 计算每个类别的F1值
    class_report = classification_report(epoch_true_labels, epoch_pred_labels, output_dict=True, zero_division=1)
    f1_per_class = {label: metrics['f1-score'] for label, metrics in class_report.items() if label.isdigit()}
    
    return conf_matrix, accuracy, precision, recall, f1, f1_per_class


#模型实例化
model = BioMedRelationExtractor().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
criterion = FocalLoss(alpha=1, gamma=2)


# 训练模型
num_epochs = 20

# 存储训练过程中每个 epoch 的结果
epoch_train_losses = []
epoch_train_accuracies = []
epoch_train_f1_scores = []
epoch_train_conf_matrices = []

# 打开文件，以追加模式（'a'）写入
with open('./figure/training_results.txt', 'a') as f:
    for epoch in tqdm(range(num_epochs), desc="Training Progress"):

        train_loss, train_acc, conf_matrix, f1 = train_model(model, train_data_loader, optimizer, criterion, device)

        #保存结果文件
        f.write(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, F1 Score: {f1:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(conf_matrix) + '\n')

        #打印输出
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, F1 Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        
        # 保存每个 epoch 的结果用于后续可视化
        epoch_train_losses.append(train_loss)
        epoch_train_accuracies.append(train_acc)
        epoch_train_f1_scores.append(f1)
        epoch_train_conf_matrices.append(conf_matrix)



with open('./figure/test_results.txt', 'w') as f:
    test_conf_matrix, test_accuracy, test_precision, test_recall, test_f1, test_f1_per_class = test_model(model, test_data_loader, criterion, device)
    f.write("Test Results:\n")
    f.write("Confusion Matrix:\n")
    f.write(str(test_conf_matrix) + '\n')
    f.write("Accuracy: " + str(test_accuracy) + '\n')
    f.write("Precision: " + str(test_precision) + '\n')
    f.write("Recall: " + str(test_recall) + '\n')
    f.write("F1 Score: " + str(test_f1) + '\n')
    f.write("F1 Score per Class:\n")
    for label, f1 in test_f1_per_class.items():
        f.write(f"Class {label}: {f1:.4f}\n")
    print("Test Results:")
    print("Confusion Matrix:")
    print(test_conf_matrix)
    print("Accuracy:", test_accuracy)
    print("Precision:", test_precision)
    print("Recall:", test_recall)
    print("F1 Score:", test_f1)
    print("F1 Score per Class:")
    for label, f1 in test_f1_per_class.items():
        print(f"Class {label}: {f1:.4f}")



##############################################画图####################################################
# 计算每个类别的 AUC
# 假设你有 `true_labels` 和 `predicted_probs` 以及 `label_map`
num_classes = 5  # 根据你的情况调整
fpr = dict()
tpr = dict()
roc_auc = dict()

# 对于每个类别，计算fpr, tpr和AUC
for i in range(num_classes):
    # 获取每个类的真值和预测概率
    class_true_labels = [1 if true_label == i else 0 for true_label in true_labels]
    class_predicted_probs = [probs[i] for probs in predicted_probs]

    # 计算 fpr 和 tpr
    fpr[i], tpr[i], _ = roc_curve(class_true_labels, class_predicted_probs)
    roc_auc[i] = auc(fpr[i], tpr[i])


##################################训练集结果画图####################################################
# 画训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_train_losses, marker='o', label='Train Loss')
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.title('Training Loss Over Epochs', fontsize=20)
plt.legend(fontsize=18)
plt.grid(True, linestyle="--", alpha=0.8)
plt.tight_layout()
plt.savefig('./figure/training_loss_over_epochs.png', dpi=300)
plt.show()

# 画训练准确率曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_train_accuracies, marker='o', label='Train Accuracy')
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.title('Training Accuracy Over Epochs', fontsize=20)
plt.legend(fontsize=18)
plt.grid(True, linestyle="--", alpha=0.8)
plt.tight_layout()
plt.savefig('./figure/training_accuracy_over_epochs.png', dpi=300)
plt.show()

# 画训练 F1 分数曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_train_f1_scores, marker='o', label='Train F1 Score')
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('F1 Score', fontsize=20)
plt.title('Training F1 Score Over Epochs', fontsize=20)
plt.legend(fontsize=18)
plt.grid(True, linestyle="--", alpha=0.8)
plt.tight_layout()
plt.savefig('./figure/training_f1_score_over_epochs.png', dpi=300)
plt.show()




################################################测试集结果画图##############################
# 画混淆矩阵热力图
plt.figure(figsize=(10, 8))
sns.heatmap(test_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_map.keys(), yticklabels=label_map.keys(), annot_kws={"size": 20})
plt.xlabel('Predicted Label', fontsize=20)
plt.ylabel('True Label', fontsize=20)
plt.title('Confusion Matrix', fontsize=20)
plt.xticks(rotation=45, ha='right')   # 设置x轴标签，旋转一定角度以避免重叠（如果需要）
plt.yticks(rotation=0)           # 设置y轴标签水平显示
plt.tight_layout()  # 调整子图布局以适应标签
plt.legend(fontsize=20)
plt.savefig('./figure/confusion_matrix_heatmap.png', dpi=300)  # 保存混淆矩阵热力图
plt.show()

# 画准确率
plt.figure(figsize=(10, 8))
bar_width = 0.3  # # 设置柱子的宽度,可以根据需要调整这个值
plt.bar(range(len(label_map)), [test_accuracy]*len(label_map), color='skyblue', width=bar_width, align='center')
plt.xlabel('Class', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('', fontsize=16)
plt.xticks(range(len(label_map)), label_map.keys(), rotation=45, ha='right')
plt.tight_layout()  # 调整子图布局以适应标签
plt.savefig('./figure/accuracy_by_class.png', dpi=300)  # 保存准确率图
plt.show()

#画AUC曲线图
plt.figure(figsize=(10, 6))

for i, label in enumerate(label_map):
    plt.plot(fpr[i], tpr[i], label=f'{label} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)  # 随机分类器的线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC)', fontsize=16)
plt.legend(loc='lower right')
plt.tight_layout()  # 调整子图布局以适应标签
plt.savefig('./figure/auc_by_class.png', dpi=300)
plt.show()
