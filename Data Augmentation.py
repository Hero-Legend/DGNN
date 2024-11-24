from transformers import BioGptForCausalLM, BioGptTokenizer
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import sacremoses

# 加载BioGPT模型和分词器
tokenizer = BioGptTokenizer.from_pretrained('/root/yuanzhu/BIO/Biomedical_DGNN/biogpt')
model = BioGptForCausalLM.from_pretrained('/root/yuanzhu/BIO/Biomedical_DGNN/biogpt')

# 加载数据集
train_data = pd.read_csv('./data/ddi2013ms/train.tsv', sep='\t')

# 检查类别分布，找出不平衡问题
category_counts = train_data['label'].value_counts()
print("Category Distribution:\n", category_counts)

# BioGPT生成函数
def generate_biogpt_text(prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,
        temperature=0.7,
        top_k=50,
        top_p=0.85,
        do_sample=True  # 启用采样，生成多样性文本
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 重点增强少数类样本
def augment_minority_class(data, target_class, num_samples=500):
    minority_samples = data[data['label'] == target_class]
    augmented_samples = []

    # 针对少数类样本生成增强数据
    for _, row in minority_samples.iterrows():
        prompt = row['sentence']
        generated_text = generate_biogpt_text(prompt)
        
        # 保持生成数据的格式与原始数据一致
        augmented_samples.append({
            'index': row['index'],  # 保持相同的index结构
            'sentence': generated_text,
            'label': target_class
        })

        # 如果达到目标样本数，停止生成
        if len(augmented_samples) >= num_samples:
            break

    return pd.DataFrame(augmented_samples)

# 根据类别的数量分布，决定要增强的类别（类别加权增强）
target_classes = category_counts[category_counts < category_counts.max()].index
augmented_data = []

# 针对每一个少数类类别进行数据增强
for target_class in target_classes:
    print(f"Augmenting data for class: {target_class}")
    # 增强到最大类别样本数
    augmented_class_data = augment_minority_class(train_data, target_class, num_samples=category_counts.max())
    augmented_data.append(augmented_class_data)

# 合并增强的数据
augmented_df = pd.concat(augmented_data)

### Step 3: 加入GAN部分 - 使用生成对抗网络生成数值特征
# 定义GAN的生成器和判别器
class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# 初始化生成器和判别器
noise_dim = 100
output_dim = train_data.shape[1] - 1  # 特征维度，排除标签
generator = Generator(noise_dim, output_dim)
discriminator = Discriminator(output_dim)

# 权重初始化函数
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# 应用权重初始化
generator.apply(weights_init)
discriminator.apply(weights_init)

# GAN训练
g_optimizer = optim.Adam(generator.parameters(), lr=0.001)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
loss_function = nn.BCELoss()

# GAN生成增强数据
num_epochs = 1000
batch_size = 64
for epoch in range(num_epochs):
    # 随机生成噪声
    noise = torch.randn((batch_size, noise_dim))

    # 生成新样本
    fake_data = generator(noise)

    # 判别器训练
    real_data = train_data.drop(columns=['label']).sample(batch_size).values
    real_data = pd.DataFrame(real_data).apply(pd.to_numeric, errors='coerce')  # 转换为数值型
    real_data.fillna(0, inplace=True)  # 用零填充NaN值
    real_data = torch.tensor(real_data.values, dtype=torch.float)
    real_labels = torch.ones((batch_size, 1))
    fake_labels = torch.zeros((batch_size, 1))

    # 判别器预测
    d_optimizer.zero_grad()
    real_predictions = discriminator(real_data)
    fake_predictions = discriminator(fake_data.detach())

    # 检查 real_predictions 的值范围
    if torch.isnan(real_predictions).any():
        print("Warning: real_predictions contains NaN values!")

    if not torch.all((real_predictions >= 0) & (real_predictions <= 1)):
        print("Error: real_predictions contains values out of range [0, 1]")

    # 判别器损失
    d_loss_real = loss_function(real_predictions, real_labels)
    d_loss_fake = loss_function(fake_predictions, fake_labels)
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    d_optimizer.step()

    # 生成器训练
    g_optimizer.zero_grad()
    fake_predictions_for_update = discriminator(fake_data)
    g_loss = loss_function(fake_predictions_for_update, real_labels)
    g_loss.backward()
    g_optimizer.step()

    # 打印训练进度
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

# 使用生成器生成增强特征数据
generated_features = generator(torch.randn((len(augmented_df), noise_dim))).detach().numpy()
augmented_df[train_data.columns.drop('label')] = generated_features

# 最终合并增强的数据
augmented_train_data = pd.concat([train_data, augmented_df], ignore_index=True)

# 保持与原数据集的列名和格式一致，保存增强后的数据
augmented_train_data.to_csv('augmented_train_data.tsv', sep='\t', index=False)
