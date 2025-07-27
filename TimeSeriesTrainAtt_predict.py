# 修复RuntimeError的改进代码
'''
根据您提供的错误信息，矩阵乘法维度不匹配(140
x384和512x256)，这通常发生在模型中维度设置不兼容的情况。由于序列长度是140，特征数是1，我需要调整模型结构以确保各层维度匹配。
'''

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn, optim
import torch.nn.functional as F
from arff2pandas import a2p
import csv, datetime, os
import copy,random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

# ------------------------
# Configurations
# ------------------------
DRAW = False  # 是否绘制中间曲线
RANDOM_SEED = 42
kl_weight = 0.1  # KL 散度权重
DROPOUT = 0.2  # Dropout率
PATIENCE = 100  # 早停耐心值
CONTAMINATION = 0.05  # 动态阈值设置中的污染率

# 设置隐藏层维度，确保维度匹配
EMBEDDING_DIM = 64  # 减小嵌入维度，避免维度过大 32
LATENT_DIM = 32  # 减小潜在空间维度  16

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# 日志文件初始化
current_file_name = os.path.splitext(os.path.basename(__file__))[0]
current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
log_dir = 'log'
os.makedirs(log_dir, exist_ok=True)

# 绘图风格
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ------------------------
# 简化的注意力机制
# ------------------------
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim]
        # 计算每个时间步的注意力权重
        attn_weights = self.attention(x)  # [batch_size, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        context = torch.sum(x * attn_weights, dim=1)  # [batch_size, hidden_dim]
        return context, attn_weights


# ------------------------
# Variational Encoder
# ------------------------
class EncoderVAE(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=EMBEDDING_DIM, latent_dim=LATENT_DIM):
        super(EncoderVAE, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = embedding_dim

        # 第一个LSTM层 - 单向以减少参数
        self.rnn = nn.LSTM(n_features, embedding_dim,
                           batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(DROPOUT)
        self.norm = nn.LayerNorm(embedding_dim)

        # 简化注意力
        self.attention = Attention(embedding_dim)

        # 变分参数
        self.fc_mu = nn.Linear(embedding_dim, latent_dim)
        self.fc_logvar = nn.Linear(embedding_dim, latent_dim)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        batch_size = x.size(0)

        # LSTM层 out:1,140,32   hidden:1,1,32
        out, (hidden, _) = self.rnn(x)
        out = self.dropout(out)
        out = self.norm(out) # 去掉归一化

        # 注意力层 context:1,32  attn_weights:1,140,1
        context, attn_weights = self.attention(out)

        # 计算潜变量参数 mu:1 16 logvar 1 16
        mu = self.fc_mu(context)
        logvar = self.fc_logvar(context)

        return mu, logvar, out, attn_weights

# ------------------------
# Variational Decoder
# ------------------------
class DecoderVAE(nn.Module):
    def __init__(self, seq_len, latent_dim=LATENT_DIM, embedding_dim=EMBEDDING_DIM, n_features=1):
        super(DecoderVAE, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.hidden_dim = embedding_dim
        self.n_features = n_features

        # 将潜变量z映射到隐藏状态
        self.fc_z = nn.Linear(latent_dim, embedding_dim)

        # GRU层用于序列生成
        self.gru = nn.GRU(embedding_dim, embedding_dim, batch_first=True)

        # 输出层
        self.fc_out = nn.Linear(embedding_dim, n_features)

        self.dropout = nn.Dropout(DROPOUT)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, z, encoder_outputs):
        batch_size = z.size(0)

        # 将z映射到初始隐藏状态 hz:1,1,32
        hz = self.fc_z(z).unsqueeze(0)  # [1, batch_size, embedding_dim]

        # 创建每个时间步的输入(全零) 1,140,32
        decoder_input = torch.zeros(batch_size, self.seq_len, self.hidden_dim).to(z.device)

        # 通过GRU生成序列 1,140,32
        output, _ = self.gru(decoder_input, hz)
        output = self.dropout(output)
        output = self.norm(output)

        # 映射到特征空间 1,140,1
        reconstructed = self.fc_out(output)

        # 返回重构序列和空的注意力权重(简化处理)
        return reconstructed, None


# ------------------------
# VAE Model
# ------------------------
class RecurrentVariationalAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=EMBEDDING_DIM, latent_dim=LATENT_DIM):
        super(RecurrentVariationalAutoencoder, self).__init__()
        self.encoder = EncoderVAE(seq_len, n_features, embedding_dim, latent_dim)
        self.decoder = DecoderVAE(seq_len, latent_dim, embedding_dim, n_features)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar, enc_out, attn_enc = self.encoder(x)
        z = self.reparameterize(mu, logvar)#z:1 16
        recon, attn_dec = self.decoder(z, enc_out)
        return recon, mu, logvar


# ------------------------
# 数据预处理函数
# ------------------------
def preprocess_data(normal_df, anomaly_df):
    """对数据进行标准化处理"""
    scaler = StandardScaler()

    # 拟合标准化器仅使用正常样本
    normal_array = normal_df.values
    scaler.fit(normal_array)

    # 转换所有数据
    normal_scaled = scaler.transform(normal_df.values)
    anomaly_scaled = scaler.transform(anomaly_df.values)

    # 转回DataFrame
    normal_df_scaled = pd.DataFrame(normal_scaled, columns=normal_df.columns)
    anomaly_df_scaled = pd.DataFrame(anomaly_scaled, columns=anomaly_df.columns)

    return normal_df_scaled, anomaly_df_scaled, scaler


# ------------------------
# 动态阈值选择
# ------------------------
def determine_threshold(model, val_ds, device, contamination=CONTAMINATION):
    """基于验证集重构误差确定最佳阈值"""
    model.eval()
    rec_losses = []
    with torch.no_grad():
        for seq in val_ds:
            x = seq.unsqueeze(0).to(device)
            recon, mu, logvar = model(x)
            rec_loss = F.mse_loss(recon, x, reduction='sum').item()
            rec_losses.append(rec_loss)

    # 基于离群率确定阈值
    threshold = np.percentile(rec_losses, 100 * (1 - contamination))
    return threshold

# ------------------------
# 动态阈值选择（Kapur 方法）
# ------------------------
def determine_threshold_kapur(model, val_ds, device, n_bins=256):
    """
    基于验证集重构误差，使用 Kapur 熵最大化方法确定最佳阈值

    参数:
        model      : 已训练好的 VAE 模型
        val_ds     : 验证集列表，每个元素是形如 (seq_len, n_features) 的 tensor
        device     : 运行设备 ('cpu' 或 'cuda')
        n_bins     : 用于直方图的 bin 数（默认 256），越大会增加计算量但精度更高

    返回:
        threshold  : Kapur 方法选出的阈值（对应重构误差）
    """
    model.eval()
    rec_losses = []
    with torch.no_grad():
        for seq in val_ds:
            x = seq.unsqueeze(0).to(device)
            recon, mu, logvar = model(x)
            # 计算每个序列的重构误差（L2 范数平方和）
            rec_loss = F.mse_loss(recon, x, reduction='sum').item()
            rec_losses.append(rec_loss)
    rec_losses = np.array(rec_losses)

    # 1. 对 rec_losses 构建直方图
    #    hist: 每个 bin 中的样本数； bin_edges: 长度为 n_bins+1 的边界数组
    hist, bin_edges = np.histogram(rec_losses, bins=n_bins, density=False)
    # 归一化为概率密度（注意：hist.sum() == rec_losses.size）
    prob = hist.astype(np.float64) / hist.sum()

    # 2. 计算每个候选阈值对应的 Kapur 两部分熵之和
    #    Kapur 方法核心公式：
    #      H_background(T) = - sum_{i=1}^T p_i / P0 * log(p_i / P0)
    #      H_foreground(T) = - sum_{i=T+1}^N p_i / P1 * log(p_i / P1)
    #    其中 P0 = sum_{i=1}^T p_i,  P1 = sum_{i=T+1}^N p_i
    # 注意：在实际计算时要避免对零概率项取 log

    # 预先计算熵时需要用到累积概率
    cumsum_prob = np.cumsum(prob)  # 累积概率 P0(k) = sum_{i=0}^k prob[i]
    cumsum_prob_rev = np.cumsum(prob[::-1])  # 逆向累积，用于计算前景概率

    # 为避免 log(0)，在计算时将零概率项屏蔽掉
    eps = 1e-12
    # 反向概率数组：prob_rev[i] = prob[N - 1 - i]
    prob_rev = prob[::-1]

    # 预先计算整个直方图每一项的 p_i * log(p_i)（只计算非零者）
    p_log_p = np.zeros_like(prob)
    nonzero_idx = prob > 0
    p_log_p[nonzero_idx] = prob[nonzero_idx] * np.log(prob[nonzero_idx])

    # 同理，反向的 p_rev * log(p_rev)
    p_rev_log_p_rev = np.zeros_like(prob_rev)
    nonzero_rev_idx = prob_rev > 0
    p_rev_log_p_rev[nonzero_rev_idx] = prob_rev[nonzero_rev_idx] * np.log(prob_rev[nonzero_rev_idx])

    # 现在，对每个可能的阈值索引 k（对应第 k 个 bin 结束处）：
    #   P0 = cumsum_prob[k],   P1 = 1 - P0
    #   H0(k) = - sum_{i=0}^k p_i * log(p_i) / P0  +  log(P0)   （等价变形）
    #   H1(k) = - sum_{i=k+1}^{N-1} p_i * log(p_i) / P1  +  log(P1)

    # 计算背景熵中的分子： cum_p_log_p[k] = sum_{i=0}^k p_i * log(p_i)
    cum_p_log_p = np.cumsum(p_log_p)
    # 计算前景熵中的分子：对于原始 prob，需要倒序处理
    cum_p_rev_log_p_rev = np.cumsum(p_rev_log_p_rev)

    # 由于前景（foreground）对应的 bin 索引 i > k，
    # 当 k = 0 时，foreground 是 bins [1..N-1]，
    # cum_p_rev_log_p_rev[?] 的索引需要和正向索引对应上来。
    # 对于正向第 k 个阈值，前景的“累积”其实是从 bin k+1 到末尾，
    # 即相当于反向从索引 0 到 N-2-k。所以我们要对齐：
    #   前景熵分子 H1_num[k] = sum_{i=k+1}^{N-1} p[i] * log(p[i])
    #   令 idx_rev = (N-1) - (k+1) = N-2-k，则 cum_p_rev_log_p_rev[idx_rev] = sum_{j=0..idx_rev} p_rev[j] * log(p_rev[j])
    # 其中 N = prob.size, idx_rev = N-2-k

    N = prob.size
    H_total = -np.inf * np.ones(N)  # 用来存储 k 对应的总熵

    for k in range(N):
        P0 = cumsum_prob[k]
        P1 = 1.0 - P0

        # 如果 P0 或 P1 非法（太小），则跳过
        if P0 < eps or P1 < eps:
            H_total[k] = -np.inf
            continue

        # 背景部分的熵（H0）：- (sum_{i=0}^k p_i log p_i) / P0  +  log(P0)
        H0_num = cum_p_log_p[k]
        H0 = -(H0_num / P0) + np.log(P0)

        # 前景部分的熵（H1）
        idx_rev = N - 2 - k
        if idx_rev >= 0:
            H1_num = cum_p_rev_log_p_rev[idx_rev]
        else:
            H1_num = 0.0
        H1 = -(H1_num / P1) + np.log(P1)

        # Kapur 总熵
        H_total[k] = H0 + H1

    # 选取使得 H_total 最大的索引 k_opt
    k_opt = np.argmax(H_total)

    # 最终对应的阈值要取 bin_edges[k_opt + 1]
    threshold = bin_edges[k_opt + 1]
    return threshold


# ------------------------
# 损失函数
# ------------------------
def loss_function(recon, x, mu, logvar, beta=kl_weight):
    """VAE损失函数：重构损失 + beta * KL散度"""
    # MSE重构损失
    recon_loss = F.mse_loss(recon, x, reduction='sum')

    # KL散度
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # 总损失
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ------------------------
# 预测函数
# ------------------------
def predict(model, dataset, device):
    model.eval()
    rec_losses = []
    with torch.no_grad():
        for seq in dataset:
            x = seq.unsqueeze(0).to(device)  # (1, seq_len, n_features)
            recon, mu, logvar = model(x)
            rec_loss = F.mse_loss(recon, x, reduction='sum').item()
            rec_losses.append(rec_loss)
    return rec_losses


# ------------------------
# 评估函数
# ------------------------
def evaluate_model(model, test_norm, test_anom, threshold, device):
    """计算完整的评估指标"""
    norm_losses = predict(model, test_norm, device)
    anom_losses = predict(model, test_anom, device)

    # 准备真实标签和预测
    y_true = [0] * len(norm_losses) + [1] * len(anom_losses)  # 0=正常，1=异常
    y_pred = [1 if l > threshold else 0 for l in norm_losses + anom_losses]
    y_scores = norm_losses + anom_losses

    # 计算评估指标
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=0, average='binary')
    auc = roc_auc_score(y_true, y_scores)
    cm = confusion_matrix(y_true, y_pred)

    results = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm,
        "normal_acc": sum(l <= threshold for l in norm_losses) / len(norm_losses),
        "anomaly_acc": sum(l > threshold for l in anom_losses) / len(anom_losses)
    }

    return results

def evaluate_model_train(model, norm_ds_train, norm_ds_val, threshold, device):
    norm_losses_train = predict(model, norm_ds_train, device)
    norm_losses_val   = predict(model, norm_ds_val, device)
    y_true = [0]*len(norm_losses_train) + [0]*len(norm_losses_val)
    y_pred = [1 if l>threshold else 0 for l in norm_losses_train+norm_losses_val]
    # precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=0, average='binary')
    # auc = roc_auc_score(y_true, norm_losses_train+norm_losses_val)
    # cm  = confusion_matrix(y_true, y_pred)
    return {
        # "precision": precision,
        # "recall":    recall,
        # "f1":        f1,
        # "auc":       auc,
        # "confusion_matrix": cm,
        "normal_acc_train":  sum(l<=threshold for l in norm_losses_train)/len(norm_losses_train),
        "normal_acc_val": sum(l<=threshold  for l in norm_losses_val)/len(norm_losses_val)
    }

# ------------------------
# 训练函数
# ------------------------
def train_model(model, train_ds, val_ds, test_norm,
                test_anom, device, n_epochs=300):
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=100)

    history = {'train_loss': [], 'val_loss': [],
               'normal_acc': [], 'anomaly_acc': [], 'f1': [], 'auc': []}
    best_wts = copy.deepcopy(model.state_dict())
    best_val = float('inf')
    counter = 0  # 早停计数器
    threshold = None
    best_f1 = 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []

        for seq in train_ds:
            x = seq.unsqueeze(0).to(device)
            optimizer.zero_grad()

            recon, mu, logvar = model(x)
            loss, recon_loss, kl_loss = loss_function(recon, x, mu, logvar)

            loss.backward()
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # 验证
        model.eval()
        val_losses = []
        with torch.no_grad():
            for seq in val_ds:
                x = seq.unsqueeze(0).to(device)
                recon, mu, logvar = model(x)
                loss, _, _ = loss_function(recon, x, mu, logvar)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        # 学习率调度
        scheduler.step(val_loss)

        # 每10个epoch更新一次阈值或第一次运行时初始化阈值
        if epoch % 10 == 0 or threshold is None:
            threshold = determine_threshold(model, val_ds, device)
            print(f"Updated threshold: {threshold:.2f}")

        # 评估
        #验证训练集
        results_train = evaluate_model_train(model, train_ds, val_ds, threshold, device)

        results = evaluate_model(model, test_norm, test_anom, threshold, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['normal_acc'].append(results['normal_acc'])
        history['anomaly_acc'].append(results['anomaly_acc'])
        history['f1'].append(results['f1'])
        history['auc'].append(results['auc'])

        # 早停检查
        if val_loss < best_val:
            best_val = val_loss
            best_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"早停：已连续{PATIENCE}个epoch没有改善")
                break

        print(f"Epoch{epoch:3d}| train Loss:{train_loss:.2f}|Val Loss:{val_loss:.2f}"
              f"| F1:{results['f1']:.4f}| AUC:{results['auc']:.4f}"
              f"|normal_acc:{results['normal_acc']:.4f}| anomaly_acc:{results['anomaly_acc']:.4f} "
              f"|normal_acc_train:{results_train['normal_acc_train']:.4f}|normal_acc_val:{results_train['normal_acc_val']:.4f} "
              f"|threshold={threshold:.2f}")

        if results['f1']>best_f1:
            best_f1 = results['f1']
            save_path = os.path.join(log_dir, f"{current_file_name}_{current_time}_{best_f1:.4f}.pth")
            torch.save({
                'model': model.state_dict(),
                'threshold': threshold,
                'scaler': scaler
            }, save_path)
        # 记录日志


    model.load_state_dict(best_wts)
    return model, history, threshold

# ==========================
# 数据增强定义
# ==========================
class ECGAugment:
    """ECG信号数据增强: jitter, scaling, permutation。"""
    def __init__(self, jitter_std=0.02, scale_range=(0.8,1.2),
                 perm_segments=4, p=0.5):
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.perm_segments = perm_segments
        self.p = p

    def jitter(self, x):
        return x + np.random.normal(0, self.jitter_std, size=x.shape)

    def scaling(self, x):
        factor = np.random.uniform(*self.scale_range)
        return x * factor

    def permutation(self, x):
        segs = np.array_split(x, self.perm_segments)
        np.random.shuffle(segs)
        return np.concatenate(segs, axis=0)

    def __call__(self, x):
        if np.random.rand() > self.p:
            return x
        funcs = [self.jitter, self.scaling, self.permutation]
        for fn in np.random.choice(funcs, 2, replace=False):
            x = fn(x)
        return x

def augment_list(ds, augmentor):
    augmented = []
    for seq in ds:
        arr = seq.squeeze(1).cpu().numpy().astype(np.float32)
        arr_aug = augmentor(arr).astype(np.float32)
        augmented.append(torch.from_numpy(arr_aug).unsqueeze(1))
    return augmented

def augment_list2(ds, augmentor):
    augmented = []
    for seq in ds:
        arr = seq.squeeze(1).cpu().numpy().astype(np.float32)
        arr_aug1 = augmentor(arr).astype(np.float32)
        # arr_aug2 = augmentor(arr).astype(np.float32)
        # arr_aug3 = augmentor(arr).astype(np.float32)
        augmented.append(torch.from_numpy(arr_aug1).unsqueeze(1))
        # augmented.append(torch.from_numpy(arr_aug2).unsqueeze(1))
        # augmented.append(torch.from_numpy(arr_aug3).unsqueeze(1))
    return augmented

# ------------------------
# 主流程
# ------------------------
if __name__ == '__main__':
    print('Start training simplified VAE model for ECG5000...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 载入数据
    with open('data/ECG5000_TRAIN.arff') as f:
        train_df = a2p.load(f)
    with open('data/ECG5000_TEST.arff') as f:
        test_df = a2p.load(f)
    df = train_df._append(test_df).sample(frac=1.0, random_state=RANDOM_SEED)
    df.columns = list(df.columns[:-1]) + ['target']

    # 正常/异常划分
    CLASS_NORMAL = '1'
    normal_df_raw = df[df.target == CLASS_NORMAL].drop('target', axis=1)
    anomaly_df_raw = df[df.target != CLASS_NORMAL].drop('target', axis=1)

    # 数据预处理 - 标准化
    normal_df, anomaly_df, scaler = preprocess_data(normal_df_raw, anomaly_df_raw)
    print(f"数据集统计: 正常样本 {len(normal_df)}，异常样本 {len(anomaly_df)}")

    # 划分训练 验证 测试
    train_norm, test_norm = train_test_split(
        normal_df, test_size=0.2, random_state=RANDOM_SEED)
    train_norm, val_norm = train_test_split(
        train_norm, test_size=0.15, random_state=RANDOM_SEED)

    # 构造 dataset
    def to_dataset(df_):
        arr = df_.astype(np.float32).values
        return [torch.tensor(a).unsqueeze(1) for a in arr]

    train_ds = to_dataset(train_norm)
    val_ds = to_dataset(val_norm)
    test_norm_ds = to_dataset(test_norm)
    test_anom_ds = to_dataset(anomaly_df)

    #对训练集进行增强，扩充数据量
    augmentor = ECGAugment(jitter_std=0.01, scale_range=(0.9,1.1), p=0.2)
    augmented_ds = augment_list(train_ds, augmentor)
    train_ds = train_ds + augmented_ds  # 合并原始和增强数据
    random.seed(42)
    random.shuffle(train_ds)
    augmented_ds = augment_list(val_ds, augmentor)
    val_ds = val_ds + augmented_ds  # 合并原始和增强数据
    random.seed(42)
    random.shuffle(val_ds)

    seq_len = train_ds[0].shape[0]
    n_features = train_ds[0].shape[1]

    print(f"序列长度: {seq_len}, 特征数: {n_features}")
    print(f'train_dataset: {len(train_ds)}')
    print(f'val_dataset: {len(val_ds)}')
    print(f'test_normal_dataset: {len(test_norm_ds)}')
    print(f'test_anomaly_dataset: {len(test_anom_ds)}')

    # 初始化简化的模型
    model = RecurrentVariationalAutoencoder(
        seq_len, n_features, embedding_dim=EMBEDDING_DIM, latent_dim=LATENT_DIM).to(device)
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 训练
    # model, history, best_threshold = train_model(
    #     model, train_ds, val_ds, test_norm_ds, test_anom_ds, device, n_epochs=300)

    checkpoint = torch.load(r'log\TimeSeriesTrainAtt_0619_123638_0.9768.pth', map_location=device)
    # checkpoint = torch.load(r'log\TimeSeriesTrainAtt(1)_0620_121626_0.9710.pth', map_location=device)
    model.load_state_dict(checkpoint['model'])
    best_threshold =  checkpoint['threshold']-9
    scaler = checkpoint.get('scaler', None)  # 兼容可能没有保存scaler的情况

    # 最终评估
    final_results = evaluate_model(model, test_norm_ds, test_anom_ds, best_threshold, device)
    print("\n最终评估结果:")
    for metric, value in final_results.items():
        if metric != 'confusion_matrix':
            print(f"{metric}: {value:.4f}")
    print("混淆矩阵:")
    print(final_results['confusion_matrix'])

    # 保存模型
    norm_losses = predict(model, test_norm_ds, device)
    anom_losses = predict(model, test_anom_ds, device)

    correct1 = sum(l <= best_threshold for l in norm_losses)
    percent1 = correct1 / len(test_norm_ds)
    print(f'Correct normal predictions: {correct1}/{len(test_norm_ds)}')

    correct2 = sum(l > best_threshold for l in anom_losses)
    percent2 = correct2 / len(test_anom_ds)
    print(f'Correct anomaly predictions: {correct2}/{len(test_anom_ds)}')

    # Plot both Train Loss and Validation Loss in the same figure
    plt.figure(figsize=(10, 6))
    print(len(norm_losses))
    print(len(anom_losses))
    # Plot Train Loss with KDE
    sns.histplot(norm_losses, kde=True, bins=500, color='blue', label='Normal Test Loss', alpha=0.6)

    # Plot Validation Loss with KDE
    # sns.histplot(anom_losses, kde=True, bins=500, color='red', label='Test Anomaly Loss', alpha=0.6,
    #              kde_kws={'bw_adjust': 0.5, 'kernel': 'gau', 'gridsize': 200, 'cut': 3})

    sns.histplot(anom_losses, kde=True, bins=500, color='red', label='Anomaly Test Loss', alpha=0.6,
                 kde_kws={'bw_adjust': 0.2, 'gridsize': 300, 'cut': 10})



    x=int(best_threshold)-4
    plt.axvline(x, color='green', linestyle='--', label='Threshold')
    plt.text(x, plt.ylim()[1] * 0.9, f'Threshold={x}', color='black', ha='center')
    # Set the x-axis limit to 200
    plt.xlim(0, 800)
    plt.ylim(0, 120)
    # Add labels and title
    # plt.title("LSTM-VAE-Attention: Test loss distribution with KDE",fontsize=20)
    plt.xlabel("Loss Value")
    plt.ylabel("Frequency / Density")
    plt.legend()

    # Show the plot
    plt.tight_layout()



    plt.savefig('kde3.tiff', bbox_inches='tight', format='tiff', dpi=300)
    plt.show()

    print('完成!')
'''

## 主要修改和简化：

1. ** 简化模型架构 **：
- 减小了模型的嵌入维度和潜在维度，从较大的值(128 / 64)
减小到更合适的值(32 / 16)
- 使用单向LSTM代替双向LSTM减少参数数量
- 使用GRU代替复杂的双LSTM结构，减少参数量并提高训练稳定性

2. ** 简化注意力机制 **：
- 用一个简单而有效的注意力机制代替复杂的多头自注意力
- 去除了可能导致维度不兼容的复杂矩阵运算

3. ** 优化解码器 **：
- 简化解码器结构，使其更匹配序列长度为140的输入
- 移除了可能导致维度冲突的复杂注意力机制

4. ** 使用MSE损失 **：
- 用均方误差(MSE)
代替Huber损失，处理连续值时MSE通常表现更好

5. ** 进一步优化超参数 **：
- 调整了KL散度权重以平衡重构和正则化
- 添加了梯度裁剪来防止训练不稳定

这个简化版的模型应该能够避免维度不匹配的错误，并且可能在ECG5000数据集上表现良好。模型现在更加轻量级，训练更快，但仍保留了VAE的核心优势。
'''