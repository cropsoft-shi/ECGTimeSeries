import torch
import copy
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from torch import nn, optim
import torch.nn.functional as F
from arff2pandas import a2p
import csv,datetime,os
from sklearn.preprocessing import StandardScaler

DRAW=True# True False
THRESHOLD = 26

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# __file__ 表示当前脚本的完整路径（如果运行环境支持该变量）
current_file_path = __file__
current_file_name = os.path.basename(current_file_path).split('.')[0]# 提取文件名
current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
fileName= f'{current_file_name}_{current_time}'
csv_filename = f"log/{fileName}.csv"
with open(csv_filename, mode='w', newline='') as f:
  writer = csv.writer(f)
  writer.writerow(["Epoch", "Train Loss", "Val Loss","Normal Acc", "Anomaly Acc"])

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
plt.rcParams['axes.unicode_minus'] = False
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

def plot_time_series_class(data, class_name, ax, n_steps=10):
  time_series_df = pd.DataFrame(data)

  smooth_path = time_series_df.rolling(n_steps).mean()
  path_deviation = 2 * time_series_df.rolling(n_steps).std()

  under_line = (smooth_path - path_deviation)[0]
  over_line = (smooth_path + path_deviation)[0]

  ax.plot(smooth_path, linewidth=2)
  ax.fill_between(
    path_deviation.index,
    under_line,
    over_line,
    alpha=.125
  )
  ax.set_title(class_name)

def create_dataset(df):
  sequences = df.astype(np.float32).to_numpy().tolist()
  dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
  n_seq, seq_len, n_features = torch.stack(dataset).shape
  return dataset, seq_len, n_features

class Encoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()

    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )

  def forward(self, x):
    x = x.reshape((1, self.seq_len, self.n_features))

    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)

    return hidden_n.reshape((self.n_features, self.embedding_dim))

class Decoder(nn.Module):
  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(Decoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )

    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    self.output_layer = nn.Linear(self.hidden_dim, n_features)

  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))

    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = x.reshape((self.seq_len, self.hidden_dim))

    return self.output_layer(x)

class RecurrentAutoencoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(RecurrentAutoencoder, self).__init__()

    self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)

    return x

def predict(model, dataset):
  predictions, losses = [], []
  criterion = nn.L1Loss(reduction='sum').to(device)#sum  mean
  with torch.no_grad():
    model = model.eval()
    for seq_true in dataset:
      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)

      loss = criterion(seq_pred, seq_true)

      predictions.append(seq_pred.cpu().numpy().flatten())
      losses.append(loss.item())
  return predictions, losses

def train_model(model, train_dataset, val_dataset,
                test_normal_dataset,test_anomaly_dataset,n_epochs):
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.L1Loss(reduction='sum').to(device)#sum  mean
  history = dict(train=[], val=[], normal_percent=[], anomaly_percent=[])
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0

  for epoch in range(1, n_epochs + 1):
    model = model.train()

    train_losses = []
    for seq_true in train_dataset:
      optimizer.zero_grad()

      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)

      loss = criterion(seq_pred, seq_true)

      loss.backward()
      optimizer.step()

      train_losses.append(loss.item())

    val_losses = []
    model = model.eval()
    with torch.no_grad():
      for seq_true in val_dataset:
        seq_true = seq_true.to(device)
        seq_pred = model(seq_true)

        loss = criterion(seq_pred, seq_true)
        val_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    # 新增：使用 predict 函数对 test_normal_dataset  anomaly_dataset  进行预测 个数相同
    _, pred_losses  = predict(model, test_normal_dataset)
    correct = sum(l <= THRESHOLD for l in pred_losses)
    normal_percent = correct / len(test_normal_dataset)
    # print(THRESHOLD)
    # print(correct, len(test_normal_dataset))

    _, pred_losses  = predict(model, test_anomaly_dataset)
    correct = sum(l > THRESHOLD for l in pred_losses)
    anomaly_percent = correct / len(test_anomaly_dataset)
    # print(correct,len(test_anomaly_dataset))

    history['normal_percent'].append(normal_percent)
    history['anomaly_percent'].append(anomaly_percent)

    history['train'].append(train_loss)
    history['val'].append(val_loss)

    if val_loss < best_loss:
      best_loss = val_loss
      best_model_wts = copy.deepcopy(model.state_dict())

    # 每个 epoch 结束后将结果追加到 CSV 文件中
    with open(csv_filename, mode='a', newline='') as f:
      writer = csv.writer(f)
      writer.writerow([epoch, train_loss, val_loss,normal_percent,anomaly_percent])
    print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss} '
          f'normal_percent:{normal_percent} anomaly_percent:{anomaly_percent}')
  model.load_state_dict(best_model_wts)
  return model.eval(), history

# ------------------------
# 预处理函数 (与训练一致)
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

if __name__ == '__main__':
  print('start...')
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

  '''
  只对normal训练
  '''
  train_norm, test_norm = train_test_split(
    normal_df, test_size=0.2, random_state=RANDOM_SEED)
  train_norm, val_norm = train_test_split(
    train_norm, test_size=0.1, random_state=RANDOM_SEED)

  test_normal_dataset, _, _ = create_dataset(test_norm)
  test_anomaly_dataset, _, _ = create_dataset(anomaly_df)

  # anomaly_dataset = test_anomaly_dataset[:len(test_normal_dataset)]

  train_dataset, seq_len, n_features = create_dataset(train_norm)
  val_dataset, _, _ = create_dataset(val_norm)

  print(f'train_dataset: {len(train_dataset)}')
  print(f'val_dataset: {len(val_dataset)}')
  print(f'test_normal_dataset: {len(test_normal_dataset)}')
  print(f'test_anomaly_dataset: {len(test_anomaly_dataset)}')

  '''    '''
  model = RecurrentAutoencoder(seq_len, n_features, 128)
  model = model.to(device)

  model, history = train_model(
    model,
    train_dataset,val_dataset,
    test_normal_dataset,test_anomaly_dataset,
    n_epochs=300
  )

  if DRAW:
    ax = plt.figure().gca()
    ax.plot(history['train'])
    ax.plot(history['val'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'])
    plt.title('Loss over training epochs')
    plt.show()

  MODEL_PATH = f'log/{fileName}.pth'
  torch.save(model, MODEL_PATH)

  # model = torch.load('log/TimeSeriesTrain/TimeSeriesTrain_0415_192723.pth', map_location=device)  # Auto-maps to GPU if available

  _, losses = predict(model, train_dataset)

  if DRAW:
    sns.histplot(losses, kde=True)  # 直方图+KDE
    plt.show()

  predictions, pred_losses = predict(model, test_normal_dataset)
  if DRAW:
    sns.histplot(pred_losses, bins=50, kde=True)
    plt.show()

  correct1 = sum(l <= THRESHOLD for l in pred_losses)
  percent1 = correct1 / len(test_normal_dataset)
  print(f'Correct normal predictions: {correct1}/{len(test_normal_dataset)}')

  anomaly_dataset = test_anomaly_dataset[:len(test_normal_dataset)]
  predictions, pred_losses = predict(model, anomaly_dataset)
  #sns.histplot(pred_losses, bins=50, kde=True);
  correct2 = sum(l > THRESHOLD for l in pred_losses)
  percent2 = correct2 / len(anomaly_dataset)
  print(f'Correct anomaly predictions: {correct2}/{len(anomaly_dataset)}')

  with open(csv_filename, mode='a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([0, f'{correct1}/{len(test_normal_dataset)}', f'{correct2}/{len(anomaly_dataset)}',
                     percent1, percent2])

  def plot_prediction(data, model, title, ax):
    predictions, pred_losses = predict(model, [data])
    ax.plot(data, label='true')
    ax.plot(predictions[0], label='reconstructed')
    ax.set_title(f'{title} (loss: {np.around(pred_losses[0], 2)})')
    ax.legend()

  if DRAW:
    fig, axs = plt.subplots(
      nrows=2,
      ncols=6,
      sharey=True,
      sharex=True,
      figsize=(22, 8)
    )

    for i, data in enumerate(test_normal_dataset[:6]):
      plot_prediction(data, model, title='Normal', ax=axs[0, i])
    for i, data in enumerate(test_anomaly_dataset[:6]):
      plot_prediction(data, model, title='Anomaly', ax=axs[1, i])
    fig.tight_layout();

  print("Doned!")

