import numpy as np
import torch as T
import torch.utils.data
import torch.optim as optim
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from dnc import FFSAM
from dnc.util import cuda
from testnet import Net

df = pd.read_csv('amb/sets/cancer_train.csv')
df = pd.get_dummies(df, columns=['Diagnosis'], drop_first=True)
xdf = df.loc[:, [col for col in df.columns if col not in ['Diagnosis_MALIGNANT', 'Code']]].values
ydf = df.loc[:, 'Diagnosis_MALIGNANT'].values
x_train, x_test, y_train, y_test = train_test_split(xdf, ydf, test_size=0.15)
x_train = T.from_numpy(x_train).float()
x_test = T.from_numpy(x_test).float()
y_train = T.from_numpy(y_train).float()
y_test = T.from_numpy(y_test).float()

rnn = FFSAM(
  input_size=x_train.size(1),
  hidden_size=64,
  output_size=2,
  num_layers=4,
  nr_cells=100,
  cell_size=32,
  read_heads=4,
  sparse_reads=4,
  gpu_id=0
)
alt_net = False
ff = Net(x_train.size(1), 2, 64, cuda=True)

(memory, read_vectors) = (None, None)

num_epochs = 20
dataset_train = T.utils.data.TensorDataset(x_train, y_train)
dl_train = T.utils.data.DataLoader(dataset_train, shuffle=True, batch_size=32, drop_last=True)
dataset_test = T.utils.data.TensorDataset(x_test, y_test)
dl_test = T.utils.data.DataLoader(dataset_test, shuffle=True, batch_size=32, drop_last=True)
loss_func = T.nn.NLLLoss()
if alt_net:
    optimizer = optim.Adam(ff.parameters(), lr=1e-4)
else:
    optimizer = optim.Adam(rnn.parameters(), lr=1e-4)
for i in range(num_epochs):
    print('Epoch: {}'.format(i))
    for batch, labels in dl_train:
        optimizer.zero_grad()
        if alt_net:
            output = ff(cuda(batch, gpu_id=0))
        else:
            output, (memory, read_vectors) = rnn(cuda(batch, gpu_id=0), (memory, read_vectors), reset_experience=True)
        loss = loss_func(output.float(), cuda(labels, gpu_id=0).long())
        print('Loss: {}'.format(loss.data[0]))
        loss.backward()
        optimizer.step()
print('Testing')
ytrue, ypred = [], []
for batch, labels in dl_test:
    if alt_net:
        output = ff(cuda(batch, gpu_id=0))
    else:
        output, (memory, read_vectors) = rnn(cuda(batch, gpu_id=0), (memory, read_vectors), reset_experience=True)
    ypred.extend(T.max(output.cpu(), 1)[1].data.numpy().astype(int).tolist())
    ytrue.extend(labels.numpy().astype(int).tolist())
f1 = f1_score(ytrue, ypred)
print('Test size: {}'.format(len(ytrue)))
print('F1 score: {}'.format(f1))
print('Accuracy: {}'.format(1. - len(np.where((np.array(ytrue) - np.array(ypred)) != 0)[0]) / len(ytrue)))
