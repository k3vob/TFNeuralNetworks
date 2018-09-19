import pandas as pd
import os

from RNN import RNN


data = pd.read_csv('/home/kevin/Downloads/cashflow.csv')
data.rename(columns={'Client_ID': 'ID', 'Yearly_income': 'Inc', 'Yearly_expends': 'Exp'}, inplace=True)
data['ID'] = data['ID'].apply(lambda str: int(str.split('_')[-1]))
ids = data['ID'].unique()
data.set_index(['ID', 'Date'], inplace=True)
data = data.iloc[:, -2:]
data[['Inc_Label', 'Exp_Label']] = data[['Inc', 'Exp']].shift(-1)
idx = [data.xs(id, drop_level=False).index[-1] for id in ids]
data.drop(idx, inplace=True)

data = (data - data.min()) / (data.max() - data.min())

model = RNN(
    num_inputs=2,
    num_outputs=2,
    hidden_sizes=[100, 100],
    num_unrollings=86,
    cell='LSTM'
)

model.set_data(data)
model.train(epochs=100, learning_rate=0.01, batch_size=5, print_step=10)
