
from pathlib import Path
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from keras import metrics
import tensorflow.keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from utils.seed import set_seed
from utils.plotting import plot_ieee_style, plot_rewards
set_seed()

project_root = Path(__file__).resolve.parents[1]
data_path = project_root/'data'/'REFIT_H2_Preprocessed.csv'

df = pd.read_csv(data_path)
df.head()

df['Time'] = pd.to_datetime(df['Time'],yearfirst=True)
datatime = df.set_index("Time", inplace=True)

train_set = df.loc["2014-03-10":"2014-08", ['Aggregate','hour', 'minute']]
test_set = df.loc["2014-09":, ['Aggregate','hour', 'minute']]

Scaler_power = MinMaxScaler()
Fit_power = Scaler_power.fit(train_set[['Aggregate']])
train_set['Aggregate'] = Fit_power.transform(train_set[['Aggregate']])
test_set['Aggregate'] = Fit_power.transform(test_set[['Aggregate']])

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 500
X_train, y_train = create_dataset(train_set, train_set.Aggregate, time_steps)
X_test, y_test = create_dataset(test_set, test_set.Aggregate, time_steps)
print(X_train.shape, y_train.shape)

model = models.Sequential()
model.add(layers.Bidirectional(
    layers.LSTM(72, activation='tanh', return_sequences=True), 
    input_shape=(X_train.shape[1], X_train.shape[2])
))
model.add(layers.Dropout(0.1))
model.add(layers.LSTM(72, activation='tanh', return_sequences=False))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(56, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
optimizer = keras.optimizers.Adam(learning_rate=1e-4)
loss = keras.losses.MeanSquaredError()
model.compile(loss=loss, optimizer=optimizer, metrics=[metrics.MeanSquaredError(), metrics.MeanAbsoluteError()])
history = model.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=32, 
    validation_split=0.2,
    shuffle=False
)

train_hist_save_path = project_root/'results'/'predictor_train_val_loss.png'
x = np.arange(0, len(history.history['loss']))
y_dict = {'train_loss' : history.history['loss'], 'validation_loss': history.history['val_loss']}
xlabel = 'Epoch'
ylabel = 'Loss'
save_path = project_root / 'results' / 'predictor_train_val_loss.png.'
plot_ieee_style(x, y_dict, xlabel, ylabel, title='', save_path=save_path, dpi=300)

y_pred = model.predict(X_test)

model.evaluate(X_test,y_test)

y_train_inv = Fit_power.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = Fit_power.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = Fit_power.inverse_transform(y_pred.reshape(-1, 1))

preds_pd = pd.Series(y_pred_inv.flatten())
preds_pd.index = test_set.index[time_steps:]
preds_df = pd.DataFrame(preds_pd, columns=['Predicted_Aggregate_Power'])
preds_df.head()

gt_pd = pd.Series(y_test_inv.flatten(), name="Aggregate")
gt_pd.index = test_set.index[time_steps:]

predictor_test_save_path = project_root/'results'/'predictor_test.png'
x = np.arange(0, len(gt_pd.loc['2014-09-10']))
y_dict = {'Ground Truth' : gt_pd.loc['2014-09-10'], 'Prediction': preds_pd.loc['2014-09-10']}
xlabel = 'Time'
ylabel = 'Power (kW)'
save_path = project_root / 'results' / 'predictor_test.png.'
plot_ieee_style(x, y_dict, xlabel, ylabel, title='', save_path=save_path, dpi=300)

data_path = project_root/'data'/'predictions.csv'
preds_df.to_csv(data_path)

keras.models.save_model(model, 'Deep_BiLSTM_Model.keras')

mae = K.mean(K.abs(y_pred_inv - y_test_inv)).numpy()
mse = K.mean(K.square(y_pred_inv - y_test_inv)).numpy()
print(mae, mse)






