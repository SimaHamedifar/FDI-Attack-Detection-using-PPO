
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.seed import set_seed
from utils.plotting import plot_ieee_style, plot_rewards
set_seed()

project_root = Path(__file__).resolve.parents[1]
data_path = project_root/'data'/'CLEAN_House2.csv'

df = pd.read_csv(data_path, parse_dates=False)
df.head()

#date_time = pd.to_datetime(df.pop('Time'), format='%Y-%m-%d %H:%M:%S')
df['Time'] = pd.to_datetime(df['Time'],yearfirst=True)

datatime = df.set_index("Time", inplace=True)

downsampled_data = df.resample('15min').mean().ffill()
downsampled_data.head()
downsampled_data.tail()

downsampled_data.pop('Issues')
downsampled_data.pop('Unix')
downsampled_data.head()

downsampled_data['hour'] = downsampled_data.index.hour
downsampled_data['minute'] = downsampled_data.index.minute
downsampled_data.loc[:,'hour_decimal'] = downsampled_data['hour'] + downsampled_data['minute']/60
downsampled_data.loc[:,'hour_cosin'] = np.cos(2 * np.pi * downsampled_data['hour_decimal']/24)
downsampled_data.loc[:,'hour_sin'] = np.sin(2 * np.pi * downsampled_data['hour_decimal']/24)
downsampled_data.loc[:,'minute_cosin'] = np.cos(2 * np.pi * downsampled_data['minute']/60)
downsampled_data.loc[:,'minute_sin'] = np.sin(2 * np.pi * downsampled_data['minute']/60)
downsampled_data.pop('hour_decimal')
downsampled_data.pop('Appliance1')
downsampled_data.pop('Appliance2')
downsampled_data.pop('Appliance3')
downsampled_data.pop('Appliance4')
downsampled_data.pop('Appliance5')
downsampled_data.pop('Appliance6')
downsampled_data.pop('Appliance7')
downsampled_data.pop('Appliance8')
downsampled_data.pop('Appliance9')
downsampled_data.head()

start_missing = '2013-10-25'
end_missing = '2014-02-14'
constant_period = downsampled_data.loc[start_missing:end_missing]
#
replacement_start = '2014-10-25'
replacement_end = '2015-02-14'
replacement_period = downsampled_data.loc[replacement_start:replacement_end]

replacement_avg = replacement_period.groupby([replacement_period.index.month,
                                              replacement_period.index.day,
                                              replacement_period.index.hour,
                                              replacement_period.index.minute])['Aggregate'].mean()

def impute(row):
    month = row.name.month
    day = row.name.day
    hour = row.name.hour
    minute = row.name.minute
    try:
        return replacement_avg.loc[(month, day, hour, minute)]
    except KeyError:
        return None
    
downsampled_data.loc[start_missing:end_missing, 'Aggregate'] = constant_period.apply(impute, axis=1)

downsampled_data.Aggregate.isna().sum()

plt.figure(figsize=(15,5))
plt.plot(downsampled_data.index, downsampled_data.Aggregate.values)
plt.show()

downsampled_data.loc['2014-09-13':'2014-09-15','Aggregate'] = downsampled_data.loc['2014-09-10':'2014-09-12','Aggregate'].values
downsampled_data.loc['2014-10-05','Aggregate'] = downsampled_data.loc['2014-10-04','Aggregate'].values
downsampled_data.loc['2014-10-18':'2014-10-19','Aggregate'] = downsampled_data.loc['2014-10-16':'2014-10-17','Aggregate'].values
downsampled_data.loc['2014-11-01':'2014-11-10','Aggregate'] = downsampled_data.loc['2014-11-11':'2014-11-20','Aggregate'].values

#%%
plt.figure(figsize=(15,5))
plt.plot(downsampled_data.index, downsampled_data[['Aggregate']])
plt.show()
# Convert Wh to kWh
downsampled_data[['Aggregate']] = downsampled_data[['Aggregate']] * 0.001

rolling_mean = downsampled_data['Aggregate'].rolling('1D', min_periods=48).mean()
rolling_std = downsampled_data['Aggregate'].rolling('1D', min_periods=48).std()

downsampled_data['is_outlier'] = (downsampled_data['Aggregate'] - rolling_mean).abs() > 1 * rolling_std
downsampled_data.loc[downsampled_data['is_outlier'], 'Aggregate'] = np.nan
downsampled_data['Aggregate'] = downsampled_data['Aggregate'].interpolate(method='time')
downsampled_data.pop('is_outlier')


downsampled_data['Aggregate'].isna().sum()


project_root = Path(__file__).resolve.parents[1]
data_path = project_root/'data'/'REFIT_H2_Preprocessed.csv'
downsampled_data.to_csv(data_path, index=False)
