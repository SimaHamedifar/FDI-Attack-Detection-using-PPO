import numpy as np
import pandas as pd
from pathlib import Path
from utils.seed import set_seed

set_seed()

project_root = Path(__file__).resolve.parents[1]
data_path = project_root/'data'/'predictions.csv'

data = pd.read_csv(data_path)

data['Time'] = pd.to_datetime(data['Time'], yearfirst=True)
data.set_index(['Time'], inplace=True)

start = data.index[0]
stop = data.index[-1]

start = data.index[0]
stop = data.index[-1]
attack_steps = pd.date_range(start=start, end=stop, freq='100T').astype(str).tolist()

data['Scaling_Attacked_Value_1'] = data['Aggregate'].values
data['Scaling_Attacked_Value_2'] = data['Aggregate'].values

data['Random_Attacked_Value_1'] = data['Aggregate'].values
data['Random_Attacked_Value_2'] = data['Aggregate'].values

data['Attacked'] = np.zeros(len(data))
data['Non_Attacked'] = np.ones(len(data))


def inject_scaling_attack(data, Attack_Steps, scaling_factor_1, scaling_factor_2):

    for i in range(len(Attack_Steps) - 1):
        start = pd.Timestamp(Attack_Steps[i])
        stop_1 = start + pd.Timedelta(minutes=20)

        # Apply scaling attack
        data.loc[start:stop_1, 'Scaling_Attacked_Value_1'] += (
            scaling_factor_1 * data.loc[start:stop_1, 'Scaling_Attacked_Value_1']
        )
        data.loc[start:stop_1, 'Scaling_Attacked_Value_2'] += (
            scaling_factor_2 * data.loc[start:stop_1, 'Scaling_Attacked_Value_2']
        )
        data.loc[start:stop_1, 'Attacked'] = 1
        data.loc[start:stop_1, 'Non_Attacked'] = 0
        
    return data

attacked_data = inject_scaling_attack (data, Attack_Steps = attack_steps, scaling_factor_1 = 0.15, scaling_factor_2 = 0.2)


def inject_Random_attack(data, Attack_Steps, scaling_factor_1, scaling_factor_2):

    for i in range(len(Attack_Steps) - 1):
        start = pd.Timestamp(Attack_Steps[i])
        stop_2 = pd.Timestamp(Attack_Steps[i + 1])

        random_timesteps = data.loc[start:stop_2].sample(20).index
        data.loc[random_timesteps, 'Random_Attacked_Value_1'] += (
            scaling_factor_1 * data.loc[random_timesteps, 'Random_Attacked_Value_1']
        )
        data.loc[random_timesteps, 'Random_Attacked_Value_2'] += (
            scaling_factor_2 * data.loc[random_timesteps, 'Random_Attacked_Value_2']
        )
        data.loc[random_timesteps, 'Attacked'] = 1
        data.loc[random_timesteps, 'Non_Attacked'] = 0

    return data
attacked_data = inject_Random_attack (data, Attack_Steps = attack_steps, scaling_factor_1 = 0.15, scaling_factor_2 = 0.2)

dataset = pd.DataFrame({'Measured': data.Aggregate,
                        'Predicted': data.Aggregate, 
                        'Scaling_Attacked_1': data.Scaling_Attacked_Value_1,
                        'Scaling_Attacked_2': data.Scaling_Attacked_Value_2,
                        'Random_Attacked_Value_1': data.Random_Attacked_Value_1,
                        'Random_Attacked_Value_2': data.Random_Attacked_Value_2,
                        'Non_Attacked': data.Non_Attacked, 
                        'Attacked': data.Attacked}, index=data.index)

data_save_path = project_root/'data'/'Attacks_Data_15_20.csv'
dataset.to_csv(data_save_path, index=True, index_label='DateTime')



