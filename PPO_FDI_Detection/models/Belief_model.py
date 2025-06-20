from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
from utils.seed import set_seed
from utils.plotting import plot_ieee_style 

set_seed()

project_root = Path(__file__).resolve.parents[1]
data_path = project_root/'data'/'Attacks_Data_15_20.csv'

df = pd.read_csv(data_path)
df.head()

df['Time'] = pd.to_datetime(df['Time'],yearfirst=True)
df.set_index("Time", inplace=True)
df.columns

df['hour'] = df.index.hour
df['minute'] = df.index.minute
df.loc[:,'hour_decimal'] = df['hour'] + df['minute']/60
df.loc[:,'hour_cosin'] = np.cos(2 * np.pi * df['hour_decimal']/24)
df.loc[:,'hour_sin'] = np.sin(2 * np.pi * df['hour_decimal']/24)
df.loc[:,'minute_cosin'] = np.cos(2 * np.pi * df['minute']/60)
df.loc[:,'minute_sin'] = np.sin(2 * np.pi * df['minute']/60)
df.pop('hour')
df.pop('minute')
df.columns

# This is repeated for each type of data. 
train_data = df.loc['2014-09':'2014-11', ['Predictions', 'Scaling_Attacked_Value_1', 
                        'hour_cosin', 'minute_cosin', 
                        'Attacked', 
                        'Non_Attacked']]

train_data_features = df.loc[:, ['Aggregate', 'Scaling_Attacked_Value_2', 'hour_cosin', 'hour_sin' , 'minute_cosin', 'minute_sin']]
train_data_labels = df.loc[:, ['Attacked', 'Non_Attacked']]


Scaler = MinMaxScaler()
train_data_features[['Aggregate', 'Scaling_Attacked_Value_2']] = Scaler.fit_transform(train_data_features[['Aggregate', 'Scaling_Attacked_Value_2']])

def slice_dataset (data, num_samples, window_size, step_size = 1):
    feature_sequence = []
    for i in range(0, num_samples - window_size + 1, step_size):
        feature_sequence.append(data[i : i+window_size])
    return np.array(feature_sequence)  #, np.array(label_sequence)


num_samples = len(train_data)
window_size = 400
train_X_sliced = slice_dataset(train_data_features, num_samples=num_samples, window_size=window_size)
train_Y_sliced = slice_dataset(train_data_labels, num_samples=num_samples, window_size=window_size)

class AttackedDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.argmax(torch.tensor(labels, dtype=torch.float32), dim=2) # Shape: (num_windows, window_size)

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

data_train = AttackedDataset(features=train_X_sliced, labels=train_Y_sliced)

batch_size = 32
train_dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=True) # shuffle=True only when training

class EstimateBeliefsWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(EstimateBeliefsWithAttention, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        atten_output, _ = self.attention(x, x, x)
        logits = self.fc2(atten_output)
        return logits 

belief_model = EstimateBeliefsWithAttention(input_dim=6, hidden_dim=32, num_heads=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
total_samples = len(train_data)
num_non_attack = len(train_data[train_data['Non_Attacked_scaling'] == 1])
w_non_attack = total_samples / (2 * num_non_attack)
num_attack = len(train_data[train_data['Attacked_scaling'] == 1])
w_attack = total_samples / (2 * num_attack)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([w_attack, w_non_attack]))
optimizer = torch.optim.Adam(belief_model.parameters(), lr=1e-4)
num_epochs=100

val_data = df.loc['2015', ['Aggregate', 'Scaling_Attacked_Value_2', 'hour_cosin', 
                           'hour_sin', 'minute_cosin', 'minute_sin']]

val_data_features = df.loc['2015':, ['Aggregate', 'Scaling_Attacked_Value_2', 
                                     'hour_cosin', 'hour_sin' , 'minute_cosin', 'minute_sin']]

val_data_labels = df.loc['2015':, ['Attacked', 'Non_Attacked']]

val_data_features[['Aggregate', 'Scaling_Attacked_Value_2']]= Scaler.transform(
    val_data_features[['Aggregate', 'Scaling_Attacked_Value_2']])

val_X = slice_dataset(data=val_data_features, num_samples=len(val_data), window_size=window_size)
val_Y = slice_dataset(data=val_data_labels, num_samples=len(val_data), window_size=window_size)

val_dataset = AttackedDataset(features=val_X, labels=val_Y)

val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

def train_model_with_early_stopping(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, device, patience=20):
    episodic_loss = []
    val_losses = []
    best_model_weights = deepcopy(model.state_dict())
    best_val_loss = float('inf')
    patience_counter = 0

    model.to(device)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)  # Shape: (batch_size, seq_len, num_classes)
            outputs = outputs.view(-1, outputs.size(-1))  # Reshape to (batch_size * seq_len, num_classes)
            targets = targets.view(-1)  # Reshape to (batch_size * seq_len)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        episodic_loss.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Load the best model weights
    model.load_state_dict(best_model_weights)
    return episodic_loss, val_losses

episodic_loss, val_losses = train_model_with_early_stopping(model=belief_model, 
                                                            train_dataloader=train_dataloader, 
                                                            val_dataloader=val_dataloader, 
                                                            criterion=criterion, 
                                                            optimizer=optimizer, 
                                                            num_epochs=num_epochs,
                                                            device=device, 
                                                            patience=10)

def evaluate_model(model, dataloader, device):
    model.eval()
    TP = 0  # True Positives
    FP = 0  # False Positives
    TN = 0  # True Negatives
    FN = 0  # False Negatives
    correct = 0
    total = 0
    probabilities = []
    Inputs = []
    Predictions = []
    Targets = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            # Move data to the device
            inputs, targets = inputs.to(device), targets.to(device)  # targets shape: (batch_size,)
            Inputs.append(inputs)
            # Forward pass
            logits = model(inputs)  # logits shape: (batch_size, num_classes)
            probabilities_ = torch.softmax(logits, dim=2)
            probabilities.append(probabilities_)
            predictions = torch.argmax(probabilities_, dim=2)  # Shape: (batch_size,)
            Predictions.append(predictions)
            Targets.append(targets)

            # Update TP, FP, TN, FN
            TP += ((predictions == 1) & (targets == 1)).sum().item()
            FP += ((predictions == 1) & (targets == 0)).sum().item()
            TN += ((predictions == 0) & (targets == 0)).sum().item()
            FN += ((predictions == 0) & (targets == 1)).sum().item()

            # Update accuracy metrics
            correct += (predictions == targets).sum().item()
            total += targets.numel()  # Total number of elements (batch_size * seq_length)

    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0  # Recall
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    
    return probabilities, Inputs, Targets, Predictions

#belief_model = torch.load("belief_model_Seq_20.pth", weights_only=False)

probabilities, Inputs, Targets, Predictions = evaluate_model(belief_model, train_dataloader, device)

probabilities_np = [prob_batch.cpu().numpy() for prob_batch in probabilities] 
probabilities_full = np.concatenate(probabilities_np, axis = 0)  
targets_np = [targ_batch.cpu().numpy() for targ_batch in Targets] 

targets_full = np.concatenate(targets_np, axis = 0) 
Inputs_np = [inputs_batch.cpu().numpy() for inputs_batch in Inputs]
Inputs_full = np.concatenate(Inputs_np, axis=0) 
real_seq_scaled = Inputs_full.reshape(-1, 6)  
real_seq = Scaler.inverse_transform(real_seq_scaled[:,0:2])
time_indices = np.arange(0, len(real_seq))
targets_full = targets_full.reshape(-1,)
targets_full = 1 - targets_full
beliefs = probabilities_full.reshape(-1,2)
dataset = pd.DataFrame(data={'predictions': real_seq[:,0],
                             'Scaling_2': real_seq[:,1],
                             'hour_cosin':real_seq_scaled[:,2],
                             'hour_sin':real_seq_scaled[:,3],
                             'minute_cosin':real_seq_scaled[:,4],
                             'minute_sin':real_seq_scaled[:,5],
                             'Attack_Belief': beliefs[:,0],
                             'NonAttack_Belief': beliefs[:,1],
                             'Targets': targets_full})    

data_save_path = project_root/'data'/'Belief_Data.csv'
dataset.to_csv(data_save_path, index=True, index_label='DateTime')   




