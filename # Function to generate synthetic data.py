import numpy as np
import pandas as pd
import pickle
import os

# Function to generate synthetic data
def new_data(data_train, method, length, seed=0):
    np.random.seed(seed)
    data_syntrain = pd.DataFrame(columns=data_train.columns, dtype=np.int32)
    
    for col in data_train.columns:
        x = data_train[col].max()
        y = data_train[col].min()
        z = data_train[col].mean()
        w = data_train[col].std()

        if method == 'Normal':
            augment = np.random.normal(loc=z, scale=w, size=length)
            augment = np.round(augment).astype(np.int32)
            augment = np.maximum(0, augment)
        elif method == 'Poisson':
            augment = np.random.poisson(lam=z, size=length).astype(np.int32)
        elif method == 'Usual':
            augment = np.random.randint(y, x + 1, size=length, dtype=np.int32)
        else:
            continue

        data_syntrain[col] = augment

    return data_syntrain

# Define data structures
data = {}
data_train = {}
data_eval = {}
data_test = {}
long_train_data = {}
long_eval_data={}

Overdose_Incident_Data = r'D:\Opioid Project\Data Set\Input Data\Overdose Incidine Data (For DOPP Only)\facility_files'
files_in_directory = os.listdir(Overdose_Incident_Data)
print(f'Files in directory: {files_in_directory}')
import os

current_directory = os.getcwd()
print(f'Current Working Directory: {current_directory}')
# Process files in smaller batches to avoid memory issues
for i in range(30):
    file_path = os.path.join(Overdose_Incident_Data, f'Facility Data Set {i}.csv')
    original_data = pd.read_csv(file_path)
    data[i] = original_data.iloc[408:26713, 1:]
    data_train[i] = data[i].iloc[:15820]
    data_eval[i] = data[i].iloc[15820:21093]
    data_test[i] = data[i].iloc[21093:]

    
    # Generate synthetic data for each facility in the current batch

    synthetic_data = new_data(data_train[i], 'Normal', 100000, 0)
    synthetic_eval_data=new_data(data_eval[i], 'Normal', 100000, 0)
    long_train_data[i] = pd.concat([data_train[i], synthetic_data], ignore_index=True)
    long_train_data[i] = long_train_data[i].iloc[15820:, :]
    long_eval_data[i] = pd.concat([data_eval[i], synthetic_eval_data], ignore_index=True)
    long_eval_data[i] = long_eval_data[i].iloc[15820:, :]
    # Clear memory after each batch
    # long_train_data.clear()
    # gc.collect()


file_path = os.path.join(r'D:\Opioid Project\Data Set\Provider_Distance_Matrix.csv')
matrix = pd.read_csv(file_path, header=0, index_col=0)
distance_matrix = matrix.to_numpy()

print('Data loading and preprocessing completed successfully.')
print(f'Training data facilities loaded: {len(data_train)}')



# Define save directory
save_directory = r'D:\Opioid Project\Data Set\Processed Data'
os.makedirs(save_directory, exist_ok=True)

# Save long training and evaluation data
for i in range(30):
    train_filename = os.path.join(save_directory, f'long_train_data_{i}.pkl')
    eval_filename = os.path.join(save_directory, f'long_eval_data_{i}.pkl')

    with open(train_filename, 'wb') as f:
        pickle.dump(long_train_data[i], f)
    
    with open(eval_filename, 'wb') as f:
        pickle.dump(long_eval_data[i], f)

    print(f'Saved {train_filename}')
    print(f'Saved {eval_filename}')

print("All processed data saved successfully.")

