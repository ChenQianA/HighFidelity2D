import yaml
import torch
import os
import shutil
from SDL.modulus.multi_fidelity import MF
import ast
from SDL.utils.my_dataset import MyOwnDataset
from SDL.utils.data_loader import train_val_test_split_list, train_data_loaders_list, val_test_data_loaders_list, zip_loaders
from SDL.utils.ema import EMA
from SDL.utils.trainer import trainer_mfd
import numpy as np
import pickle
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import pandas as pd
import ast


device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

wdir='./logs/multi_fidelity/c2db_hse_20230601_154806'

with open(wdir+'/config_mfd.yaml', 'r') as c:
    config = yaml.safe_load(c)

for key, val in config.items():
    if type(val) is str:
        try:
            config[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass

best_state_dict = torch.load(wdir+'/best/model.pt', map_location=device)

model_config = config['model_config']
data_dir = config['data_dir']
datasets = config['datasets']
batch_size = config['batch_size']
early_stopping_task = config['early_stopping_task']
assert early_stopping_task in [i[0] for i in datasets], "early stopping task should be included in datasets"
early_stopping_task_id = {i[0]: idx for idx, i in enumerate(datasets)}[early_stopping_task]
datatype_list = [i[1] for i in datasets]
dataout_dim = [i[2] for i in datasets]
model_config.update(dict(task=[datatype_list, dataout_dim]))

model = MF(**model_config).to(device)
model.load_state_dict(best_state_dict)
model.eval()

mean_std_c2db = torch.load(os.path.join(data_dir, '/'.join([datasets[early_stopping_task_id][0], 'mean_std.pt'])), map_location=device)

matpedia_dir = os.path.join(data_dir, datasets[-2][0])
dataset_matpedia = MyOwnDataset(matpedia_dir, datatype=datasets[-2][1])
loader_matpedia = DataLoader(dataset=dataset_matpedia, batch_size=32, shuffle=False)
mean_std_matpedia = torch.load(os.path.join(matpedia_dir, 'mean_std.pt'), map_location=device)
jarvis_dir = os.path.join(data_dir, datasets[-1][0])
dataset_jarvis = MyOwnDataset(jarvis_dir, datatype=datasets[-1][1])
loader_jarvis = DataLoader(dataset=dataset_jarvis, batch_size=32, shuffle=False)
mean_std_jarvis = torch.load(os.path.join(jarvis_dir, 'mean_std.pt'), map_location=device)


id_list = []
y_list = []
preds_list = []
for batch in tqdm(loader_matpedia):
    id_list.append(batch.id)
    preds = model(batch, task_id=early_stopping_task_id)
    preds = preds * mean_std_c2db['std'] + mean_std_c2db['mean']
    preds_list.append(preds.detach().numpy())
    y = batch.y * mean_std_matpedia['std'] + mean_std_matpedia['mean']
    y_list.append(y.detach().numpy())
matpedia_id = np.concatenate(id_list, axis=0)
matpedia_y = np.concatenate(y_list, axis=0)[:,[0,2,1]]
matpedia_preds = np.concatenate(preds_list, axis=0)
matpedia_df = pd.DataFrame({'2dmatpedia id': matpedia_id, '2dmatpedia band structure (BG, VBM, CBM)': matpedia_y.tolist(),
                            'preds band structure (BG, Direc BG, VBM, CBM)': matpedia_preds.tolist()})
matpedia_df.to_csv('./result/hse06_2dmatpedia_from_c2db.csv', index=False)


id_list = []
y_list = []
preds_list = []
for batch in tqdm(loader_jarvis):
    id_list.append(batch.id)
    preds = model(batch, task_id=early_stopping_task_id)
    preds = preds * mean_std_c2db['std'] + mean_std_c2db['mean']
    preds_list.append(preds.detach().numpy())
    y = batch.y * mean_std_jarvis['std'] + mean_std_jarvis['mean']
    y_list.append(y.detach().numpy())
jarvis_id = np.concatenate(id_list, axis=0)
jarvis_y = np.concatenate(y_list, axis=0)
jarvis_preds = np.concatenate(preds_list, axis=0)
jarvis_df = pd.DataFrame({'jarvis id': jarvis_id, 'jarvis band structure (BG)': jarvis_y.tolist(),
                          'preds band structure (BG, Direc BG, VBM, CBM)': jarvis_preds.tolist()})
jarvis_df.to_csv('./result/hse06_jarvis_from_c2db.csv', index=False)




matpedia_df = pd.read_csv('./result/hse06_2dmatpedia_from_c2db.csv')
data_2dmatpedia = np.array([eval(i.replace('nan', 'np.nan')) for i in matpedia_df['2dmatpedia band structure (BG, VBM, CBM)'].values])
pred_2dmatpedia = np.array([eval(i) for i in matpedia_df['preds band structure (BG, Direc BG, VBM, CBM)'].values])
matpedia_df = pd.DataFrame({'2dmatpedia_id': matpedia_df['2dmatpedia id'].values, 'BG_optB88': data_2dmatpedia[:, 0],
                            'VBM_optB88': data_2dmatpedia[:, 1], 'CBM_optB88': data_2dmatpedia[:, 2],
                            'BG_pred': pred_2dmatpedia[:, 0], 'Direct_BG_pred': pred_2dmatpedia[:, 1],
                            'VBM_pred': pred_2dmatpedia[:, 2], 'CBM_pred': pred_2dmatpedia[:, 3]})
matpedia_df.to_csv('C:/Users/IanTs/Desktop/桌面/异质结/Manucript/图/Figure 6/data/hse06_2dmatpedia_from_c2db.csv', index=False)


jarvis_df = pd.read_csv('./result/hse06_jarvis_from_c2db.csv')
data_jarvis = np.array([eval(i.replace('nan', 'np.nan')) for i in jarvis_df['jarvis band structure (BG)'].values])
pred_jarvis = np.array([eval(i) for i in jarvis_df['preds band structure (BG, Direc BG, VBM, CBM)'].values])
jarvis_df = pd.DataFrame({'jarvis_id': jarvis_df['jarvis id'].values, 'BG_optB88': data_jarvis[:, 0],
                          'BG_pred': pred_jarvis[:, 0], 'Direct_BG_pred': pred_jarvis[:, 1],
                          'VBM_pred': pred_jarvis[:, 2], 'CBM_pred': pred_jarvis[:, 3]})
jarvis_df.to_csv('C:/Users/IanTs/Desktop/桌面/异质结/Manucript/图/Figure 6/data/hse06_jarvis_from_c2db.csv', index=False)


best_dir = os.path.join(wdir, 'best')
with open(best_dir+"/test_metrics.pickle", 'rb') as f:
    test_metrics = pickle.load(f)
hse06_mean, hse06_std = test_metrics['mean_std_list'][early_stopping_task_id].values()
predict = test_metrics['result'][early_stopping_task_id]['predict'] * hse06_std + hse06_mean
target = test_metrics['result'][early_stopping_task_id]['target'] * hse06_std + hse06_mean
c2db_df = pd.DataFrame({'BG_pred': predict[:, 0], 'Direct_BG_pred': predict[:, 1], 'VBM_pred': predict[:, 2],
                        'CBM_pred': predict[:, 3], 'BG_target': target[:, 0], 'Direct_BG_target': target[:, 1],
                        'VBM_target': target[:, 2], 'CBM_target': target[:, 3]})
c2db_df.to_csv('C:/Users/IanTs/Desktop/桌面/异质结/Manucript/图/Figure 6/data/hse06_c2db_from_c2db.csv', index=False)
