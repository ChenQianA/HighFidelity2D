import yaml
import torch
import logging
from datetime import datetime
import os
import shutil
from SDL.modulus.pyg_att import Matformer
import ast
from SDL.utils.my_dataset import MyOwnDataset
from SDL.utils.data_loader import get_sample_weights_from_class
from SDL.utils.ema import EMA
from SDL.utils.trainer import trainer_scratch
import numpy as np
import pickle
import math
from torch_geometric.loader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.optim import AdamW


device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

with open('config_scratch.yaml', 'r') as c:
    config = yaml.safe_load(c)

for key, val in config.items():
    if type(val) is str:
        try:
            config[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass

torch.manual_seed(config['seed'])
torch.cuda.manual_seed_all(config['seed'])

logger = logging.getLogger()
logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter(fmt='%(asctime)s (%(levelname)s): %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel('INFO')

logdir = config['logdir']

directory = os.path.join(logdir, '_'.join([config['dataset'][0],datetime.now().strftime("%Y%m%d_%H%M%S")]))
logging.info(f"Directory: {directory}")
if not os.path.exists(directory):
    os.makedirs(directory)

best_dir = os.path.join(directory, 'best')
if not os.path.exists(best_dir):
    os.makedirs(best_dir)
best_loss_file = os.path.join(best_dir, 'metrics_best.pickle')
best_ckpt_file = os.path.join(best_dir, 'model.pt')

log_dir = os.path.join(directory, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
step_ckpt_folder = log_dir

fh = logging.FileHandler(directory + "/log.txt", mode='w')
fh.setFormatter(formatter)
logger.addHandler(fh)
shutil.copyfile('config_scratch.yaml', directory+'/config_scratch.yaml')

torch.set_default_dtype(torch.float32)

model_config = config['model_config']

data_dir = config['data_dir']
dataset_config = config['dataset']
batch_size = config['batch_size']

dataset = MyOwnDataset(os.path.join(data_dir, dataset_config[0]), datatype=dataset_config[1])
mean_std = torch.load(os.path.join(data_dir, '/'.join([dataset_config[0], 'mean_std.pt'])))
mean_std = {i: j.numpy() for i, j in mean_std.items()}
datatype = dataset_config[1]
dataout_dim = dataset_config[2]

model_config.update(dict(output_features=dataout_dim))

model = Matformer(**model_config).to(device)

dataset = dataset.shuffle()
train_data, val_data, test_data = dataset[:math.floor(len(dataset)*0.8)], dataset[math.floor(len(dataset)*0.8):math.floor(len(dataset)*0.9)], dataset[math.floor(len(dataset)*0.9):]
if datatype == 'r':
    train_loader, val_loader, test_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True),\
        DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True), DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
elif datatype == 'c':
    train_sampler = WeightedRandomSampler(get_sample_weights_from_class(train_data), num_samples=len(train_data), replacement=True)
    train_loader, val_loader, test_loader = DataLoader(dataset=train_data, batch_size=batch_size, sampler=train_sampler),\
        DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True), DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

learning_rate = config['learning_rate']
weight_decay = config['weight_decay']
decay = config['decay']
num_epochs = config['num_epochs']
steps_per_epoch = math.ceil(len(train_data)/batch_size)

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, epochs=num_epochs, steps_per_epoch=steps_per_epoch, pct_start=0.3)
ema = EMA(model, decay)
ema.register()
trainer = trainer_scratch(model=model, optimizer=optimizer, scheduler=scheduler, ema=ema, datatype=datatype)

train_metrics = []
val_metrics = []
best_metrics = {'iteration': 0, 'loss': np.inf}
iteration = 0
for epoch in range(num_epochs):
    trainer.train()
    for batch in train_loader:
        train_loss = trainer.train_on_batch(batch=batch, train_metrics=train_metrics)
        iteration += 1

    if epoch %5 ==0:
        trainer.eval()
        val_loss, result_dict = trainer.test_on_batch(loader=val_loader, val_metrics=val_metrics)
        logging.info(f"{epoch + 1}/{num_epochs} (epoch {epoch + 1}) (iteration {iteration}): "
                     f"train_loss {train_loss:.6f}, validation_loss {val_loss:.6f}")
        trainer.save_state_dict_step(step_ckpt_folder=step_ckpt_folder)
        if val_metrics[-1]['loss'] < best_metrics['loss']:
            best_metrics.update(val_metrics[-1])
            if epoch > 10:
                trainer.save_state_dict_best(best_ckpt_file=best_ckpt_file)

with open(best_dir+"/train_metrics.pickle", 'wb') as f:
    pickle.dump(train_metrics, f)

with open(best_dir+"/validation_metrics.pickle", 'wb') as f:
    pickle.dump(val_metrics, f)

with open(best_loss_file, 'wb') as f:
    pickle.dump(best_metrics, f)

model.load_state_dict(torch.load(best_ckpt_file))
model.eval()
test_metrics_placeholder = []
test_loss, result_dict = trainer.test_on_batch(loader=test_loader, val_metrics=test_metrics_placeholder)
test_metrics = {'loss': test_loss, 'result': result_dict, 'mean_std': mean_std}

with open(best_dir+"/test_metrics.pickle", 'wb') as f:
    pickle.dump(test_metrics, f)

logging.info(f"congradulation: you have finished training! "
             f"test_loss={test_loss:.6f}")

torch.save(model, best_dir+'/model')
































