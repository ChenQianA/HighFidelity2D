import yaml
import torch
import logging
from datetime import datetime
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


device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

with open('config_mfd.yaml', 'r') as c:
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

directory = os.path.join(logdir, '_'.join([config['early_stopping_task'], datetime.now().strftime("%Y%m%d_%H%M%S")]))
logging.info(f"Directory: {directory}")
if not os.path.exists(directory):
    os.makedirs(directory)

best_dir = os.path.join(directory, 'best')
if not os.path.exists(best_dir):
    os.makedirs(best_dir)
best_loss_file = os.path.join(best_dir, 'best_metrics.pickle')
best_ckpt_file = os.path.join(best_dir, 'model.pt')

log_dir = os.path.join(directory, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
step_ckpt_folder = log_dir

fh = logging.FileHandler(directory + "/log.txt", mode='w')
fh.setFormatter(formatter)
logger.addHandler(fh)
shutil.copyfile('config_mfd.yaml', directory+'/config_mfd.yaml')

torch.set_default_dtype(torch.float32)

model_config = config['model_config']

data_dir = config['data_dir']
datasets = config['datasets']
batch_size = config['batch_size']
early_stopping_task = config['early_stopping_task']
assert early_stopping_task in [i[0] for i in datasets], "early stopping task should be included in datasets"
early_stopping_task_id = {i[0]: idx for idx, i in enumerate(datasets)}[early_stopping_task]

dataset_list = [MyOwnDataset(os.path.join(data_dir, i[0]), datatype=i[1]) for i in datasets]
mean_std_list = [torch.load(os.path.join(data_dir, '/'.join([i[0], 'mean_std.pt']))) for i in datasets]
mean_std_list = [{i: j.numpy() for i, j in mean_std.items()} for mean_std in mean_std_list]
datatype_list = [i[1] for i in datasets]
dataout_dim = [i[2] for i in datasets]
loss_weights = torch.Tensor([i[3] for i in datasets]).to(device)
loss_weights = loss_weights/loss_weights.sum()

model_config.update(dict(task=[datatype_list, dataout_dim]))

model = MF(**model_config).to(device)

train_data_list, val_data_list, test_data_list = train_val_test_split_list(dataset_list)

train_loader_list = train_data_loaders_list(train_data_list=train_data_list, datatype_list=datatype_list, batch_size=batch_size)
train_zip_loader = zip_loaders(train_loader_list)

val_loader_list = val_test_data_loaders_list(val_test_data_list=val_data_list, batch_size=batch_size*len(val_data_list))
test_loader_list = val_test_data_loaders_list(val_test_data_list=test_data_list, batch_size=batch_size*len(test_data_list))

learning_rate = config['learning_rate']
weight_decay = config['weight_decay']
decay = config['decay']
num_epochs = config['num_epochs']
steps_per_epoch = config['steps_per_epoch']
embedding_decay = config['embedding_decay']

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, epochs=num_epochs, steps_per_epoch=steps_per_epoch, pct_start=0.3)
ema = EMA(model, decay)
ema.register()
trainer = trainer_mfd(model=model, optimizer=optimizer, scheduler=scheduler, ema=ema, datatype_list=datatype_list, loss_weights=loss_weights, embedding_decay=embedding_decay)

train_metrics = []
val_metrics = []
best_metrics = {'iteration': 0, 'loss': np.inf, 'loss_list': [np.inf]*len(datasets)}
iteration = 0
for epoch in range(num_epochs):
    trainer.train()
    for step_epoch in range(steps_per_epoch):
        batches = train_zip_loader.next()
        train_loss_list = trainer.train_on_batch(batches=batches, train_metrics=train_metrics)
        iteration += 1

    if epoch %5 ==0:
        trainer.eval()
        val_loss_list, result_dict = trainer.test_on_batch(loader_list=val_loader_list, val_metrics=val_metrics)
        logging.info(f"{epoch + 1}/{num_epochs} (epoch {epoch + 1}) (iteration {iteration}): "
                     f"train_loss_list=" + str([round(i, 4) for i in train_loss_list]) + ", "
                     f"validation_loss_list=" + str([round(i, 4) for i in val_loss_list]))
        trainer.save_state_dict_step(step_ckpt_folder=step_ckpt_folder)
        if early_stopping_task is None and val_metrics[-1]['loss'] < best_metrics['loss']:
            best_metrics.update(val_metrics[-1])
            if epoch > 10:
                trainer.save_state_dict_best(best_ckpt_file=best_ckpt_file)
        elif early_stopping_task is not None and val_metrics[-1]['loss_list'][early_stopping_task_id] < best_metrics['loss_list'][early_stopping_task_id]:
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
test_loss_list, result_dict = trainer.test_on_batch(loader_list=test_loader_list, val_metrics=test_metrics_placeholder)
test_metrics = {'loss': test_loss_list, 'result': result_dict, 'mean_std_list': mean_std_list}
logging.info(f"congradulation: you have finished training! "
             f"test_loss_list=" + str([round(i, 6) for i in test_loss_list]))

with open(best_dir+"/test_metrics.pickle", 'wb') as f:
    pickle.dump(test_metrics, f)

torch.save(model, best_dir+'/model')


























