import os
import sys
import logging
import time
import shutil
import argparse
import pdb
import wandb

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import dataloading as dl
import model as mdl

#logger_py = logging.getLogger(__name__)

# Fix seeds
np.random.seed(42)
torch.manual_seed(42)

# Arguments
parser = argparse.ArgumentParser(
    description='Training of UNISURF model'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of '
                         'seconds with exit code 2.')
parser.add_argument("--project_name",type=str, 
                help="model path named after project name")
parser.add_argument("--resume", "-r", action="store_true", 
                help="continue training")
args = parser.parse_args()


cfg = dl.load_config(
    os.path.join(os.getcwd(),args.config), 
    'configs/default.yaml'
)
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")
image_resolution = cfg['dataloading']['img_size']
cfg['model']['encoder']['scale_size'] = image_resolution

# params
out_dir = cfg['training']['out_dir']
checkpoint_path = cfg['training']['checkpoints_path']
#backup_every = cfg['training']['backup_every']
epochs = cfg['training']['epochs']
exit_after = args.exit_after
batch_size_train = cfg['dataloading']['batchsize']['train']
batch_size_val = cfg['dataloading']['batchsize']['val']
n_workers = cfg['dataloading']['n_workers']
lr = cfg['training']['learning_rate']
if not args.resume:
  wandb.init(project='imagesurf')
  os.environ["WANDB_RUN_ID"] = wandb.run.name
  project_name = wandb.run.name
else:
  project_name = args.project_name
  os.environ["WANDB_RESUME"] = "allow"
  os.environ["WANDB_RUN_ID"] = project_name
  wandb.init(project='imagesurf')

# init dataloader
train_loader = dl.get_dataloader(cfg, mode='train')
val_loader = dl.get_dataloader(cfg, mode='val')
#iter_val = iter(val_loader)
#data_val = next(iter_val)
test_loader = dl.get_dataloader(cfg, mode='test')
iter_test = iter(test_loader)
# init network
model_cfg = cfg['model']
model = mdl.NeuralNetwork(model_cfg)
print(model)
# init optimizer
weight_decay = cfg['training']['weight_decay']
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# init checkpoints and load
checkpoint_io = mdl.CheckpointIO(checkpoint_path, project_name=project_name)
#checkpoint_io = mdl.CheckpointIO(checkpoint_path)
try:
    load_dict = checkpoint_io.load('model_240.pth')
except FileExistsError:
    load_dict = dict()
epoch_it = load_dict.get('epoch_it', -1)
it = load_dict.get('it', -1)
model = load_dict.get('model', model)
optimizer = load_dict.get('optimizer', optimizer)
checkpoint_io = mdl.CheckpointIO(
  checkpoint_path, 
  model=model, 
  optimizer=optimizer, 
  project_name = project_name
)
# init renderer
rendering_cfg = cfg['rendering']
renderer = mdl.Renderer(model, rendering_cfg, device=device)
# init training
training_cfg = cfg['training']
trainer = mdl.Trainer(renderer, optimizer, training_cfg, device=device)

#scheduler = optim.lr_scheduler.MultiStepLR(
 #   optimizer, cfg['training']['scheduler_milestones'],
  #  gamma=cfg['training']['scheduler_gamma'], last_epoch=epoch_it)

#logger = SummaryWriter(os.path.join(out_dir, 'logs'))
# init training output
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
visualize_every = cfg['training']['visualize_every']
render_path = os.path.join(out_dir, 'rendering')
if visualize_every > 0:
    visualize_skip = cfg['training']['visualize_skip']
    visualize_path = os.path.join(out_dir, 'visualize')
    if not os.path.exists(visualize_path):
        os.makedirs(visualize_path)

# Print model
#nparameters = sum(p.numel() for p in model.parameters())
#logger_py.info(model)
#logger_py.info('Total number of parameters: %d' % nparameters)
#t0b = time.time()
#print('before while')
#print(t0b)
while epoch_it<=epochs:
    epoch_it += 1
    train_net_loss = []
    val_net_loss = []
    mse_loss = []
    val_mse_loss = []
    for batch in train_loader:
        it += 1
        #print(f"iteration: {it}")
        loss_dict = trainer.train_step(batch, it)
        train_net_loss.append(loss_dict['loss'])
        mse_loss.append(loss_dict['mse_loss'])
        #metric_val_best = loss
    #pdb.set_trace()
    train_loss = torch.sum(torch.tensor(train_net_loss*batch_size_train))/(len(train_net_loss)*(batch_size_train))
    train_psnr = -10*np.log(np.float32(train_loss))/np.log(10)
    train_mse = torch.sum(torch.tensor(mse_loss*batch_size_train))/(len(mse_loss)*(batch_size_train))  
    train_mse_psnr = -10*np.log(np.float32(train_mse))/np.log(10)
    #    # Print output
    for batch in val_loader:
        loss_dict = trainer.eval_step(batch, it)
        val_net_loss.append(loss_dict['loss'])
        val_mse_loss.append(loss_dict['mse_loss'])
    #    #metric_val_best = loss
    val_loss = torch.sum(torch.tensor(val_net_loss*batch_size_val))/(len(val_net_loss)*(batch_size_val))
    val_psnr = -10*np.log(np.float32(val_loss))/np.log(10)
    val_mse = torch.sum(torch.tensor(val_mse_loss*batch_size_val))/(len(val_mse_loss)*(batch_size_val))
    val_mse_psnr = -10*np.log(np.float32(val_mse))/np.log(10)
    print(f"Epoch {epoch_it}\n-------------------------------")
    print(f"""Train Loss:{train_loss}""")
    print(f"""MSE Train Loss:{train_mse}""")
    print(f"""Val Loss:{val_loss}""")
    print(f"""MSE Val Loss:{val_mse}""")
    # in order to avoid StopIteraion as soon as length of datalaoder is reached
    # use try-except 
    log = {'trn_optim_loss': train_mse}
    log['train_optim_psnr'] = train_mse_psnr
    log['trn_unisurf_loss'] = train_loss
    log['trn_unisurf_psnr'] = train_psnr
    log['val_unisurf_loss'] = val_loss
    log['val_unisurf_psnr'] = val_psnr
    log['val_optim_loss'] = val_mse
    log['val_optim_psnr'] = val_mse_psnr
    
    if np.ceil(epoch_it/10)!=np.ceil((epoch_it+1)/10):
        try:
            data_test = next(iter_test)
        except StopIteration:
            iter_test = iter(test_loader)
            data_test = next(iter_test)
        ## In case you want to test same instance per epoch uncomment 168-169
        # and comment 160-165
        #iter_test = iter(test_loader)
        #data_test = next(iter_test)
        test_rgb, test_depth, target= trainer.render_visdata(
               data_test, 
               cfg['training']['vis_resolution'], 
               it, None)# out_render_path)
        
        log['target'] = wandb.Image(target)
        log['unisurf'] = wandb.Image(test_rgb)
        log['depth'] = wandb.Image(test_depth)
    wandb.log(log, step=epoch_it)

    # Save checkpoint
    
    if epoch_it>0 and np.ceil(epoch_it/checkpoint_every)!=np.ceil((epoch_it+1)/checkpoint_every):
       print('Saving checkpoint')
       filename = 'model_'+str(epoch_it)+'.pth'
       checkpoint_io.save(filename, epoch_it=epoch_it, it=it,
                       model=model)#loss_val_best=metric_val_best)

