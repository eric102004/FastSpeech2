import math
import argparse
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

#from torchmeta.datasets.helpers import omniglot, miniimagenet
#from torchmeta.utils.data import BatchMetaDataLoader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os

from fastspeech2 import FastSpeech2
from loss import FastSpeech2Loss
from meta_dataset import Dataset
from optimizer import ScheduledOptim
from evaluate import evaluate
import hparams as hp
import utils
import audio as Audio

import higher

import hypergrad as hg



class Task:
    """
    Handles the train and valdation loss for a single task
    """
    def __init__(self, reg_param, meta_model, data, batch_size=None):
        self.device = next(meta_model.parameters()).device

        # stateless version of meta_model
        self.fmodel = higher.monkeypatch(meta_model, device=self.device, copy_initial_weights=True)

        self.n_params = len(list(meta_model.parameters()))
        self.sample_tr, self.sample_te = data
        self.reg_param = reg_param
        self.batch_size = 1 if not batch_size else batch_size
        self.val_loss = None
        self.loss_fn = FastSpeech2Loss().to(self.device)

    def bias_reg_f(self, bias, params):
        # l2 biased regularization
        return sum([((b - p) ** 2).sum() for b, p in zip(bias, params)])

    def train_loss_f(self, params, hparams):
        # biased regularized cross-entropy loss where the bias are the meta-parameters in hparams
        total_loss = None

        text = torch.from_numpy(self.sample_tr['text']).long().to(self.device)
        mel_target = torch.from_numpy(self.sample_tr['mel_target']).float().to(self.device)
        D = torch.from_numpy(self.sample_tr['D']).long().to(self.device)
        log_D = torch.from_numpy(self.sample_tr['log_D']).float().to(self.device)
        f0 = torch.from_numpy(self.sample_tr['f0']).float().to(self.device)
        energy = torch.from_numpy(self.sample_tr['energy']).float().to(self.device)
        src_len = torch.from_numpy(self.sample_tr['src_len']).long().to(self.device)
        mel_len = torch.from_numpy(self.sample_tr['mel_len']).long().to(self.device)
        max_src_len = np.max(self.sample_tr['src_len']).astype(np.int32)
        max_mel_len = np.max(self.sample_tr['mel_len']).astype(np.int32)
        #forward
        mel_output, mel_postnet_output, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _ =  self.fmodel(text, src_len, mel_len, D, f0, energy, max_src_len, max_mel_len, params=params)
        #cal loss
        mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss = self.loss_fn(log_duration_output, log_D, f0_output, f0, energy_output, energy, mel_output, mel_postnet_output, mel_target, ~src_mask, ~mel_mask)
        total_loss = mel_loss + mel_postnet_loss + d_loss + f_loss + e_loss
        total_loss = total_loss / self.batch_size
        total_loss += 0.5 * self.reg_param * self.bias_reg_f(hparams, params)
        return total_loss
        #out = self.fmodel(self.train_input, params=params)
        #return F.cross_entropy(out, self.train_target) + 0.5 * self.reg_param * self.bias_reg_f(hparams, params)

    def val_loss_f(self, params, hparams):
        # cross-entropy loss (uses only the task-specific weights in params
        #out = self.fmodel(self.test_input, params=params)
        #val_loss = F.cross_entropy(out, self.test_target)/self.batch_size
        text = torch.from_numpy(self.sample_te['text']).long().to(self.device)
        mel_target = torch.from_numpy(self.sample_te['mel_target']).float().to(self.device)
        D = torch.from_numpy(self.sample_te['D']).long().to(self.device)
        log_D = torch.from_numpy(self.sample_te['log_D']).float().to(self.device)
        f0 = torch.from_numpy(self.sample_te['f0']).float().to(self.device)
        energy = torch.from_numpy(self.sample_te['energy']).float().to(self.device)
        src_len = torch.from_numpy(self.sample_te['src_len']).long().to(self.device)
        mel_len = torch.from_numpy(self.sample_te['mel_len']).long().to(self.device)
        max_src_len = np.max(self.sample_te['src_len']).astype(np.int32)
        max_mel_len = np.max(self.sample_te['mel_len']).astype(np.int32)
        #forward
        mel_output, mel_postnet_output, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _ =  self.fmodel(text, src_len, mel_len, D, f0, energy, max_src_len, max_mel_len, params = params)
        #cal loss
        mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss = self.loss_fn(log_duration_output, log_D, f0_output, f0, energy_output, energy, mel_output, mel_postnet_output, mel_target, ~src_mask, ~mel_mask)
        val_loss = mel_loss + mel_postnet_loss + d_loss + f_loss + e_loss
        self.val_mel_loss = mel_loss.item()
        self.val_mel_postnet_loss = mel_postnet_loss.item()
        self.val_d_loss = d_loss.item()
        self.val_f_loss = f_loss.item()
        self.val_e_loss = f_loss.item()
        self.val_loss = val_loss.item()  # avoid memory leaks

        #pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        #self.val_acc = pred.eq(self.test_target.view_as(pred)).sum().item() / len(self.test_target)

        return val_loss


def main(args):

    log_interval = hp.log_step                #the steps between printing time and loss of inner loop
    eval_interval = hp.eval_step               #the steps between performing eval for meta-model
    inner_log_interval = None
    inner_log_interval_test = None
    #ways = 5
    batch_size = hp.batch_size                   #the batch_size for each subtask in the inner loop
    n_tasks_test = hp.n_tasks_test  # usually 1000 tasks are used for testing     #num of tasks chosen in testing
    '''
    if args.dataset == 'omniglot':
        reg_param = 2  # reg_param = 2
        T, K = 16, 5  # T, K = 16, 5
    elif args.dataset == 'miniimagenet':
        reg_param = 0.5  # reg_param = 0.5
        T, K = 10, 5  # T, K = 10, 5
    else:
        raise NotImplementedError(args.dataset, " not implemented!")
    '''
    reg_param = hp.reg_param
    T = hp.T
    K = hp.K
    
    T_test = T
    inner_lr = hp.inner_lr
	
    '''
    loc = locals()
    del loc['parser']
    del loc['args']
    '''

    #print(args, '\n', loc, '\n')
    print(args, '\n')
    
    #get device
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    # the following are for reproducibility on GPU, see https://pytorch.org/docs/master/notes/randomness.html
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    '''
    if args.dataset == 'omniglot':
        dataset = omniglot("data", ways=ways, shots=1, test_shots=15, meta_train=True, download=True)
        test_dataset = omniglot("data", ways=ways, shots=1, test_shots=15, meta_test=True, download=True)

        meta_model = get_cnn_omniglot(64, ways).to(device)
    elif args.dataset == 'miniimagenet':
        dataset = miniimagenet("data", ways=ways, shots=1, test_shots=15, meta_train=True, download=True)
        test_dataset = miniimagenet("data", ways=ways, shots=1, test_shots=15, meta_test=True, download=True)

        meta_model = get_cnn_miniimagenet(32, ways).to(device)
    else:
        raise NotImplementedError("DATASET NOT IMPLEMENTED! only omniglot and miniimagenet ")

    dataloader = BatchMetaDataLoader(dataset, batch_size=batch_size, **kwargs)
    test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=batch_size, **kwargs)
    '''
    #dataset and dataloader
    print("setting up dataset and dataloader...")
    print("meta_train")
    dataset = Dataset(filelist=hp.filelist_tr, num_subtasks=hp.num_subtasks,num_subtask_training_data=hp.num_subtask_training_data, num_subtask_testing_data = hp.num_subtask_testing_data)
    dataloader = DataLoader(dataset, batch_size = hp.batch_size, shuffle=True, collate_fn=dataset.collate_fn, drop_last= True, num_workers=0)
    test_dataset = Dataset(filelist=hp.filelist_tr, num_subtasks=hp.num_subtasks, num_subtask_training_data=hp.num_subtask_training_data, num_subtask_testing_data = hp.num_subtask_testing_data)
    test_dataloader = DataLoader(test_dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=test_dataset.collate_fn, drop_last= True, num_workers=0)            #since only have tr data now, use training data 
    print("meta-training data file list:", dataset.filelist)
    print("meta-testing data file list:", test_dataset.filelist)
    
    #define model
    print("defining model...")
    meta_model = FastSpeech2().to(device)
    ## check  require_grad in meta_model parameters
    '''
    for name, param in meta_model.named_parameters():
        if param.requires_grad==False:
            print('name:',name)
            print('data:', param.data)
    print(done)
    '''
    print("model has been defined")
    num_param = utils.get_param_num(meta_model)
    print("num of FastSpeech2 Parameters", num_param)

    #optimizer and loss
    print("setting up optimizer and loss")
    outer_opt = torch.optim.Adam(params=meta_model.parameters(), betas = hp.betas, eps=hp.eps, weight_decay = hp.weight_decay)
    scheduled_optim = ScheduledOptim(outer_opt, hp.decoder_hidden, hp.n_warm_up_step, args.restore_step)
    Loss = FastSpeech2Loss().to(device)
    # outer_opt = torch.optim.SGD(lr=0.1, params=meta_model.parameters())
    inner_opt_class = hg.GradientDescent
    inner_opt_kwargs = {'step_size': inner_lr}

    def get_inner_opt(train_loss):
        return inner_opt_class(train_loss, **inner_opt_kwargs)

    #load checkpoint is exists
    checkpoint_path = os.path.join(hp.checkpoint_path)
    try:
        checkpoint = torch.load(os.path.join(
            checkpoint_path, 'checkpoint_{}.pth.tar'.format(args.restore_step)))
        meta_model.load_state_dict(checkpoint['model'])
        outer_opt.load_state_dict(checkpoint['optimizer'])
        print("\n---Model Restored at Step {}---\n".format(args.restore_step))
    except:
        print("\n---Start New Training---\n")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

    # Load vocoder
    if hp.vocoder == 'melgan':
        melgan = utils.get_melgan()
        melgan.to(device)
    elif hp.vocoder == 'waveglow':
        waveglow = utils.get_waveglow()
        waveglow.to(device)

    # Init logger
    log_path = hp.log_path
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        os.makedirs(os.path.join(log_path, 'train'))
        os.makedirs(os.path.join(log_path, 'validation'))
    train_logger = SummaryWriter(os.path.join(log_path, 'train'))
    val_logger = SummaryWriter(os.path.join(log_path, 'validation'))

    # Init synthesis directory
    synth_path = hp.synth_path
    if not os.path.exists(synth_path):
        os.makedirs(synth_path)

      
    current_step = args.restore_step

    #start training
    print("start training")
    for epoch in range(hp.epochs):
      print(f"###########  Epoch_{epoch+1}       ###########")
      for k, (batch_tr, batch_te) in enumerate(dataloader):
        start_time = time.time()
        meta_model.train()

        current_step += 1

        #tr_xs, tr_ys = batch["train"][0].to(device), batch["train"][1].to(device)
        #tst_xs, tst_ys = batch["test"][0].to(device), batch["test"][1].to(device)

        outer_opt.zero_grad()
        scheduled_optim.zero_grad()

        train_loss = 0
        train_mel_loss = 0
        train_mel_postnet_loss = 0
        train_d_loss = 0
        train_f_loss = 0
        train_e_loss = 0
        forward_time, backward_time = 0, 0

        #start training in each subtask
        assert len(batch_tr)==len(batch_te)
        #for t_idx, (tr_x, tr_y, tst_x, tst_y) in enumerate(zip(tr_xs, tr_ys, tst_xs, tst_ys)):
        for t_idx in range(len(batch_tr)):
            start_time_task = time.time()

            # single task set up
            task = Task(reg_param, meta_model, (batch_tr[t_idx], batch_te[t_idx]), batch_size=batch_tr[t_idx]['text'].shape[0])
            inner_opt = get_inner_opt(task.train_loss_f)

            # single task inner loop
            params = [p.detach().clone().requires_grad_(True) for p in meta_model.parameters()]
            last_param = inner_loop(meta_model.parameters(), params, inner_opt, T, log_interval=inner_log_interval)[-1]
            forward_time_task = time.time() - start_time_task

            # single task hypergradient computation
            if args.hg_mode == 'CG':
                # This is the approximation used in the paper CG stands for conjugate gradient
                cg_fp_map = hg.GradientDescent(loss_f=task.train_loss_f, step_size=1.)
                hg.CG(last_param, list(meta_model.parameters()), K=K, fp_map=cg_fp_map, outer_loss=task.val_loss_f)
            elif args.hg_mode == 'fixed_point':
                hg.fixed_point(last_param, list(meta_model.parameters()), K=K, fp_map=inner_opt,
                               outer_loss=task.val_loss_f)    #gradient will add to p.grad for p in model parameters

            backward_time_task = time.time() - start_time_task - forward_time_task

            train_loss += task.val_loss
            train_mel_postnet_loss += task.val_mel_postnet_loss
            train_mel_loss += task.val_mel_loss
            train_d_loss += task.val_d_loss
            train_f_loss += task.val_f_loss
            train_e_loss += task.val_e_loss
            #val_acc += task.val_acc/task.batch_size

            forward_time += forward_time_task
            backward_time += backward_time_task

        #clipping gradient to avoid gradient explosion
        nn.utils.clip_grad_norm_(meta_model.parameters(), hp.grad_clip_thresh)

        #Update weights
        scheduled_optim.step_and_update_lr()
        scheduled_optim.zero_grad()

        #outer_opt.step()
        step_time = time.time() - start_time

        if current_step % hp.log_step ==0:
            str1 = "Epoch [{}/{}], Step {}:".format( \
                epoch+1, hp.epochs, current_step)
            str2 = 'MT k={} ({:.3f}s F: {:.3f}s, B: {:.3f}s)' \
                .format(k, step_time, forward_time, backward_time)
            str3 = 'Val total Loss : {:.2e} Mel Loss: {:.2e} Mel Postnet Loss: {:.2e} D Loss: {:.2e} F Loss: {:.2e} E Loss: {:.2e}' \
                .format(train_loss, train_mel_loss, train_mel_postnet_loss, train_d_loss, train_f_loss, train_e_loss)
            print('\n'+str1)
            print(str2)
            print(str3)
      
            #write std output to log file
            with open(os.path.join(log_path, "log.txt"), "a") as f_log:
                f_log.write(str1 + '\n')
                f_log.write(str2 + '\n')
                f_log.write(str3 + '\n')
                f_log.write('\n')

            train_logger.add_scalar('Loss/total_loss', train_loss, current_step)
            train_logger.add_scalar('Loss/mel_loss', train_mel_loss, current_step)
            train_logger.add_scalar('Loss/mel_postnet_loss', train_mel_postnet_loss, current_step)
            train_logger.add_scalar('Loss/duration_loss', train_d_loss, current_step)
            train_logger.add_scalar('Loss/F0_loss', train_f_loss, current_step)
            train_logger.add_scalar('Loss/energy_loss', train_e_loss, current_step)

        if current_step % hp.save_step ==0:
            torch.save({'model':meta_model.state_dict(), 'optimizer': optimizer.state_dict()}, op.path.join(checkpoint_path, 'checkpoint_{}.pth.tar'.format(surrent_step)))
            print('save model at step {} ...'.format(current_step))

        #if current_step % hp.synth_step == 0:
            # todo

        if current_step % hp.eval_step == 0:         
            print("evaluating....")
            val_losses, val_mel_losses, val_mel_postnet_losses, val_d_losses, val_f_losses, val_e_losses = evaluate(n_tasks_test, test_dataloader, meta_model, T_test, get_inner_opt,
                                          reg_param, log_interval=inner_log_interval_test)

            print("Test loss {:.2e} +- {:.2e}(mean +- std over {} tasks)."
                  .format(val_losses.mean(), val_losses.std(), len(val_losses)))

            val_logger.add_scalar('Val_Loss/total_loss', val_losses.mean(), current_step)
            val_logger.add_scalar('Val_Loss/mel_loss', val_mel_losses.mean(), current_step)
            val_logger.add_scalar('Val_Loss/mel_postnet_loss', val_mel_postnet_losses.mean(), current_step)
            val_logger.add_scalar('Val_Loss/duration_loss', val_d_losses.mean(), cuurent_step)
            val_logger.add_scalar('Val_Loss/F0_loss', val_f_losses.mean(), current_step)
            val_logger.add_scalar('Val_Loss/energy_loss', val_e_losses.mean(), current_step)





def inner_loop(hparams, params, optim, n_steps, log_interval, create_graph=False):
    params_history = [optim.get_opt_params(params)]

    for t in range(n_steps):
        params_history.append(optim(params_history[-1], hparams, create_graph=create_graph))

        if log_interval and (t % log_interval == 0 or t == n_steps-1):
            print('t={}, Loss: {:.6f}'.format(t, optim.curr_loss.item()))

    return params_history


def evaluate(n_tasks, dataloader, meta_model, n_steps, get_inner_opt, reg_param, log_interval=None):
    meta_model.train()
    device = next(meta_model.parameters()).device

    val_losses = []
    val_mel_postnet_losses = []
    val_mel_losses = []
    val_d_losses = []
    val_f_losses = []
    val_e_losses = []
    while(True):
      for k, (batch_tr, batch_te) in enumerate(dataloader):
        assert len(batch_tr)==len(batch_te)
        for t_idx in range(len(batch_tr)):
            task = Task(reg_param, meta_model, (batch_tr[t_idx], batch_te[t_idx]), batch_size=batch_tr[t_idx]['text'].shape[0])
            inner_opt = get_inner_opt(task.train_loss_f)

            params = [p.detach().clone().requires_grad_(True) for p in meta_model.parameters()]
            last_param = inner_loop(meta_model.parameters(), params, inner_opt, n_steps, log_interval=log_interval)[-1]

            task.val_loss_f(last_param, meta_model.parameters())

            val_losses.append(task.val_loss)
            #val_accs.append(task.val_acc)
            val_mel_postnet_losses.append(task.val_mel_postnet_loss)
            val_mel_losses.append(task.val_mel_loss)
            val_d_losses.append(task.val_d_loss)
            val_f_losses.append(task.val_f_loss)
            val_e_losses.append(task.val_e_loss)

            if len(val_losses) >= n_tasks:
                return np.array(val_losses), np.array(val_mel_losses), np.array(val_mel_postnet_loss), np.array(val_d_losses), np.array(val_f_losses), np.array(val_e_losses)

'''
def get_cnn_omniglot(hidden_size, n_classes):
    def conv_layer(ic, oc, ):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.BatchNorm2d(oc, momentum=1., affine=True,
                           track_running_stats=True # When this is true is called the "transductive setting"
                           )
        )

    net =  nn.Sequential(
        conv_layer(1, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        nn.Flatten(),
        nn.Linear(hidden_size, n_classes)
    )

    initialize(net)
    return net


def get_cnn_miniimagenet(hidden_size, n_classes):
    def conv_layer(ic, oc):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.BatchNorm2d(oc, momentum=1., affine=True,
                           track_running_stats=False  # When this is true is called the "transductive setting"
                           )
        )

    net = nn.Sequential(
        conv_layer(3, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        nn.Flatten(),
        nn.Linear(hidden_size*5*5, n_classes,)
    )

    initialize(net)
    return net


def initialize(net):
    # initialize weights properly
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            #m.weight.data.normal_(0, 0.01)
            #m.bias.data = torch.ones(m.bias.data.size())
            m.weight.data.zero_()
            m.bias.data.zero_()

    return net
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data HyperCleaner')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='omniglot', metavar='N', help='omniglot or miniimagenet')
    parser.add_argument('--hg-mode', type=str, default='CG', metavar='N',
                        help='hypergradient approximation: CG or fixed_point')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--restore_step',type=int, default=0)
    args = parser.parse_args()
    main(args)
