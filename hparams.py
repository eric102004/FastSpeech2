import os

# Dataset
dataset = "LJSpeech"
data_path = "../LJSpeech-1.1"
#dataset = "Blizzard2013"
#data_path = "./Blizzard-2013/train/segmented/"


# Text
text_cleaners = ['english_cleaners']


# Audio and mel
### for LJSpeech ###
sampling_rate = 22050
filter_length = 1024        
hop_length = 256                    
win_length = 1024                    
### for Blizzard2013 ###
#sampling_rate = 16000
#filter_length = 800
#hop_length = 200
#win_length = 800

max_wav_value = 32768.0
n_mel_channels = 80
mel_fmin = 0.0
mel_fmax = 8000.0


# FastSpeech 2
encoder_layer = 4
encoder_head = 2
encoder_hidden = 256
decoder_layer = 4
decoder_head = 2
decoder_hidden = 256
fft_conv1d_filter_size = 1024   
fft_conv1d_kernel_size = (9, 1)
encoder_dropout = 0.2
decoder_dropout = 0.2

variance_predictor_filter_size = 256
variance_predictor_kernel_size = 3
variance_predictor_dropout = 0.5

max_seq_len = 1000


# Quantization for F0 and energy
### for LJSpeech ###
f0_min = 71.0
f0_max = 795.8
energy_min = 0.018
energy_max = 315.0

### for Blizzard2013 ###
#f0_min = 71.0
#f0_max = 786.7
#energy_min = 21.23
#energy_max = 101.02

n_bins = 256


# Checkpoints and synthesis path
preprocessed_path = os.path.join("./preprocessed/", dataset)
checkpoint_path = os.path.join("./ckpt/", dataset)
synth_path = os.path.join("./synth/", dataset)
eval_path = os.path.join("./eval/", dataset)
log_path = os.path.join("./log/", dataset)
test_path = "./results"


# Optimizer
batch_size = 3                #original:16
epochs = 1000
n_warm_up_step = 4000
grad_clip_thresh = 1.0
acc_steps = 1

betas = (0.9, 0.98)
eps = 1e-9
weight_decay = 0.


# Vocoder
vocoder = 'melgan' # 'waveglow' or 'melgan'


# Log-scaled duration
log_offset = 1.


# Save, log and synthesis
save_step = 10000
synth_step = 1000
eval_step = 1000
eval_size = 256
log_step = 10              #change from 1000 to 10  
clear_Time = 20


#--------------------------------------------------
#some parameters for iMAML
reg_param = 2               #the coef of the distance loss of model parameters in inner loop training
T = 16           # the steps taken in inner loop
K = 5            # the steps taken in computing hypergradient
n_tasks_test = 1000  # the num of tasks taken in testing phase
inner_lr = .1     # learning rate of inner loop
num_subtasks = 5           # the num of subtasks in meta learning (num o speakers),used to initializedataset
filelist_tr = ['train_1.txt', 'train_2.txt', 'train_3.txt', 'train_4.txt', 'train_5.txt']
filelist_val = ['val_1.txt', 'val_2.txt', 'val_2.txt', 'val_4.txt', 'val_5.txt']
num_subtask_training_data = 5      # the num of training data in each subtasks, used to initialize the dataset
num_subtask_testing_data = 5        # the num of testing data in each subasks
