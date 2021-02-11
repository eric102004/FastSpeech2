import os

# Dataset
#dataset = "LJSpeech"
#data_path = "../LJSpeech-1.1"
#dataset = "Blizzard2013"
#data_path = "./Blizzard-2013/train/segmented/"
dataset = "LibriTTS"
data_path = "../LibriTTS"

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
#f0_min = 71.0
#f0_max = 795.8
#energy_min = 0.018
#energy_max = 315.0

### for Blizzard2013 ###
#f0_min = 71.0
#f0_max = 786.7
#energy_min = 21.23
#energy_max = 101.02

### for LibriTTS ###
f0_min = 71.00003919558432
f0_max =  798.4944092042294
energy_min = 0.0
energy_max = 525.9888305664062

n_bins = 256


# Checkpoints and synthesis path
preprocessed_path = os.path.join("./preprocessed/", dataset)
checkpoint_path = os.path.join("./ckpt/", dataset)
synth_path = os.path.join("./synth/", dataset)
eval_path = os.path.join("./eval/", dataset)
log_path = os.path.join("./log/", dataset)
test_path = "./results"


# Optimizer
batch_size = 6                #original:16
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
save_step = 200          #change from 10000 to 10
synth_step = 1000
eval_step = 100           #change from 1000 to 10
eval_size = 256
log_step = 10              #change from 1000 to 10  
clear_Time = 20


#--------------------------------------------------
exp_name = 'test_ft'
exp_set = {'meta_ft','baseline_ft','meta_emb','baseline_emb'}
exp_mode = 'meta'
#some parameters for iMAML
reg_param = 2               #the coef of the distance loss of model parameters in inner loop training
T = 15           # the steps taken in inner loop       ##original: 16
K = 5            # the steps taken in computing hypergradient  #original:5
n_tasks_test = 20  # the num of tasks taken in testing phase          ##change from 1000 to 20
inner_lr = 0.1     # learning rate of inner loop          ##change from 0.1 to 0.001
#filelist_tr = ['103.txt', '1069.txt', '1088.txt', '1098.txt']
filelist_tr = ['103.txt', '1034.txt', '1040.txt', '1069.txt', '1081.txt', '1088.txt', '1098.txt', '1116.txt', '118.txt', '1183.txt', '1235.txt', '1246.txt', '125.txt', '1263.txt', '1334.txt', '1355.txt', '1363.txt', '1447.txt', '1455.txt', '150.txt', '1502.txt', '1553.txt', '1578.txt', '1594.txt', '1624.txt', '163.txt', '1737.txt', '1743.txt', '1841.txt', '1867.txt', '1898.txt', '19.txt', '1926.txt', '196.txt', '1963.txt', '1970.txt', '198.txt', '1992.txt', '200.txt', '2002.txt', '2007.txt', '201.txt', '2092.txt', '211.txt', '2136.txt', '2159.txt', '2182.txt', '2196.txt', '226.txt', '2289.txt', '229.txt', '233.txt', '2384.txt', '2391.txt', '2416.txt', '2436.txt', '248.txt', '250.txt', '2514.txt', '2518.txt', '254.txt', '26.txt', '2691.txt', '27.txt', '2764.txt', '2817.txt', '2836.txt', '2843.txt', '289.txt', '2893.txt', '2910.txt', '2911.txt', '2952.txt', '298.txt', '2989.txt', '302.txt', '307.txt', '311.txt', '3112.txt', '3168.txt', '32.txt', '3214.txt', '322.txt', '3235.txt', '3240.txt', '3242.txt', '3259.txt', '332.txt', '3374.txt', '3436.txt', '3440.txt', '3486.txt', '3526.txt', '3607.txt', '3664.txt', '3699.txt', '3723.txt', '374.txt', '3807.txt', '3830.txt', '3857.txt', '3879.txt', '39.txt', '3947.txt', '3982.txt', '3983.txt', '40.txt', '4014.txt', '4018.txt', '403.txt', '405.txt', '4051.txt', '4088.txt', '412.txt', '4137.txt', '4160.txt', '4195.txt', '4214.txt', '426.txt', '4267.txt', '4297.txt', '4340.txt', '4362.txt', '4397.txt', '4406.txt', '4441.txt', '446.txt', '4481.txt', '458.txt', '460.txt', '4640.txt', '4680.txt', '4788.txt', '481.txt', '4813.txt', '4830.txt', '4853.txt', '4859.txt', '4898.txt', '5022.txt', '5049.txt', '5104.txt', '5163.txt', '5192.txt', '5322.txt', '5339.txt', '5390.txt', '5393.txt', '5456.txt', '5463.txt', '5514.txt', '5561.txt', '5652.txt', '5678.txt', '5703.txt', '5750.txt', '5778.txt', '5789.txt', '5808.txt', '5867.txt', '587.txt', '60.txt', '6000.txt', '6019.txt', '6064.txt', '6078.txt', '6081.txt', '6147.txt', '6181.txt', '6209.txt', '625.txt', '6272.txt', '6367.txt', '6385.txt', '6415.txt', '6437.txt', '6454.txt', '6476.txt', '6529.txt', '6531.txt', '6563.txt', '669.txt', '6818.txt', '6836.txt', '6848.txt', '6880.txt', '6925.txt', '696.txt', '7059.txt', '7067.txt', '7078.txt', '7113.txt', '7148.txt', '7178.txt', '7190.txt', '7226.txt', '7264.txt', '7278.txt', '730.txt', '7302.txt', '7312.txt', '7367.txt', '7402.txt', '7447.txt', '7505.txt', '7511.txt', '7517.txt', '7635.txt', '7780.txt', '7794.txt', '78.txt', '7800.txt', '7859.txt', '8014.txt', '8051.txt', '8063.txt', '8088.txt', '8095.txt', '8098.txt', '8108.txt', '8123.txt', '8226.txt', '8238.txt', '83.txt', '831.txt', '8312.txt', '8324.txt', '839.txt', '8419.txt', '8425.txt', '8465.txt', '8468.txt', '8580.txt', '8609.txt', '8629.txt', '8630.txt', '87.txt', '8747.txt', '8770.txt', '8797.txt', '8838.txt', '887.txt', '89.txt', '8975.txt', '909.txt', '911.txt']
exception_filelist_tr = [ '5688.txt']
#filelist_val = ['1116.txt']
filelist_val = ['1272.txt', '1462.txt', '1673.txt', '174.txt', '1919.txt', '1988.txt', '1993.txt', '2035.txt', '2078.txt', '2086.txt', '2277.txt', '2412.txt', '2428.txt', '251.txt', '2803.txt', '2902.txt', '3000.txt', '3081.txt', '3170.txt', '3536.txt', '3576.txt', '3752.txt', '3853.txt', '422.txt', '5338.txt', '5536.txt', '5694.txt', '5895.txt', '6241.txt', '6295.txt', '6313.txt', '6319.txt', '6345.txt', '652.txt', '777.txt', '7850.txt', '7976.txt', '8297.txt', '84.txt', '8842.txt']
num_subtasks_tr = 4           # the num of subtasks in meta learning (num o speakers),used to initializedataset
num_subtasks_val = 4

meta_testing_ratio = 0.2

#the portion of model to be fine-tuned in the inner loop
#fine_tune_model_set = {'var','dec','mel','pos','emb'}
fine_tune_model_set = {'var','dec','mel','emb'}

#parameters for synthesis
synthesize_speaker_list = ['84','174']
#synthesize_speaker_list = ['1116']
syn_fine_tune_step = 3000
syn_fine_tune_batch_size = 3

#baseline model for synthesize
model_mode = 'meta'
use_spk_embed = True
use_pretrained_emb = False
spk_embed_dim = 256
spk_embed_weight_std = 0.01

# evaluation
num_eval_data = 20    # num of data for evaluation in evaluation.py

# parameters for meta embeddinng
n_meta_emb = 5

if __name__ =='__main__':
    print('filelist_tr:',sorted(filelist_tr))
    print('filelist_val:',sorted(filelist_val))
