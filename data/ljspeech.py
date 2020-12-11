import numpy as np
import os
import tgt
from scipy.io.wavfile import read
import pyworld as pw
import torch
import audio as Audio
from utils import get_alignment
from text import _clean_text
import hparams as hp

def prepare_align(in_dir):
#<<<<<<< Updated upstream
    with open(os.path.join(in_dir, 'metadata_that.csv'), encoding='utf-8') as f:  #change
#=======
    #with open(os.path.join(in_dir, 'metadata_vowel.csv'), encoding='utf-8') as f:  #修改
#>>>>>>> Stashed changes
        for line in f:
            parts = line.strip().split('|')
            basename = parts[0]
            text = parts[2]
            text = _clean_text(text, hp.text_cleaners)
            
            with open(os.path.join(in_dir, 'wavs', '{}.txt'.format(basename)), 'w') as f1:
                f1.write(text)

def build_from_path(in_dir, out_dir):
    index = 1
    train = list()
    val = list()
    f0_max = energy_max = 0
    f0_min = energy_min = 1000000
    n_frames = 0
#<<<<<<< Updated upstream
    with open(os.path.join(in_dir, 'metadata_that.csv'), encoding='utf-8') as f:  #chane
#=======
    #with open(os.path.join(in_dir, 'metadata_vowel.csv'), encoding='utf-8') as f:   #修改寫新的metadata.csv
#>>>>>>> Stashed changes
        for line in f:
            parts = line.strip().split('|')
            basename = parts[0]
            text = parts[2]
            
            ret = process_utterance(in_dir, out_dir, basename)     #把此行寫進下六行的if內
            if ret is None:
                continue
            else:
                info, f_max, f_min, e_max, e_min, n = ret
            #added by eric
            '''
            print("info:",info)
            print("f_max:",f_max)
            print("f_min:",f_min)
            print("e_max:",e_max)
            print("e_min:",e_min)
            print("n:",n)
            print(done)
            '''
            
#<<<<<<< Updated upstream
            if basename[:2] in ['06']:
#=======
            #if basename[:5] in ['LJ001', 'LJ002', 'LJ003']:      #調整要收集的speaker(我們只要train存五個speaker,val先定為存1個speaker)
#>>>>>>> Stashed changes
                val.append(info)
            else:
                train.append(info)

            if index % 100 == 0:
                print("Done %d" % index)
            index = index + 1

            f0_max = max(f0_max, f_max)
            f0_min = min(f0_min, f_min)
            energy_max = max(energy_max, e_max)
            energy_min = min(energy_min, e_min)
            n_frames += n
    
    with open(os.path.join(out_dir, 'stat.txt'), 'w', encoding='utf-8') as f:     
        strs = ['Total time: {} hours'.format(n_frames*hp.hop_length/hp.sampling_rate/3600),
                'Total frames: {}'.format(n_frames),
                'Min F0: {}'.format(f0_min),
                'Max F0: {}'.format(f0_max),
                'Min energy: {}'.format(energy_min),
                'Max energy: {}'.format(energy_max)]
        for s in strs:
            print(s)
            f.write(s+'\n')
    
    return [r for r in train if r is not None], [r for r in val if r is not None]

def process_utterance(in_dir, out_dir, basename):
    wav_path = os.path.join(in_dir, 'wavs', '{}.wav'.format(basename))   #更改wav_path
    tg_path = os.path.join(out_dir, 'TextGrid', '{}.TextGrid'.format(basename)) #更改tg_path(做新的textgrid)
    
    # Get alignments
    textgrid = tgt.io.read_textgrid(tg_path)
    phone, duration, start, end = get_alignment(textgrid.get_tier_by_name('phones'))
    '''
    print("basename:",basename)
    print("phone:",phone)
    print("duration:",duration)
    print("start:",start)
    print("end",end)
    '''
    text = '{'+ '}{'.join(phone) + '}' # '{A}{B}{$}{C}', $ represents silent phones
    text = text.replace('{$}', ' ')    # '{A}{B} {C}'
    text = text.replace('}{', ' ')     # '{A B} {C}'

    if start >= end:
        return None

    # Read and trim wav files
    _, wav = read(wav_path)
    #print("len of wav(before):", len(wav))
    wav = wav[int(hp.sampling_rate*start):int(hp.sampling_rate*end)].astype(np.float32)

    #print(np.size(wav,0))           #自加: remove the wav files that are too short
    if np.size(wav,0)<1024:
        return None
    '''
    print("sum of duration:", sum(duration))
    print("len of wav(after)", len(wav))
    '''
    # Compute fundamental frequency
    f0, _ = pw.dio(wav.astype(np.float64), hp.sampling_rate, frame_period=hp.hop_length/hp.sampling_rate*1000)            #change from dio to harvest
    f0 = f0[:sum(duration)]
    if max(f0)==0:                    #自加: remove the wav files which f0 are all 0
        return None

    # Compute mel-scale spectrogram and energy
    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(torch.FloatTensor(wav))
    mel_spectrogram = mel_spectrogram.numpy().astype(np.float32)[:, :sum(duration)]
    energy = energy.numpy().astype(np.float32)[:sum(duration)]
    if mel_spectrogram.shape[1] >= hp.max_seq_len:
        return None
    '''
    #added by eric
    print("wav:\n",wav)
    print("f0:\n",f0)
    print("mel_spectrogram:\n",mel_spectrogram)
    print("energy:",energy)
    '''
    # Save alignment
    ali_filename = '{}-ali-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'alignment', ali_filename), duration, allow_pickle=False)                      #更改儲存路徑

    # Save fundamental prequency
    f0_filename = '{}-f0-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'f0', f0_filename), f0, allow_pickle=False)  #更改儲存路徑

    # Save energy
    energy_filename = '{}-energy-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'energy', energy_filename), energy, allow_pickle=False)         #更改儲存路徑

    # Save spectrogram
    mel_filename = '{}-mel-{}.npy'.format(hp.dataset, basename)
#<<<<<<< Updated upstream
    np.save(os.path.join(out_dir, 'mel', mel_filename), mel_spectrogram.T, allow_pickle=False)

    return '|'.join([basename, text]), max(f0), min([f for f in f0 if f>0]), max(energy), min(energy), mel_spectrogram.shape[1]              #change: f0 can be zero
#=======
    #np.save(os.path.join(out_dir, 'mel', mel_filename), mel_spectrogram.T, allow_pickle=False)       #更改儲存路徑
 
    #return '|'.join([basename, text]), max(f0), min([f for f in f0 if f != 0]), max(energy), min(energy), mel_spectrogram.shape[1]
#>>>>>>> Stashed changes
