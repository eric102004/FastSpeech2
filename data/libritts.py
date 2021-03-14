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
import torchaudio

def prepare_align(in_dir):
    with open(os.path.join(in_dir, 'metadata_that.csv'), encoding='utf-8') as f:  #change
    #with open(os.path.join(in_dir, 'metadata_vowel.csv'), encoding='utf-8') as f:  #修改
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
    with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:  #chane
        for line in f:
            parts = line.strip().split('|')
            speaker = parts[0]
            speaker_sub = parts[1]
            basename = parts[2]
            text = parts[3]

            try:
                ret = process_utterance(in_dir, out_dir, speaker, speaker_sub, basename)     #把此行寫進下六行的if內
            except:
                ret = None
                print('error file:',speaker, speaker_sub, basename)
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
            
            if basename[:4] in ['1116']:            
                val.append(info)
            else:
                train.append(info)

            if index % 100 == 0:
                print("#files done %d" % index)
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


def build_from_path_meta(in_dir, out_dir):
    index = 1
    train = dict()
    val = dict()
    f0_max = energy_max = 0
    f0_min = energy_min = 1000000
    n_frames = 0
    for suffix in ['_test']:
        if suffix=='_dev':
            print('processing dev set')
        elif suffix=='':
            print('processing training set')
        elif suffix=='_test':
            print('processing testing set')
        with open(os.path.join(in_dir, f'metadata{suffix}.csv'), encoding='utf-8') as f:  #chane
            for line in f:
                parts = line.strip().split('|')
                speaker = parts[0]
                speaker_sub = parts[1]
                basename = parts[2]
                text = parts[3]

                #try:
                if True:
                    if suffix=='':
                        ret = process_utterance(in_dir, out_dir, speaker, speaker_sub, basename, dir_name='train-clean-100')
                    elif suffix=='_dev':
                        ret = process_utterance(in_dir, out_dir, speaker, speaker_sub, basename, dir_name='dev-clean')
                    elif suffix=='_test':
                        ret = process_utterance(in_dir, out_dir, speaker, speaker_sub, basename, dir_name='test-clean')
                #except:
                else:
                    ret = None
                    print('error file:',speaker, speaker_sub, basename)
                if ret is None:
                    continue
                else:
                    info, f_max, f_min, e_max, e_min, n = ret
                #added by eric
               
                if speaker+'.txt' in hp.filelist_tr:
                    if speaker not in train.keys():
                        train[speaker] = []
                    train[speaker].append(info)
                elif speaker+'.txt' in hp.filelist_val or speaker+'.txt' in hp.filelist_test:
                    if speaker not in val.keys():
                        val[speaker] = []
                    val[speaker].append(info)
                else:
                    print('not in the filelist!' + '\n')
                    continue
                    

                if index % 100 == 0:
                    print("#files done %d" % index)
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

    #remove None from 2 dicts
    for s in train.keys():
        train[s] = [r for r in train[s] if r is not None]
    for s in val.keys():
        val[s] = [r for r in val[s] if r is not None]
    return train, val

def process_utterance(in_dir, out_dir, speaker, speaker_sub, basename, dir_name='train-clean-100'):
    #wav_path = os.path.join(in_dir, 'train-clean-100', speaker, speaker_sub, '{}.wav'.format(basename))
    wav_path = os.path.join(in_dir, dir_name, speaker, speaker_sub, '{}.wav'.format(basename))
    tg_path = os.path.join(out_dir, 'TextGrid', speaker, speaker_sub, '{}.TextGrid'.format(basename)) #更改tg_path(做新的textgrid)
    
    # Get alignments
    try:
        textgrid = tgt.io.read_textgrid(tg_path)
    except:
        print('no textgrid file')
        return None
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
    #wav, _ = torchaudio.load(wav_path)
    #wav = wav[0].numpy()
    
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
    np.save(os.path.join(out_dir, 'mel', mel_filename), mel_spectrogram.T, allow_pickle=False)

    return '|'.join([basename, text]), max(f0), min([f for f in f0 if f>0]), max(energy), min(energy), mel_spectrogram.shape[1]              #change: f0 can be zero
    #np.save(os.path.join(out_dir, 'mel', mel_filename), mel_spectrogram.T, allow_pickle=False)       #更改儲存路徑
 
    #return '|'.join([basename, text]), max(f0), min([f for f in f0 if f != 0]), max(energy), min(energy), mel_spectrogram.shape[1]
