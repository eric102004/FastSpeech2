#from ..data import ljspeech
import numpy as np
import scipy.io.wavfile as read
import torch
import tgt
import audio as Audio
from utils import get_alignment
#from text import _clean_text
import os
import hparams as hp

def process_utterance_try(in_dir, out_dir, basename):
    wav_path = os.path.join(in_dir, 'wavs', '{}.wav'.format(basename))
    tg_path = os.path.join(out_dir, 'TextGrid', '{}.TextGrid'.format(basename)) 
    
    # Get alignments
    textgrid = tgt.io.read_textgrid(tg_path)
    phone, duration, start, end = get_alignment(textgrid.get_tier_by_name('phones'))
    print("phone:", phone)
    print("duration:",duration)
    print("start", start)
    print("end:",end)
    text = '{'+ '}{'.join(phone) + '}' # '{A}{B}{$}{C}', $ represents silent phones
    text = text.replace('{$}', ' ')    # '{A}{B} {C}'
    text = text.replace('}{', ' ')     # '{A B} {C}'

    if start >= end:
        return None

    # Read and trim wav files
    _, wav = read(wav_path)
    wav = wav[int(hp.sampling_rate*start):int(hp.sampling_rate*end)].astype(np.float32)
    
    # Compute fundamental frequency
    f0, _ = pw.dio(wav.astype(np.float64), hp.sampling_rate, frame_period=hp.hop_length/hp.sampling_rate*1000)
    f0 = f0[:sum(duration)]

    # Compute mel-scale spectrogram and energy
    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(torch.FloatTensor(wav))
    mel_spectrogram = mel_spectrogram.numpy().astype(np.float32)[:, :sum(duration)]
    energy = energy.numpy().astype(np.float32)[:sum(duration)]
    if mel_spectrogram.shape[1] >= hp.max_seq_len:
        return None

    #check some value
    print("wav:\n",wav)
    print("f0:\n",f0)
    print("mel_spectrogram:\n",mel_spectrogram)
    print("energy:",energy)

if __name__=="__main__":
    in_dir = os.path.join("..",hp.data_path)
    out_dir = os.path.join("..",hp.preprocessed_path)
    basename = "LJ001-0001.wav" 
    process_utterance_try(in_dir, out_dir, basename) 
