import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import re
from string import punctuation
from g2p_en import G2p

from fastspeech2 import FastSpeech2
from text import text_to_sequence, sequence_to_text
import hparams as hp
import utils
import audio as Audio

#import the modules needed for fine-tuning
from torch.utils.data import DataLoader
from loss import FastSpeech2Loss
from dataset import Dataset
from optimizer import ScheduledOptim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess(text):
    text = text.rstrip(punctuation)

    g2p = G2p()
    phone = g2p(text)
    phone = list(filter(lambda p: p != ' ', phone))
    phone = '{'+ '}{'.join(phone) + '}'
    phone = re.sub(r'\{[^\w\s]?\}', '{sp}', phone)
    phone = phone.replace('}{', ' ')
    
    print('|' + phone + '|')    
    sequence = np.array(text_to_sequence(phone, hp.text_cleaners))
    sequence = np.stack([sequence])

    return torch.from_numpy(sequence).long().to(device)

def get_and_fine_tune_FastSpeech2(num, loader):
    checkpoint_path = os.path.join(hp.checkpoint_path, "checkpoint_{}.pth.tar".format(num))
    #model = nn.DataParallel(FastSpeech2())
    model = FastSpeech2().to(device)
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    #model.requires_grad = False
    #model.eval()

    #fine-tuning
    #optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), betas=hp.betas, eps=hp.eps, weight_decay = hp.weight_decay)
    scheduled_optim = ScheduledOptim(optimizer, hp.decoder_hidden, hp.n_warm_up_step, args.restore_step)
    Loss = FastSpeech2Loss().to(device)

    #fine-tuning
    print('start fine-tuning')
    model = model.train()
    current_step = 0
    while current_step < hp.syn_fine_tune_step:
        for i,batchs in enumerate(loader):
            for j, data_of_batch in enumerate(batchs):
                # Get Data
                text = torch.from_numpy(data_of_batch["text"]).long().to(device)
                mel_target = torch.from_numpy(data_of_batch["mel_target"]).float().to(device)
                D = torch.from_numpy(data_of_batch["D"]).long().to(device)
                log_D = torch.from_numpy(data_of_batch["log_D"]).float().to(device)
                f0 = torch.from_numpy(data_of_batch["f0"]).float().to(device)
                energy = torch.from_numpy(data_of_batch["energy"]).float().to(device)
                src_len = torch.from_numpy(data_of_batch["src_len"]).long().to(device)
                mel_len = torch.from_numpy(data_of_batch["mel_len"]).long().to(device)
                max_src_len = np.max(data_of_batch["src_len"]).astype(np.int32)
                max_mel_len = np.max(data_of_batch["mel_len"]).astype(np.int32)
               
                # Forward
                mel_output, mel_postnet_output, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _ = model(
                    text, src_len, mel_len, D, f0, energy, max_src_len, max_mel_len)
                
                # Cal Loss
                mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss = Loss(
                        log_duration_output, log_D, f0_output, f0, energy_output, energy, mel_output, mel_postnet_output, mel_target, ~src_mask, ~mel_mask)
                total_loss = mel_loss + mel_postnet_loss + d_loss + f_loss + e_loss 
                
                # Backward
                total_loss = total_loss / hp.acc_steps
                total_loss.backward()
                
                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_thresh)

                # Update weights
                scheduled_optim.step_and_update_lr()
                scheduled_optim.zero_grad()
                
                current_step +=1

    print('finish fine-tuning...')
    model.requires_grad = False
    model.eval()
    return model

def synthesize(model, waveglow, melgan, text, sentence, prefix='', speaker):
    sentence = sentence[:200] # long filename will result in OS Error
    
    src_len = torch.from_numpy(np.array([text.shape[1]])).to(device)
        
    mel, mel_postnet, log_duration_output, f0_output, energy_output, _, _, mel_len = model(text, src_len)
    
    mel_torch = mel.transpose(1, 2).detach()
    mel_postnet_torch = mel_postnet.transpose(1, 2).detach()
    mel = mel[0].cpu().transpose(0, 1).detach()
    mel_postnet = mel_postnet[0].cpu().transpose(0, 1).detach()
    f0_output = f0_output[0].detach().cpu().numpy()
    energy_output = energy_output[0].detach().cpu().numpy()

    if not os.path.exists(os.path.join(hp.test_path, speaker)):
        os.makedirs(os.path.join(hp.test_path, speaker))

    Audio.tools.inv_mel_spec(mel_postnet, os.path.join(hp.test_path, speaker,'{}_griffin_lim_{}.wav'.format(prefix, sentence)))
    if waveglow is not None:
        utils.waveglow_infer(mel_postnet_torch, waveglow, os.path.join(hp.test_path, speaker,'{}_{}_{}.wav'.format(prefix, hp.vocoder, sentence)))
    if melgan is not None:
        utils.melgan_infer(mel_postnet_torch, melgan, os.path.join(hp.test_path, speaker,'{}_{}_{}.wav'.format(prefix, hp.vocoder, sentence)))
    
    utils.plot_data([(mel_postnet.numpy(), f0_output, energy_output)], ['Synthesized Spectrogram'], filename=os.path.join(hp.test_path, speaker, '{}_{}.png'.format(prefix, sentence)))


if __name__ == "__main__":
    # Test
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=30000)
    parser.add_argument('--restore_step',type=int, default=0)
    args = parser.parse_args()
    
    sentences = [
        "Advanced text to speech models such as Fast Speech can synthesize speech significantly faster than previous auto regressive models with comparable quality. The training of Fast Speech model relies on an auto regressive teacher model for duration prediction and knowledge distillation, which can ease the one to many mapping problem in T T S. However, Fast Speech has several disadvantages, 1, the teacher student distillation pipeline is complicated, 2, the duration extracted from the teacher model is not accurate enough, and the target mel spectrograms distilled from teacher model suffer from information loss due to data simplification, both of which limit the voice quality.",
        "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition",
        "in being comparatively modern.",
        "For although the Chinese took impressions from wood blocks engraved in relief for centuries before the woodcutters of the Netherlands, by a similar process",
        "produced the block books, which were the immediate predecessors of the true printed book,",
        "the invention of movable metal letters in the middle of the fifteenth century may justly be considered as the invention of the art of printing.",
        "And it is worth mention in passing that, as an example of fine typography,",
        "the earliest book printed with movable types, the Gutenberg, or \"forty-two line Bible\" of about 1455,",
        "has never been surpassed.",
        "Printing, then, for our purpose, may be considered as the art of making books by means of movable types.",
        "Now, as all books not primarily intended as picture-books consist principally of types composed to form letterpress,"
        ]

    syn_speaker_list = hp.synthesize_speaker_list
    for speaker in syn_speaker_list:
        print(f'fine-tuning on speaker:{speaker}')
        dataset = Dataset(f'{speaker}.txt')
        loader = DataLoader(dataset, batch_size=hp.batch_size**2, shuffle=True, collate_fn=dataset.collate_fn, drop_last=True, num_workers=0)
        model = get_and_fine_tune_FastSpeech2(args.step, loader).to(device)
        melgan = waveglow = None
        if hp.vocoder == 'melgan':
            melgan = utils.get_melgan()
            melgan.to(device)
        elif hp.vocoder == 'waveglow':
            waveglow = utils.get_waveglow()
            waveglow.to(device)
        
        for sentence in sentences:
            text = preprocess(sentence)
            synthesize(model, waveglow, melgan, text, sentence, prefix='step_{}'.format(args.step), speaker)
