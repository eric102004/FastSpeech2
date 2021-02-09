import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import re
from string import punctuation
from g2p_en import G2p


from text import text_to_sequence, sequence_to_text
import hparams as hp
import utils
import audio as Audio

if hp.model_mode == 'meta':
    from fastspeech2 import FastSpeech2
elif hp.model_mode == 'baseline':
    from fastspeech2_emb import FastSpeech2
else:
    raise ValueError('model_mode should be meta or baseline')

#import the modules needed for fine-tuning
from torch.utils.data import DataLoader
from loss import FastSpeech2Loss
from dataset import Dataset
from optimizer import ScheduledOptim
from torch.utils.tensorboard import SummaryWriter


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

def get_and_fine_tune_FastSpeech2(num, loader, speaker):
    # makedir
    if not os.path.exists(os.path.join(hp.test_path, speaker)):
        os.makedirs(os.path.join(hp.test_path, speaker))
        os.makedirs(os.path.join(hp.test_path, speaker, 'train'))

    # init logger
    print('initing logger...')
    train_logger = SummaryWriter(os.path.join(hp.test_path, speaker, 'train'))


    if hp.exp_name in hp.exp_set:
        checkpoint_path = os.path.join(hp.checkpoint_path,hp.exp_name,"checkpoint_{}.pth.tar".format(num))
        print(f'load model in exp:{hp.exp_name}, step:{num}')
    else:
        checkpoint_path = os.path.join(hp.checkpoint_path, "checkpoint_{}.pth.tar".format(num))
    #model = nn.DataParallel(FastSpeech2())
    if hp.model_mode =='meta':
        model = FastSpeech2().to(device)
        try:
            model.load_state_dict(torch.load(checkpoint_path)['model'])
        except:
            try:
                ckpt = torch.load(checkpoint_path)['model']
                for n, p in ckpt.items():
                    if n[7:] not in model.state_dict():
                        print('not in meta_model:', n)
                        continue
                    if n[7:10]=='emb' and hp.use_pretrained_emb==False:
                        continue
                    if isinstance(p, nn.parameter.Parameter):
                        p = p.data
                    model.state_dict()[n[7:]].copy_(p)
            except:
                raise RuntimeError('Failed to load model')

    else:
        model = FastSpeech2(n_spkers=1).to(device)
        ckpt = torch.load(checkpoint_path)['model']
        for n, p in ckpt.items():
            if n[7:] not in model.state_dict():
                print('not in meta_model:', n)
                continue
            if n[7:10]=='emb' and hp.use_pretrained_emb==False:
                continue
            if isinstance(p, nn.parameter.Parameter):
                p = p.data
            model.state_dict()[n[7:]].copy_(p)
        
    #model.requires_grad = False
    #model.eval()

    #fine-tuning
    #optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), betas=hp.betas, eps=hp.eps, weight_decay = hp.weight_decay)
    scheduled_optim = ScheduledOptim(optimizer, hp.decoder_hidden, hp.n_warm_up_step, args.step)
    Loss = FastSpeech2Loss().to(device)

    #fine-tuning
    print('start fine-tuning')
    model = model.train()
    #check grad
    for n,p in model.named_parameters():
        #print(n, p.requires_grad)
        '''
        if n[:3] !='var':
            p.requires_grad = False
        '''
        if n[:3] not in hp.fine_tune_model_set or n[8:11]=='pos':
            p.requires_grad = False
        print(n, p.requires_grad)
    current_step = 0
    break_sig = False
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
                if hp.model_mode=='baseline' and hp.use_spk_embed:
                    spk_ids = torch.tensor([0]*hp.syn_fine_tune_batch_size).type(torch.int64).to(device)
                    mel_output, mel_postnet_output, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _ = model(text, src_len, mel_len, D, f0, energy, max_src_len, max_mel_len, speaker_ids = spk_ids)
                else:
                    mel_output, mel_postnet_output, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _ = model(text, src_len, mel_len, D, f0, energy, max_src_len, max_mel_len)

                # Cal Loss
                mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss = Loss(
                        log_duration_output, log_D, f0_output, f0, energy_output, energy, mel_output, mel_postnet_output, mel_target, ~src_mask, ~mel_mask)
                total_loss = mel_loss + mel_postnet_loss + d_loss + f_loss + e_loss

                train_logger.add_scalar('Loss/total_loss', total_loss, current_step)
                train_logger.add_scalar('Loss/mel_loss', mel_loss, current_step)
                train_logger.add_scalar('Loss/mel_postnet_loss', mel_postnet_loss, current_step)
                train_logger.add_scalar('Loss/duration_loss', d_loss, current_step)
                train_logger.add_scalar('Loss/F0_loss', f_loss, current_step)
                train_logger.add_scalar('Loss/energy_loss', e_loss, current_step)
               
                # print loss
                if (current_step+1)%10==0:
                    str2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Duration Loss: {:.4f}, F0 Loss: {:.4f}, Energy Loss: {:.4f};".format(total_loss, mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss)
                    print(str2 + '\n')

                # Backward
                total_loss = total_loss / hp.acc_steps
                total_loss.backward()
                
                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_thresh)

                # Update weights
                scheduled_optim.step_and_update_lr()
                scheduled_optim.zero_grad()
                
                current_step +=1
                if current_step>=hp.syn_fine_tune_step:
                    break_sig=True
                    break
            if break_sig: break

    print('finish fine-tuning...')
    model.requires_grad = False
    model.eval()
    return model

def synthesize(model, waveglow, melgan, text, sentence, speaker, prefix=''):
    sentence = sentence[:200] # long filename will result in OS Error
    
    src_len = torch.from_numpy(np.array([text.shape[1]])).to(device)
        
    mel, mel_postnet, log_duration_output, f0_output, energy_output, _, _, mel_len = model(text, src_len)
    
    mel_torch = mel.transpose(1, 2).detach()
    mel_postnet_torch = mel_postnet.transpose(1, 2).detach()
    mel = mel[0].cpu().transpose(0, 1).detach()
    mel_postnet = mel_postnet[0].cpu().transpose(0, 1).detach()
    f0_output = f0_output[0].detach().cpu().numpy()
    energy_output = energy_output[0].detach().cpu().numpy()


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
    parser.add_argument('--shot', type=int, default=1000)
    args = parser.parse_args()
    assert(hp.model_mode in ['baseline','meta'])
    sentences = ["Weather forecast for tonight: dark.",
            "I put a dollar in a change machine. Nothing changed.",
            "“No comment” is a comment.",
            "So far, this is the oldest I’ve been.",
            "I am in shape. Round is a shape."
        ]
    '''
    sentences = ["Maximilian.",
            "Villefort rose, half ashamed of being surprised in such a paroxysm of grief.",
            "The terrible office he had held for twenty-five years had succeeded in making him more or less than man.",
            "His glance, at first wandering, fixed itself upon Morrel.",
            "\"Go!--do you hear?\" said Villefort, while d'Avrigny advanced to lead Morrel out."
        ]
    '''
    syn_speaker_list = hp.synthesize_speaker_list
    for speaker in syn_speaker_list:
        print(f'fine-tuning on speaker:{speaker}')
        dataset = Dataset(f'{speaker}.txt', few_shot=args.shot)
        batch_size = min(args.shot, hp.syn_fine_tune_batch_size**2)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn, drop_last=True, num_workers=0)
        model = get_and_fine_tune_FastSpeech2(args.step, loader, speaker).to(device)
        melgan = waveglow = None
        if hp.vocoder == 'melgan':
            melgan = utils.get_melgan()
            melgan.to(device)
        elif hp.vocoder == 'waveglow':
            waveglow = utils.get_waveglow()
            waveglow.to(device)
        
        for sentence in sentences:
            text = preprocess(sentence)
            synthesize(model, waveglow, melgan, text, sentence, speaker, prefix='step_{}'.format(args.step))
