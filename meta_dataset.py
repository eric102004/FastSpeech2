import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import math
import os

import hparams
import audio as Audio
from utils import pad_1D, pad_2D, meta_process_meta
from text import text_to_sequence, sequence_to_text

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Dataset(Dataset):
    def __init__(self, mode = 'train', num_subtasks = hparams.num_subtasks, num_subtask_data =hparams.num_subtask_data, filelist=[f"train_{i}.txt" for i in range(1,hparams.num_subtasks+1)], sort=True):
        self.num_subtasks = num_subtasks
        if mode =='train':
             self.filelist = [f"train_{i}.txt" for i in range(1,self.num_subtasks+1)]
        elif mode =='val':
             self.filelist = [f"val_{i}.txt" for i in range(1, self.num_subtasks+1)]
        else:
             raise ValueError("mode should be train or val") 
        self.num_subtask_data = num_subtask_data
        self.basename, self.text = meta_process_meta(filelist, self.num_subtasks, self.num_subtask_data)
        self.sort = sort

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename_list = self.basename[idx]
        phone_list = [np.array(text_to_sequence(self.text[idx][task], [])) for task in range(hparams.num_subtasks)]
        sample_list = []
        for task in hparams.num_subtasks:
            mel_path = os.path.join(
                hparams.preprocessed_path, "mel", "{}-mel-{}.npy".format(hparams.dataset, basename_list[task]))
            mel_target = np.load(mel_path)
            D_path = os.path.join(
                hparams.preprocessed_path, "alignment", "{}-ali-{}.npy".format(hparams.dataset, basename_list[task]))
            D = np.load(D_path)
            f0_path = os.path.join(
                hparams.preprocessed_path, "f0", "{}-f0-{}.npy".format(hparams.dataset, basename_list[task]))
            f0 = np.load(f0_path)
            energy_path = os.path.join(
                hparams.preprocessed_path, "energy", "{}-energy-{}.npy".format(hparams.dataset, basename_list[task]))
            energy = np.load(energy_path)
        
            sample = {"id": basename_list[task],
                      "text": phone_list[task],
                      "mel_target": mel_target,
                      "D": D,
                      "f0": f0,
                      "energy": energy}
            sample_list.append(sample)
        #sample = {'train':[tr_xs,tr_ys], 'test':[tst_xs,tst_ys]}
        return sample_list

    def reprocess(self, batch, cut_list, task):
        ids = [batch[ind][task]["id"] for ind in cut_list]
        texts = [batch[ind][task]["text"] for ind in cut_list]
        mel_targets = [batch[ind][task]["mel_target"] for ind in cut_list]
        Ds = [batch[ind][task]["D"] for ind in cut_list]
        f0s = [batch[ind][task]["f0"] for ind in cut_list]
        energies = [batch[ind][task]["energy"] for ind in cut_list]
        for text, D, id_ in zip(texts, Ds, ids):
            if len(text) != len(D):
                print(text, text.shape, D, D.shape, id_)
        length_text = np.array(list())
        for text in texts:
            length_text = np.append(length_text, text.shape[0])

        length_mel = np.array(list())
        for mel in mel_targets:
            length_mel = np.append(length_mel, mel.shape[0])
        
        texts = pad_1D(texts)
        Ds = pad_1D(Ds)
        mel_targets = pad_2D(mel_targets)
        f0s = pad_1D(f0s)
        energies = pad_1D(energies)
        log_Ds = np.log(Ds + hparams.log_offset)

        out = {"id": ids,
               "text": texts,
               "mel_target": mel_targets,
               "D": Ds,
               "log_D": log_Ds,
               "f0": f0s,
               "energy": energies,
               "src_len": length_text,
               "mel_len": length_mel}

        return out

    def collate_fn(self, batch):
        output_list = []
        for task in range(hparams.num_subtasks):
            len_arr = np.array([d[task]["text"].shape[0] for d in batch])    #an array recording len of text in each sample
            index_arr = np.argsort(-len_arr)     #sort array in decreasing order of text length         #the index array of all datas in a subtask
            batchsize = len(batch)
            #real_batchsize = int(math.sqrt(batchsize))          #need to modify
        
            '''
            cut_list = list()                    #need to modify(split batch into batch*batch)
            for i in range(real_batchsize):
               if self.sort:
                    cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])
                else:
                    cut_list.append(np.arange(i*real_batchsize, (i+1)*real_batchsize))
            '''
            output = self.reprocess(batch, index_arr, task)      #output is a sample for a subtask

        output_list.append(output)

        return output_list

if __name__ == "__main__":
    # Test
    # Test
    
    dataset = Dataset(mode = 'val', num_subtasks = 2, num_subtask_data = 3)
    print("filelist:", dataset.filelist)
    training_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn,
        drop_last=True, num_workers=0)
    total_step = hparams.epochs * len(training_loader)

    cnt = 0
    for i, batch in enumerate(training_loader):
        for j, sample in enumerate(batch):
            mel_target = torch.from_numpy(
                sample["mel_target"]).float().to(device)
            D = torch.from_numpy(sample["D"]).int().to(device)
            if mel_target.shape[1] == D.sum().item():
                cnt += 1
    print("cnt:",cnt)

