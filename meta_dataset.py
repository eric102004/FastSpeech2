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
    def __init__(self, mode = 'train', num_subtasks = hparams.num_subtasks, num_subtask_training_data = hparams.num_subtask_training_data, num_subtask_testing_data = hparams.num_subtask_testing_data, filelist=[f"train_{i}.txt" for i in range(1,hparams.num_subtasks+1)], sort=True):
        self.num_subtasks = num_subtasks
        if mode =='train':
             self.filelist = [f"train_{i}.txt" for i in range(1,self.num_subtasks+1)]
        elif mode =='val':
             self.filelist = [f"val_{i}.txt" for i in range(1, self.num_subtasks+1)]
        else:
             raise ValueError("mode should be train or val") 
        self.num_subtask_training_data = num_subtask_training_data
        self.num_subtask_testing_data = num_subtask_testing_data
        self.basename_tr, self.text_tr, self.basename_te, self.text_te = meta_process_meta(self.filelist, self.num_subtasks, self.num_subtask_training_data, self.num_subtask_testing_data)
        self.sort = sort

    def __len__(self):
        return len(self.text_tr)

    def __getitem__(self, idx):
        basename_list_tr = self.basename_tr[idx]
        phone_list_tr = [np.array(text_to_sequence(self.text_tr[idx][task], [])) for task in range(self.num_subtasks)]
        sample_list_tr = []
        for task in range(self.num_subtasks):
            mel_path = os.path.join(
                hparams.preprocessed_path, "mel", "{}-mel-{}.npy".format(hparams.dataset, basename_list_tr[task]))
            mel_target = np.load(mel_path)
            D_path = os.path.join(
                hparams.preprocessed_path, "alignment", "{}-ali-{}.npy".format(hparams.dataset, basename_list_tr[task]))
            D = np.load(D_path)
            f0_path = os.path.join(
                hparams.preprocessed_path, "f0", "{}-f0-{}.npy".format(hparams.dataset, basename_list_tr[task]))
            f0 = np.load(f0_path)
            energy_path = os.path.join(
                hparams.preprocessed_path, "energy", "{}-energy-{}.npy".format(hparams.dataset, basename_list_tr[task]))
            energy = np.load(energy_path)
        
            sample = {"id": basename_list_tr[task],
                      "text": phone_list_tr[task],
                      "mel_target": mel_target,
                      "D": D,
                      "f0": f0,
                      "energy": energy}
            sample_list_tr.append(sample)
        #sample = {'train':[tr_xs,tr_ys], 'test':[tst_xs,tst_ys]}
        #load meta - testing data 
        basename_list_te = self.basename_te[idx]
        phone_list_te = [np.array(text_to_sequence(self.text_te[idx][task], [])) for task in range(self.num_subtasks)]
        sample_list_te = []
        for task in range(self.num_subtasks):
            mel_path = os.path.join(
                hparams.preprocessed_path, "mel", "{}-mel-{}.npy".format(hparams.dataset, basename_list_te[task]))
            mel_target = np.load(mel_path)
            D_path = os.path.join(
                hparams.preprocessed_path, "alignment", "{}-ali-{}.npy".format(hparams.dataset, basename_list_te[task]))
            D = np.load(D_path)
            f0_path = os.path.join(
                hparams.preprocessed_path, "f0", "{}-f0-{}.npy".format(hparams.dataset, basename_list_te[task]))
            f0 = np.load(f0_path)
            energy_path = os.path.join(
                hparams.preprocessed_path, "energy", "{}-energy-{}.npy".format(hparams.dataset, basename_list_te[task]))
            energy = np.load(energy_path)
        
            sample = {"id": basename_list_te[task],
                      "text": phone_list_te[task],
                      "mel_target": mel_target,
                      "D": D,
                      "f0": f0,
                      "energy": energy}
            sample_list_te.append(sample)
        #merge into a dict
        #sample = dict()
        #sample['training'] = sample_list_tr
        #sample['testing'] = sample_list_te
        return sample_list_tr, sample_list_te

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
        
        #creating log_Ds
        #log_Ds = [np.log(D + hparams.log_offset) for D in Ds]

        #print("log_Ds.shape:",len(log_Ds),log_Ds[0].shape)
        #print("Ds.shape",len(Ds),Ds[0].shape)
        #mel_targets_pad = pad_2D(mel_targets)
        #print("mel_targets.shape:",len(mel_targets),len(mel_targets[0]), len(mel_targets[0][0]))
        #print("mel_tagets.shape(after padding):",mel_targets_pad.shape)

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
        output_list_tr = []
        output_list_te = []
        for task in range(self.num_subtasks):
            len_arr = np.array([d[task]["text"].shape[0] for d,_ in batch])    #an array recording len of text in each sample
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
            #output = self.reprocess(batch, index_arr, task)      #output is a sample for a subtask
            output = self.reprocess([b for b,_ in batch], index_arr, task)
            output_list_tr.append(output)

            #build output_list_te
            len_arr = np.array([d[task]["text"].shape[0] for _,d in batch])
            index_arr = np.argsort(-len_arr)
            #output = self.reprocess(batch, index_arr, task)
            output = self.reprocess([b for _,b in batch], index_arr, task)
            output_list_te.append(output)

        return output_list_tr, output_list_te

if __name__ == "__main__":
    # Test
    # Test
    
    dataset = Dataset(mode = 'train', num_subtasks = 5, num_subtask_training_data = 3, num_subtask_testing_data = 3)
    print("filelist:", dataset.filelist)
    training_loader = DataLoader(dataset, batch_size=3, shuffle=False, collate_fn=dataset.collate_fn,
        drop_last=True, num_workers=0)
    total_step = hparams.epochs * len(training_loader)

    cnt = 0
    for i, (batch_tr, batch_te) in enumerate(training_loader):       #次數=num_subtask_data / batch_size
        print('len(batch_tr):',len(batch_tr))
        print('len(batch_te):',len(batch_te))
        for j, sample in enumerate(batch_te):            #次數
            #print("i:",i)
            #print('j:',j)
            #print(sample)
            #print("mel_target.shape:",sample["mel_target"][0].shape)
            #print("D.shape:",sample["D"][0])
            for k in range(3):                      #次數:batch_size
                mel_target = torch.from_numpy(                    #change from from_numpy to tensor 
                    sample["mel_target"][k]).float().to(device)
                D = torch.from_numpy(sample["D"][k]).int().to(device)  #change from from_numpy to tensor
                text = torch.from_numpy(sample["text"][k]).int().to(device)
                log_D = torch.from_numpy(sample["log_D"][k]).float().to(device)
                f0 = torch.from_numpy(sample["f0"][k]).float().to(device)
                energy = torch.from_numpy(sample["energy"][k]).float().to(device)
                print("mel_target_len:", mel_target.shape[0])
                print("sum of split len:",D.sum().item())
                print("len of text:",text.shape[0])
                print("len of log_D:", log_D.shape[0])
                print("len of f0", f0.shape[0])
                print("len fo energy:", energy.shape[0])
                if mel_target.shape[0] == D.sum().item():
                    #print("length met")
                    cnt += 1
                else:
                    print("length not met")
    print("cnt:",cnt)

