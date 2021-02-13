import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import math
import os

import hparams
import audio as Audio
from utils import pad_1D, pad_2D, process_meta, multi_process_meta
from text import text_to_sequence, sequence_to_text

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Dataset(Dataset):
    def __init__(self, filename="train.txt", sort=True, few_shot=None):
        if type(filename)==list:
            self.basename, self.text = multi_process_meta(filename)
        else:
            self.basename, self.text = process_meta(os.path.join(hparams.preprocessed_path, filename))
        self.sort = sort
        assert((few_shot==None) or (isinstance(few_shot,int)))
        self.few_shot = few_shot

        self.get_spk_table()

    def __len__(self):
        if self.few_shot==None:
            return len(self.text)
        else:
            return min(len(self.text), self.few_shot)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        phone = np.array(text_to_sequence(self.text[idx], []))
        mel_path = os.path.join(
            hparams.preprocessed_path, "mel", "{}-mel-{}.npy".format(hparams.dataset, basename))
        mel_target = np.load(mel_path)
        D_path = os.path.join(
            hparams.preprocessed_path, "alignment", "{}-ali-{}.npy".format(hparams.dataset, basename))
        D = np.load(D_path)
        f0_path = os.path.join(
            hparams.preprocessed_path, "f0", "{}-f0-{}.npy".format(hparams.dataset, basename))
        f0 = np.load(f0_path)
        energy_path = os.path.join(
            hparams.preprocessed_path, "energy", "{}-energy-{}.npy".format(hparams.dataset, basename))
        energy = np.load(energy_path)
        
        sample = {"id": basename,
                  "text": phone,
                  "mel_target": mel_target,
                  "D": D,
                  "f0": f0,
                  "energy": energy}

        return sample

    def reprocess(self, batch, cut_list):
        ids = [batch[ind]["id"] for ind in cut_list]

        if hparams.use_spk_embed:
            spk_ids = [self.spk_table[_id.split("_")[0]] for _id in ids]
        texts = [batch[ind]["text"] for ind in cut_list]
        mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
        Ds = [batch[ind]["D"] for ind in cut_list]
        f0s = [batch[ind]["f0"] for ind in cut_list]
        energies = [batch[ind]["energy"] for ind in cut_list]
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
        if hparams.use_spk_embed:
            out.update({"spk_ids": spk_ids})
        
        return out

    def collate_fn(self, batch):
        len_arr = np.array([d["text"].shape[0] for d in batch])
        index_arr = np.argsort(-len_arr)
        batchsize = len(batch)
        real_batchsize = int(math.sqrt(batchsize))

        cut_list = list()
        for i in range(real_batchsize):
            if self.sort:
                cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])
            else:
                cut_list.append(np.arange(i*real_batchsize, (i+1)*real_batchsize))
        
        output = list()
        for i in range(real_batchsize):
            output.append(self.reprocess(batch, cut_list[i]))

        return output

    def get_spk_table(self):
        
        spk = hparams.filelist_tr
        self.spk_table = dict()
        for i in range(len(spk)):
            self.spk_table[spk[i][:-4]]= i
        self.inv_spk_table = {i:spk_id for spk_id,i in self.spk_table.items()}


if __name__ == "__main__":
    # Test
    dataset = Dataset('val.txt')
    training_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate_fn,
        drop_last=True, num_workers=0)
    total_step = hparams.epochs * len(training_loader) * hparams.batch_size

    cnt = 0
    for i, batchs in enumerate(training_loader):
        print("len(batchs):",len(batchs))
        for j, data_of_batch in enumerate(batchs):
            print("i:",i)
            print("j:",j)
            mel_target = torch.from_numpy(
                data_of_batch["mel_target"]).float().to(device)
            D = torch.from_numpy(data_of_batch["D"]).int().to(device)
            print("mel_shape:",mel_target.shape)
            print("D.sum().item():",D.sum().item())
            if mel_target.shape[1] == D.sum().item():
                cnt += 1

    print(cnt, len(dataset))
