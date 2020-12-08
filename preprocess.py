import os
from data import ljspeech, blizzard2013
import hparams as hp

def read_metadata_all(train_path = 'preprocessed/LJSpeech/train_all.txt', val_path = 'preprocessed/LJSpeech/val_all.txt'):
    train = []
    #count_list = [0 for i in range(5)]
    train_count = 0
    target_speaker = 1
    val = []
    val_count = 0
    bound = 9
    with open(train_path , 'r+') as F:
        lines =  F.readlines()
        for line in lines:
            index = int(line[1])-1
            if index==target_speaker-1 and train_count<bound:
                train.append(line.strip())
                train_count+=1

    with open(val_path, 'r+') as F:
        lines = F.readlines()
        for line in lines:
            if val_count>=bound:
                break
            else:
                val.append(line.strip())
                val_count+=1

    print('num of training data:',len(train))
    print('num of val data:',len(val))
    assert len(train)==bound 
    assert len(val) ==bound
    return train, val
            

def write_metadata(train, val, out_dir = 'preprocessed/LJSpeech'):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in train:
            f.write(m + '\n')
    with open(os.path.join(out_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        for m in val:
            f.write(m + '\n')

def main():
    in_dir = hp.data_path
    out_dir = hp.preprocessed_path
    mel_out_dir = os.path.join(out_dir, "mel")
    if not os.path.exists(mel_out_dir):
        os.makedirs(mel_out_dir, exist_ok=True)
    ali_out_dir = os.path.join(out_dir, "alignment")
    if not os.path.exists(ali_out_dir):
        os.makedirs(ali_out_dir, exist_ok=True)
    f0_out_dir = os.path.join(out_dir, "f0")
    if not os.path.exists(f0_out_dir):
        os.makedirs(f0_out_dir, exist_ok=True)
    energy_out_dir = os.path.join(out_dir, "energy")
    if not os.path.exists(energy_out_dir):
        os.makedirs(energy_out_dir, exist_ok=True)

    if hp.dataset == "LJSpeech":
        train, val = ljspeech.build_from_path(in_dir, out_dir)
    if hp.dataset == "Blizzard2013":
        train, val = blizzard2013.build_from_path(in_dir, out_dir)
    write_metadata(train, val, out_dir)
    
if __name__ == "__main__":
    train, val =  read_metadata_all()
    write_metadata(train, val)
