import os
import hparams as hp

def data_check():
    max_count = 0
    for speaker in os.listdir(os.path.join(hp.preprocessed_path, 'TextGrid')):
        count=0
        for sub in os.listdir(os.path.join(hp.preprocessed_path, 'TextGrid', speaker)):
            if sub[-4:]!='.txt':
                count += len(os.listdir(os.path.join(hp.preprocessed_path, 'TextGrid', speaker, sub)))
        max_count = max(max_count, count)
    print(max_count)
        #if count<5:
        #    print(speaker, count)

if __name__ == '__main__':
    data_check()
