import os
import glob
import hparams as hp

speaker_list = [103,1069,1088,1098,1116]    # the version for female speaker only
data_path = hp.data_path

def write_meta(speaker_list):
    with open(os.path.join(data_path, 'metadata.csv'), 'w+') as F:
        for speaker in speaker_list:
            speaker_sub_list = os.listdir(os.path.join(data_path, 'train-clean-100', str(speaker)))
            for speaker_sub in speaker_sub_list:
                wav_dir = os.path.join(data_path, 'train-clean-100', str(speaker), speaker_sub)
                for f in glob.glob(os.path.join(wav_dir, '*.wav')):
                    filename = f[:-4].split('/')[-1]
                    text_file = open(os.path.join(wav_dir, filename) + '.original.txt')
                    text = text_file.readline()
                    line = '|'.join([str(speaker), speaker_sub, filename, text])
                    F.write(line + '\n')
    return None

if __name__ =='__main__':
    write_meta(speaker_list)

