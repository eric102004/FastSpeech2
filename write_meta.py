import os
import glob
import hparams as hp

#speaker_list = [103,1069,1088,1098,1116]    # the version for female speaker only
data_path = hp.data_path

def write_meta():
    speaker_list = os.listdir(os.path.join(data_path, 'train-clean-100'))
    with open(os.path.join(data_path, 'metadata.csv'), 'w+') as F:
        for speaker in speaker_list:
            print('processing speaker:', speaker)
            speaker_sub_list = os.listdir(os.path.join(data_path, 'train-clean-100', speaker))
            for speaker_sub in speaker_sub_list:
                wav_dir = os.path.join(data_path, 'train-clean-100', speaker, speaker_sub)
                for f in glob.glob(os.path.join(wav_dir, '*.wav')):
                    filename = f[:-4].split('/')[-1]
                    text_file = open(os.path.join(wav_dir, filename) + '.original.txt')
                    text = text_file.readline()
                    line = '|'.join([speaker, speaker_sub, filename, text])
                    F.write(line + '\n')
    return None

def write_meta_dev():
    speaker_list = os.listdir(os.path.join(data_path, 'dev-clean'))
    with open(os.path.join(data_path, 'metadata_dev.csv'), 'w+') as F:
        for speaker in speaker_list:
            print('processing speaker:', speaker)
            speaker_sub_list = os.listdir(os.path.join(data_path, 'dev-clean', speaker))
            for speaker_sub in speaker_sub_list:
                wav_dir = os.path.join(data_path, 'dev-clean', speaker, speaker_sub)
                for f in glob.glob(os.path.join(wav_dir, '*.wav')):
                    filename = f[:-4].split('/')[-1]
                    text_file = open(os.path.join(wav_dir, filename) + '.original.txt')
                    text = text_file.readline()
                    line = '|'.join([speaker, speaker_sub, filename, text])
                    F.write(line + '\n')
    return None
def write_meta_test():
    speaker_list = os.listdir(os.path.join(data_path, 'test-clean'))
    with open(os.path.join(data_path, 'metadata_test.csv'), 'w+') as F:
        for speaker in speaker_list:
            print('processing speaker:', speaker)
            speaker_sub_list = os.listdir(os.path.join(data_path, 'test-clean', speaker))
            for speaker_sub in speaker_sub_list:
                wav_dir = os.path.join(data_path, 'test-clean', speaker, speaker_sub)
                for f in glob.glob(os.path.join(wav_dir, '*.wav')):
                    filename = f[:-4].split('/')[-1]
                    text_file = open(os.path.join(wav_dir, filename) + '.original.txt')
                    text = text_file.readline()
                    line = '|'.join([speaker, speaker_sub, filename, text])
                    F.write(line + '\n')
    return None
if __name__ =='__main__':
    #write_meta(speaker_list)
    #write_meta_dev()
    write_meta_test()

