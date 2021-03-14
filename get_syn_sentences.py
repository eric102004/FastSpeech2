import os 
import hparams as hp

def get_syn_sentences(num_sentences):
    speaker_list = [name[:-4] for name in hp.filelist_test]
    sentence_filename_dict = dict()
    for speaker in speaker_list:
        sentence_filename_dict[speaker] = []
        end_sig = 0
        sub_dir_list =  os.listdir(os.path.join(hp.data_path, 'test-clean', speaker))
        for sub_dir in sorted(sub_dir_list):
            current_path = os.path.join(hp.data_path,'test-clean',speaker,sub_dir)
            for filename in sorted(os.listdir(os.path.join(current_path))):
                if filename.endswith('original.txt'):
                    wav_filename = filename[:-12]+'wav'
                    wav_path = os.path.join(os.path.join(current_path,wav_filename))
                    if not os.path.exists(wav_path):
                        print(wav_filename)
                    assert(os.path.exists(wav_path))
                    with open(os.path.join(current_path, filename), 'r+') as F:
                        line = F.readline().strip()
                    if speaker!='61' and (len(line)>100 or len(line)<15):
                        continue
                    sentence_filename_dict[speaker].append((line, filename))
                    if len(sentence_filename_dict[speaker])>=num_sentences:
                        end_sig=1
                        break
            if end_sig:
                break
    return sentence_filename_dict


if __name__ == '__main__':
    sentence_filename_dict = get_syn_sentences(num_sentences=5)
    print('sentence__filename_dict:',sentence_filename_dict)
    print('num_sentence_check:')
    for speaker in sentence_filename_dict.keys():
        print(speaker, len(sentence_filename_dict[speaker]))
