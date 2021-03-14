import os
import random
import hparams as hp
def sample_file(speaker_id, shot, times):
    with open(os.path.join(hp.preprocessed_path, f'{speaker_id}.txt'), 'r+') as F:
        lines = F.readlines()
    for i in range(times):
        if len(lines)<shot:
            print(speaker_id, len(lines))
            print(f'pass {speaker_id}')
            continue
        sub_lines = random.sample(lines, shot)
        with open(os.path.join(hp.preprocessed_path, f'{speaker_id}_{shot}_{i}.txt'), 'w+') as f:
            for l in sub_lines:
                f.write(l)

if __name__ =='__main__':
    speaker_list = [filename[:-4] for filename in hp.filelist_test]
    for spk_id in speaker_list:
        sample_file(spk_id, 20, 1)
