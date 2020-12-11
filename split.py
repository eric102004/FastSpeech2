import os

def split():
    speaker_data_list = [[] for i in range(50)]
    with open("val.txt", 'r+') as F:
        lines  = F.readlines()
        for line in lines:
            if line[3]=='0':
                idx = int(line[4])-1
            else:
                idx = int(line[3:5])-1
            speaker_data_list[idx].append(line)
    for i in range(3):
        print(len(speaker_data_list[i]))
        with open(f"val_{i+1}.txt", 'w+') as F:
            for data in speaker_data_list[i]:
                F.write(data)


if __name__ =="__main__":
    split()
    
