from fairseq.binarizer import safe_readline
import sys

def remove_lines(filename, drop_list):
    with open(filename, 'r', encoding='utf-8') as f:
        line = safe_readline(f)
        reduce_line = []
        num = 0
        j = 0
        print(len(drop_list))
        while line:
            #import pdb; pdb.set_trace()
            if j < len(drop_list) and num == int(drop_list[j].split(',')[0]):
                j += 1
            else:
                reduce_line.append(line)
            num += 1
            if num % 100000 == 0:
                print(num)
            line = f.readline()
    # filename_reduce = save_path + filename.split('/')[-1]
    print('filter j is {}'.format(j))
    save_path = filename + '.filter'
    with open(save_path, 'w', encoding='utf8') as f:
        for line in reduce_line:
            f.write(str(line))
        f.close()

def reduce_words(filename, drop_filename):
    # read drop file
    with open(drop_filename, 'r', encoding='utf-8') as f:
        line = safe_readline(f)
        drop_list = []
        while line:
            drop_list.append(line)
            line = f.readline()
    # name_list = ['.en', '.de', '.bert.en', '.bert.de', '.bart.en', '.bart.de']
    # for name in name_list:
    remove_lines(filename, drop_list)

filename = sys.argv[1]
drop_filename = sys.argv[2]
reduce_words(filename, drop_filename)