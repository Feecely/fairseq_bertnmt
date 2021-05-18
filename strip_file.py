import sys

with open(sys.argv[1], 'r', encoding='utf8') as f:
    lines = f.readlines()
    with open(sys.argv[1]+'.strip', 'w', encoding='utf8') as out_f:
        for line in lines:
            if line.startswith('b'):
                out_f.write(line.strip()[2:-1] + '\n')
            else:
                out_f.write(line.strip() + '\n')