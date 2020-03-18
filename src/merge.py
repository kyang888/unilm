import os
BIG_SPLIT_TOKENS = ['。', '？', '!', '！', '?']
def split_sentece(text):
    t = ""
    for w in text.split('\n'):
        t += w
        if w in BIG_SPLIT_TOKENS:
            t += "\n"
    return t
def merge(files):
    r = []
    tmp = ""
    for i,f in enumerate(files):
        with open(f,'r',encoding='utf-8') as f:
            lines = f.readlines()
            print(len(lines))
            if i == 0:
                for j,line in enumerate(lines):
                    if (j+1) % 4 == 2:
                        line = split_sentece(line)
                    tmp += line
                    if j + 1 % 4 == 0:
                        r.append(tmp)
                        tmp = ""
                print(len(r))
                print(r[-1])
            else:
                t = 0
                for j,line in enumerate(lines):
                    if (j+1) % 4 == 0:
                        r[t] = r[t] + line
                        t += 1
    r = sorted(r,key=lambda x:len(x))
    with open('/data/nfs/yangkang227/summary/result/all_rules.txt', 'w') as f:
        for line in r:
            f.write(line + '\n')
path = []
dir = '/data/nfs/yangkang227/summary/result'
for i in range(4):
    path.append(os.path.join(dir, 'nlpcc_eval_{}_4_150.txt'.format(i)))
#merge(path)
with open("111.txt",'r',encoding='utf-8') as f:
    print(f.readlines())
