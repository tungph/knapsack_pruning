import re
import collections
import numpy as np
from operator import itemgetter
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(message)s')
with open("/Users/yonathan/Desktop/infer.txt", "r") as file:
    lst = file.read().split('\n')


def parse_file(lst):
    i = 1
    dct = {}
    while i < len(lst):
        o1 = lst[i]
        if o1.startswith('batch'):
            o2 = lst[i + 1]
            i += 1
            mtch1 = 'batch size: (\d+), number of channels (\d+), kernel size (\d+) and input dim (\d+)'
            mtch2 = 'MACS:(\d+) time: elapsed: (\d+\.\d+) ms. Ratio = (\d+\.\d+) (\D+)'
            x1 = re.match(mtch1, o1)
            x2 = re.match(mtch2, o2)
            data1 = list(x1.groups())
            data2 = list(x2.groups())[:-1]
            batch_size = int(data1[0])
            ch = int(data1[1])
            k = int(data1[2])
            dim = int(data1[3])
            total_macs = int(data2[0])*batch_size
            time = float(data2[1])
            r = batch_size*(ch*dim*k)**2/(1e6*time)
            key = total_macs
            if key in dct:
                dct[key].append((batch_size, ch, k, dim, time, r))
            else:
                dct[key] = [(batch_size, ch, k, dim, time, r)]

        i += 1
    return dct


dct = parse_file(lst)
dct = collections.OrderedDict(sorted(dct.items()))

fig, ax = plt.subplots()
index_x=[]
index_y=[]
for k, v in dct.items():
    lst_ratio = [x[-1] for x in v]
    if len(v) > 1:
        sorted_index = list(np.argsort(lst_ratio)[::-1])
        dct[k] = list(itemgetter(*sorted_index)(v))
    v = dct[k]
    logging.info(f'{k} : {list(v)}')
    for x in v:
        index_x.append(k)
        index_y.append(x[-2])

x=np.array(index_x)
y=np.array(index_y)


ax.scatter(x, y, marker='.', s=1)
# fig = plt.gcf()
# fig.set_size_inches(20, 20)
# plt.show()
plt.savefig('./temp.png', dpi=200)