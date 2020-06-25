# manually generate syn data


# we have 3 interval: 0-10 100 counts; 10-20 0 counts; 20-30 100 counts 

import numpy as np 

d_num = 20
def syn_0405():
    ob = []
    for i in range(d_num):
        ob.append([])
        for j in range(d_num):
            ob[i].append([])
    print(ob)
    c_index = np.array([0,0,0,0,1,1,1,1])
    c_index = np.zeros(d_num,dtype=int)
    c_index[int(d_num/2):] = 1

    for i in range(300):
        if i<100 or i>200:
            for m in range(d_num):
                for n in range(m,d_num):
                    if m != n:
                        ob[m][n].append(i/10)
                        ob[n][m].append(i/10)

        else:
            for m in range(len(c_index)):
                for n in range(m,len(c_index)):
                    if m !=n and c_index[m] == c_index[n]:
                        ob[m][n].append(i/10)
                        ob[n][m].append(i/10)

            
    # print(ob[0][1])
    # print(ob[0][4])
    # exit()

    return d_num , ob, 30