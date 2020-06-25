from fcp_0406_sym_syn import FCP_Hawkes
from utilities import process_txt_asym, preprocess_txt, process_txt_sym, preprocess_undata, preprocess_undata_0323
from syn_0405 import syn_0405
import numpy as np 
import time
import pickle
from fcp_0406_sym_syn import logit
import copy
# b ----- eta, zeta direction reversible


b_prior_mu = 0
b_prior_sigma = 5
zeta_prior_mu = 0
zeta_prior_sigma = 5
eta_prior_mu = 0
eta_prior_sigma = 5
b_max = 20
eta_max = 3
zeta_max = 3
sample_num = 50
mh_number = 1
# d_num, u_id, observation_list, observation_table, ob_binary,binary_x, binary_y, T = process_txt('C:\\Users\\zy3\\work\\workspace\\fcp_windows\\radoslaw_email.txt')
# d_num, u_id, observation_list, observation_table, ob_binary,binary_x, binary_y, T = process_txt('out.radoslaw_email_email')


'''
infectious data
dat = preprocess_txt('infectious')
d_num, u_id, observation_list, ob_binary,binary_x, binary_y, T = process_txt_sym(dat)
'''


'''
undata
'''
d_num , observation_list, T = syn_0405()
u_id = []


print('overall T', T, 'd_num', d_num)



# cluster_index = np.load('syn_c_index_80_dynamic.npy')
cluster_index = []
fcp = FCP_Hawkes(b_prior_mu, b_prior_sigma, zeta_prior_mu, zeta_prior_sigma, 
eta_prior_mu, eta_prior_sigma, observation_list, b_max, eta_max, zeta_max, particle_num=40, c_max=1000, 
d_num = d_num, u_id=u_id, nu=0.04, xi=0.04, b=b_max/2, zeta=zeta_max/2, eta=eta_max/2, T=T, coefficient_omgea=2, cluster_index = cluster_index)

over_likelog = np.zeros(sample_num)
scaling_likelog = np.zeros(sample_num)

for sample_index in range(sample_num):
    print(sample_index,'sample_index')

    start = time.time()
    fcp.sample_z(sample_index)
    print(time.time()-start,'sample z time')
    print(fcp.path_entity,'path_entity')
    
    for t in range(len(fcp.FCPList_exist)):
        
        for m in range(len(fcp.FCPList_appear[t])):
            for n in range(m,len(fcp.FCPList_appear[t])):
                c_send = fcp.FCPList_appear[t][m]
                c_receive = fcp.FCPList_appear[t][n]
                print(c_send,fcp.c_time[c_send,0],c_receive,fcp.c_time[c_receive,0],
                'c_send,self.c_time[c_send,0],c_receive,self.c_time[c_receive,0]')
                print(c_send,c_receive,'c_send,c_receive')
                if c_send == c_receive and fcp.entity_num[c_send] == 1:
                    pass
                else:
                    print(c_send,c_receive,'c_send,c_receive')

                    for n in range(3):
                        for mh in range(mh_number):
                            s = fcp.sample_hawkes(n,c_send, c_receive) 
                            over_likelog[sample_index] = over_likelog[sample_index]+s
        
        if t!=0:
            print(fcp.FCPList_remain[t],'fcp.FCPList_remain[t]')
            print(fcp.FCPList_appear[t],'fcp.FCPList_appear[t]')
            for c_send in fcp.FCPList_remain[t]:
                for c_receive in fcp.FCPList_appear[t]:
                    if c_send == c_receive and fcp.entity_num[c_send] == 1:
                        pass
                    else:
                        print(c_send,c_receive,'c_send,c_receive')
                        for n in range(3):
                            for mh in range(mh_number):
                                s = fcp.sample_hawkes(n,c_send, c_receive)
                                over_likelog[sample_index] = over_likelog[sample_index]+s
            
            # print(fcp.FCPList_exist[t],'fcp.FCPList_exist[t]')
            # print(fcp.FCPList_appear[t],'fcp.FCPList_appear[t]')
    

    for n in range(3):
        scaling_likelog[sample_index] = scaling_likelog[sample_index] + fcp.sample_scaling(n)

    print(fcp.FCPList_time)
    for exist in fcp.FCPList_exist:
        print(exist,'exist')
    print(time.time()-start,'sample all time')

    # if sample_index == 3:
    #     exit()
    
    

print(scaling_likelog)

for i in range(fcp.d_num):
    print(fcp.path_entity[i])

np.save('over_like',over_likelog)
np.save('scaling_like',scaling_likelog)

for exist in fcp.FCPList_exist:
    print(exist,'exist')
    
    for m in exist:
        for n in exist:
            print(np.around(fcp.b*logit(fcp.Hawkes_b[m,n]), decimals=1),
            np.around(fcp.eta*logit(fcp.Hawkes_eta[m,n]), decimals=1),
            np.around(fcp.zeta*logit(fcp.Hawkes_zeta[m,n]), decimals=1),m,n,
            'fcp.Hawkes_b[m,n],fcp.Hawkes_eta[m,n],fcp.Hawkes_zeta[m,n],m,n')

# extend all path
        
        # extend all entity path based on the extended self.FCPList_time


live_c_index = np.where(fcp.c_time[:,0]>-1)[0]

ori_time = np.unique(fcp.c_time[live_c_index,0])
ori_time = np.sort(ori_time)
print(ori_time,'ori_time')
pe = np.zeros((fcp.d_num,len(ori_time)))-1
for i in range((fcp.d_num)):
    path_i = fcp.path_entity[i]
    for p in path_i:
        s = fcp.c_time[p,0]
        for j in range(len(ori_time)):
            if ori_time[j] == s:
                pe[i,j] = p
                break
    for j in range(1,len(ori_time)):
        if pe[i,j] == -1:
            pe[i,j] = pe[i,j-1]             
            
            
        
for i in range(fcp.d_num):
    print(pe[i,:],i,'path_i')


filehandler = open('fcprecord_0409_1', 'wb') 
pickle.dump(fcp, filehandler)

# filehandler = open('fcprecord_0409_1', 'rb') 
# yz = pickle.load(filehandler)
# assert isinstance(yz, FCP_Hawkes)
# print(yz.path_entity)
# print(yz.FCPList_time)