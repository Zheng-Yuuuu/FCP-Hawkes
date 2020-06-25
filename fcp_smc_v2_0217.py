import numpy as np 
import copy
import numpy.random as npr



#### asymmetric version

npr.seed(50)

# fixed the cluster for entities
# to recover , just modify the initialization
      
class FCP_Hawkes(object):

    def __init__(self, b_prior_mu, b_prior_sigma, zeta_prior_mu, zeta_prior_sigma, eta_prior_mu, eta_prior_sigma,
                 observation_list, b_max, eta_max, zeta_max, particle_num = 100, c_max=20, 
                 d_num = 0, u_id = [],  
                 nu = 1, xi = 1, b = 1, zeta = 1, eta = 1, T = 0, coefficient_omgea = 2, cluster_index = []):
                
        # auxilary
        
        self.particle_num = particle_num
        self.u_id = u_id
        self.coefficient_omgea = coefficient_omgea
        self.undefine_para_value = -100
        self.c_max = c_max

        self.min = -10
        self.max = 10

        #hyperparamter
        
        self.b_prior_mu = b_prior_mu
        self.b_prior_sigma = b_prior_sigma
        self.zeta_prior_mu = zeta_prior_mu
        self.zeta_prior_sigma = zeta_prior_sigma
        self.eta_prior_mu = eta_prior_mu
        self.eta_prior_sigma = eta_prior_sigma
        self.b_max = b_max
        self.eta_max = eta_max
        self.zeta_max = zeta_max
        
        self.T = T
        
        # parameter
        self.nu = nu # fragmentation parameter
        self.xi = xi # coagulation parameter
        
        self.b = b # hawkes scaling parameter range (0, self.max_b) 
        self.eta = eta    
        self.zeta = zeta
    
        # hawkes parameter for community pair(b,zeta,eta)
        
        self.Hawkes_b = np.zeros((self.c_max,self.c_max))+self.undefine_para_value
        self.Hawkes_eta = np.zeros((self.c_max,self.c_max))+self.undefine_para_value
        self.Hawkes_zeta = np.zeros((self.c_max,self.c_max))+self.undefine_para_value
        
        self.old_Hawkes_b = np.zeros((self.c_max,self.c_max))+self.undefine_para_value
        self.old_Hawkes_eta = np.zeros((self.c_max,self.c_max))+self.undefine_para_value
        self.old_Hawkes_zeta = np.zeros((self.c_max,self.c_max))+self.undefine_para_value
        
        # FCPNode information
        
        self.children = [[] for i in range(c_max)]
        self.parents = [[] for i in range(c_max)]
        self.sliblings_depart = np.zeros(self.c_max,dtype = int)+self.c_max
        self.sliblings_arrive = np.zeros(self.c_max,dtype = int)+self.c_max

        
        self.entity_num = np.zeros(c_max,dtype=int)
        self.entity_list = [[] for i in range(c_max)]
        
        self.old_children = [[] for i in range(c_max)]
        self.old_parents = [[] for i in range(c_max)]   
        
        # community information
        
        self.FCPList_exist = []
        self.FCPList_time = []
        self.FCPList_disppear = []
        self.FCPList_appear = []
        self.FCPList_remain = []
        
        self.c_time = np.zeros((self.c_max,2)) # record beginning and ending time
        self.c_time[:,0] = -1 # use c_time[:,0] to check the community is alive
        self.c_time[:,1] = self.T # set initializtion of ending time being self.T for convenience
        
        # entity information
        
        self.d_num = d_num # entity number
        self.observation_list = observation_list # observation_list
        self.path_entity = [[] for i in range(self.d_num)] # community path for each entity
        

        # transition probability matrix
        
        self.trans_matrix = []
        
        #initialization
        
        # initialize c_time (community at time 0)

        self.cluster_index = cluster_index

        print(cluster_index,'cluster_index')
        
        for i in range(self.d_num):
            if self.cluster_index[i,1] == 1:
                self.cluster_index[i,1] = 3
            if self.cluster_index[i,2] == 1:
                self.cluster_index[i,2] = 4
            if self.cluster_index[i,3] == 1:
                self.cluster_index[i,3] =6
            if self.cluster_index[i,3] == 2:
                self.cluster_index[i,3] = 5
                
        print(self.cluster_index,'self.cluster_index')
        '''
        if cluster_index!=[]:

            for i in range(self.d_num):
                self.path_entity[i].append(cluster_index[i])
                self.c_time[cluster_index[i],0] = 0
        else:
            pass
        
        '''

        self.c_time[0,0] = 0
        self.c_time[1,0] = 0
        self.c_time[1,1] = 2.5
        self.c_time[2,0] = 2.5
        self.c_time[2,1] = 5
        self.c_time[3,0] = 2.5
        self.c_time[3,1] = 5
        self.c_time[4,0] = 5
        self.c_time[4,1] = 7.5
        self.c_time[5,0] = 7.5
        self.c_time[6,0] = 7.5
        
        for i in range(self.d_num):
            if self.cluster_index[i,0] == 0:
                self.path_entity[i].append(0)
                self.entity_list[0].append(i)
                self.entity_num[0] = self.entity_num[0]+1
            else:
                for j in range(np.shape(self.cluster_index)[1]):
                    c = self.cluster_index[i,j]
                    self.path_entity[i].append(c)
                    self.entity_list[c].append(i)
                    self.entity_num[c] = self.entity_num[c]+1
        
        fcpexist = [[0,1],[0,2,3],[0,4],[0,5,6]]

        for exist in fcpexist:
            for p in exist:
                for q in exist:
                    if p == q:
                        self.Hawkes_b[p,q] = 10
                        self.Hawkes_eta[p,q] = 0
                        self.Hawkes_zeta[p,q] = 0
                    else:
                        self.Hawkes_b[p,q] = -.5
                        self.Hawkes_eta[p,q] = 0
                        self.Hawkes_zeta[p,q] = 0
            
        

        self.get_fcp_from_c_time()

        print(self.path_entity,'self.path_entity init')


                
    def iter_path(self, node, path, total_path): # checked
        
        path.append(node)
        if self.children[node]==[]:
            total_path.append(copy.copy(path))
            del path[-1]
        else:
            for i in self.children[node]:
                self.iter_path(i, path, total_path)
            del path[-1]
              
    
    def sample_z(self):

        # jump_array: 0 (no jump) 1 (jump)
        
        wrong_signal = 0 #test_signal

        for i in range(self.d_num):

            # print(i, 'in sample_z') 
            

            prev_time = self.preprocess_entity_path_i(i)
            
            prev_time = [time for time in prev_time if time!=[]] # smc just need prev_time, so delete [] in prev_time
            
            self.get_fcp_from_c_time()

            self.process_entity_paths()

            # test begin
            
            # print('end of preprocess_entity_path in sample_z and check entity number is correct')

            # for m in range(len(self.path_entity)):
            #     print(self.path_entity[m],m,'self.path_entity[m] in sample_z')

            virtual_jump = self.sample_potential_jump()

            print(virtual_jump,'virtual jump')

            jump_array = self.derive_jump_array(virtual_jump, prev_time)

            self.extend_all_paths(i)

            print(self.FCPList_time,'(self.FCPList_time')
            
            path_i, empty_para_array = self.smc_z(i,jump_array) 

            print(path_i,'path_i')
            self.postprocess_after_sample_z(jump_array, path_i, i, empty_para_array) 

            self.get_fcp_from_c_time() 

            self.process_entity_paths() 

            self.get_fcp_from_c_time()

            c_index = np.where(self.c_time[:,0]<0)[0]

            self.remove_hawkes_parameter(c_index)

        
        # for pe in range(self.d_num):
        #     print(self.path_entity[pe],'pe')
        
        # exit()

            


            # test begin
        '''
            
            for l in self.FCPList_exist:
                for p in l:
                    for q in l:
                        if self.Hawkes_b[p,q]==self.undefine_para_value or self.Hawkes_b[q,p]==self.undefine_para_value:
                            print(self.Hawkes_b[p,q],self.Hawkes_b[q,p],p,q,'self.Hawkes_b[p,q]<=self.undefine_para_value or self.Hawkes_b[q,p]<=self.undefine_para_value in end of sample_z')
                            exit()

            ti = np.unique(self.c_time[:,0])
            # print(ti)
            ti = np.sort(ti, kind = 'mergesort')
            if ti[0] == -1:
                ti = np.delete(ti, 0) # delete the first item, the value of first item is inactive
                ti = np.delete(ti,0)
                for tii in ti:
                    index_1 = np.where(self.c_time[:,0]==tii)[0]
                    index_2 = np.where(self.c_time[:,1]==tii)[0]
                    if len(index_1)>0 and len(index_2)>0 and len(index_1)+len(index_2)==3:
                        pass
                    else:
                        print('time wrong!!! end of sample z')
                        exit()
                    
                    entity_num = 0
                    entity_num_ = 0
                    entity_list = []
                    for c in range(self.c_max):
                        if self.c_time[c,0]>=0:
                            
                            if self.c_time[c,0]<=tii and self.c_time[c,1]>tii:
                                entity_num = entity_num + len(self.entity_list[c])
                                entity_num_ = entity_num_ + self.entity_num[c]
                                entity_list.extend(self.entity_list[c])
                    if entity_num == self.d_num and entity_num_==self.d_num and set(entity_list) == set([ q for q in range(self.d_num)]):
                        pass
                        # print('correct end of sample_z')
                    else:
                        print('wrong end of sample_z!!!')
                        print(entity_num,entity_num_, sorted(entity_list),'entity_num,entity_num_, sorted(entity_list) in end of sample_z')
                        exit()
            # test end

            live_index = np.where(self.c_time[:,0]>=0)[0]
            for p in live_index:
                p_set = []
                for m in self.FCPList_exist:
                    if p in m:
                        p_set.extend(m)
                p_set = set(p_set)
                # print(p_set,'p_set in sample_z')
                for q in range(self.c_max):
                    if q in p_set:
                        if self.Hawkes_b[p,q]>self.undefine_para_value:
                            pass
                        else:
                            wrong_signal = 1
                            print(p,q,self.Hawkes_b[p,q],'wrong 1 p,q in exist end of sample_z')
                    else:
                        if self.Hawkes_b[p,q]==self.undefine_para_value:
                            pass
                        else:
                            wrong_signal = 1
                            print(p,q,self.Hawkes_b[p,q],'wrong 2 p,q not in exist end of sample_z')


                
            # for m in self.FCPList_exist:
                # print(m,'self.exist end of sample_z')
            if wrong_signal == 1:
                exit()
        
        c_live = np.where(self.c_time[:,0]>=0)[0]

        for c in c_live:
            # print(self.c_time[c,:],self.T)
            if self.c_time[c,0]==0 and self.c_time[c,1]==self.T:
                pass
            elif self.c_time[c,0]!=0 and self.c_time[c,1]==self.T:
                if len(self.children[c])==0:
                    index_1 = np.where(self.c_time[:,0]==self.c_time[c,0])[0]
                    index_2 = np.where(self.c_time[:,1]==self.c_time[c,0])[0]
                    if (len(index_1)==2 and len(index_2)==1) or  (len(index_1)==1 and len(index_2)==2):
                        pass
                    else:
                        print(index_1)
                        print(index_2)
                        print('wrong structure 1')
                        exit()
                else:
                    print('wrong structure 2')
                    exit()
            elif self.c_time[c,0]==0 and self.c_time[c,1]!=self.T:
                if len(self.parents[c])==0:
                    index_1 = np.where(self.c_time[:,0]==self.c_time[c,1])[0]
                    index_2 = np.where(self.c_time[:,1]==self.c_time[c,1])[0]
                    if (len(index_1)==2 and len(index_2)==1) or  (len(index_1)==1 and len(index_2)==2):
                        pass
                    else:
                        print(index_1)
                        print(index_2)
                        print('wrong structure 3')
                        exit()
                else:
                    print('wrong structure 4')
                    exit()
            else:
                index_1 = np.where(self.c_time[:,0]==self.c_time[c,0])[0]
                index_2 = np.where(self.c_time[:,1]==self.c_time[c,0])[0]
                # print(self.parents[c],self.children[c],'self.parents[c],self.children[c]')
                if (len(index_1)==2 and len(index_2)==1) or  (len(index_1)==1 and len(index_2)==2):
                    pass
                else:
                    print(index_1)
                    print(index_2)
                    print('wrong structure 5')
                    exit()

                index_1 = np.where(self.c_time[:,0]==self.c_time[c,1])[0]
                index_2 = np.where(self.c_time[:,1]==self.c_time[c,1])[0]
                if (len(index_1)==2 and len(index_2)==1) or  (len(index_1)==1 and len(index_2)==2):
                    pass
                else:
                    print(index_1)
                    print(index_2)
                    print('wrong structure 6')
                    exit()
        '''
                
                 

            

       
                                                        
    def sample_potential_jump(self):
        
        '''
        1. for fcplist calculate the transition rate
        2. select the maximum of rates
        3. sample
        4. calculate the transition probability
        '''       
        
        # derive omega
        if len(self.FCPList_time)==len(self.FCPList_exist):
            pass
        else:
            print('wrong, len(self.FCPList_time) is not equal to len(self.FCPList_exist)')
            exit()
        
        self.trans_matrix = np.zeros((len(self.FCPList_time),self.c_max+1,self.c_max+1))
        rate_array = np.zeros(len(self.FCPList_exist))
        
        
        for i in range(len(self.FCPList_exist)):
            rate_i_array = np.zeros(len(self.FCPList_exist[i])+1)
            s = 0 
            for j in self.FCPList_exist[i]:
                rate_i_array[s] = self.nu/self.entity_num[j]
                s = s+1
            rate_i_array[-1] = self.xi * len(self.FCPList_exist[i])
            rate_array[i] = np.amax(rate_i_array)
        # print(rate_array,'rate_array in sample_potential_jump')
        rate_array = self.coefficient_omgea * rate_array  
           
        # print(rate_array,'rate_array in sample_potential_jump')
           
        
        
        for t in range(len(self.FCPList_time)):
            omega = rate_array[t]
            # print(omega,'omega in sample_potential_jump')
            for m in self.FCPList_exist[t]:
                self.trans_matrix[t,m,m] =  1 - self.nu/self.entity_num[m]/omega # on-diagonal transition probability
                # self.trans_matrix[self.c_max,self.c_max] =  1 - self.xi * len(self.FCPList_exist[i])/omega 
                if self.entity_num[m]==0:
                    print('entity_num equals to zero in sample_potential jump',m)
                    exit()
                
                self.trans_matrix[t,m,self.c_max] = self.nu/self.entity_num[m]/omega # off-diagonal transition probability
                self.trans_matrix[t,self.c_max,m] = self.xi/omega 
            # print(np.sum(self.trans_matrix[t,self.c_max,:]),'np.sum(self.trans_matrix[t,self.c_max,:]) in sample_potential_jump')
            
                    
        # derive potential jump time
        
        t = 0
        potential_jump_time = []
        s = 1

        omega = rate_array[0]

        while t<self.T:

            u = npr.uniform()
            tau = -np.log(u)/omega
            t = t + tau

            if t<self.T:

                index = np.searchsorted(self.FCPList_time,t,side = 'right')-1
                omega_new = rate_array[index]

                u = npr.uniform()
                if u < omega_new/omega:
                    potential_jump_time.append(t)
                omega = omega_new
            else:
                break

        # print(potential_jump_time,'potential_jump_time in sample potential')
        return potential_jump_time

    
    def derive_jump_array(self, virtual_jump_time, prev_jump_time):
        
        # update FCPList_exist, FCPList_time, self.trans_matrix
        
        
        ori_trans_matrix = np.copy(self.trans_matrix)
        
        potential_time = virtual_jump_time+prev_jump_time
        
        self.FCPList_time = sorted(self.FCPList_time.tolist()+potential_time)
        self.trans_matrix = np.zeros((len(self.FCPList_time),self.c_max+1,self.c_max+1))

        jump_array = np.zeros(len(self.FCPList_time))
        
        jump_index = []
        for t in potential_time:
            jump_index.append(self.FCPList_time.index(t))
            
        jump_array[jump_index] = 1
        
        ori_index = set(np.arange(len(self.FCPList_time),dtype = int))-set(jump_index)
        ori_index = sorted(ori_index)
        ori_index = (list(ori_index))
                
        self.trans_matrix[ori_index,:,:] = np.copy(ori_trans_matrix[:,:,:])
        for t in range(len(self.FCPList_time)):
            if t in ori_index:
                pass
            else:
                self.trans_matrix[t,:,:] = np.copy(self.trans_matrix[t-1,:,:])
        
        
        new_FCPList_exist = [[] for i in range(len(self.FCPList_time))]
        new_FCPList_appear = [[] for i in range(len(self.FCPList_time))]
        new_FCPList_disappear = [[] for i in range(len(self.FCPList_time))]
        new_FCPList_remain = [[] for i in range(len(self.FCPList_time))]

        for i in range(len(ori_index)):
            new_FCPList_exist[int(ori_index[i])] = self.FCPList_exist[i]
            new_FCPList_appear[int(ori_index[i])] = self.FCPList_appear[i]
            new_FCPList_disappear[int(ori_index[i])] = self.FCPList_disppear[i]
            new_FCPList_remain[int(ori_index[i])] = self.FCPList_remain[i]
        
        self.FCPList_exist = new_FCPList_exist
        self.FCPList_appear = new_FCPList_appear
        self.FCPList_disppear = new_FCPList_disappear
        self.FCPList_remain = new_FCPList_remain
            
        for i in range(len(self.FCPList_exist)):
            if self.FCPList_exist[i]==[]:
                self.FCPList_exist[i] = copy.copy(self.FCPList_exist[i-1])
        return jump_array
    
    
    def remove_hawkes_parameter(self,community_index):
        
        self.Hawkes_b[community_index,:] = self.undefine_para_value 
        self.Hawkes_b[:,community_index] = self.undefine_para_value
        self.Hawkes_eta[community_index,:] = self.undefine_para_value
        self.Hawkes_eta[:,community_index] = self.undefine_para_value
        self.Hawkes_zeta[community_index,:] = self.undefine_para_value
        self.Hawkes_zeta[:,community_index] = self.undefine_para_value
                
    
    def sample_hawkes(self, para_select, c_send, c_receive):
        
        # not consider bound yet!!!
        
        # detemine the community location
        new_b = 0
        new_eta = 0
        new_zeta = 0

        mu = 0
        sigma = 1


        if self.c_time[c_send,0]==0 and self.c_time[c_receive,0]==0:
            if para_select == 0:
                mu = self.Hawkes_b[c_send,c_receive]
                new_b = npr.normal(mu, self.b_prior_sigma)
            if para_select == 1:
                mu = self.Hawkes_eta[c_send,c_receive]
                new_eta = npr.normal(mu, self.eta_prior_sigma)
            if para_select == 2:
                mu = self.Hawkes_zeta[c_send,c_receive]
                new_zeta = npr.normal(mu, self.zeta_prior_sigma)

        else:
            if self.c_time[c_send,0]>self.c_time[c_receive,0]:
                p = np.asarray(self.parents[c_send],dtype=int)
                if para_select == 0:
                    mu = np.sum(self.Hawkes_b[p,c_receive])/len(p)
                    new_b = npr.normal(mu, self.b_prior_sigma)
                if para_select == 1:
                    mu = np.sum(self.Hawkes_eta[p,c_receive])/len(p)
                    new_eta = npr.normal(mu, self.eta_prior_sigma)
                if para_select == 2:
                    mu = np.sum(self.Hawkes_zeta[p,c_receive])/len(p)
                    new_zeta = npr.normal(mu, self.zeta_prior_sigma)                
                        
            elif self.c_time[c_send,0]<self.c_time[c_receive,0]:
                p = np.asarray(self.parents[c_receive],dtype=int)
                if para_select == 0:
                    mu = np.sum(self.Hawkes_b[c_send,p])/len(p)
                    new_b = npr.normal(mu, self.b_prior_sigma)
                if para_select == 1:
                    mu = np.sum(self.Hawkes_eta[c_send,p])/len(p)
                    new_eta = npr.normal(mu, self.eta_prior_sigma)
                if para_select == 2:
                    mu = np.sum(self.Hawkes_zeta[c_send,p])/len(p)
                    new_zeta = npr.normal(mu, self.zeta_prior_sigma)     
            
            else: 
                p = np.asarray(self.parents[c_send],dtype=int)  
                if para_select == 0:
                    mu = np.sum(self.Hawkes_b[p,p])/len(p)
                    new_b = npr.normal(mu, self.b_prior_sigma)
                if para_select == 1:
                    mu = np.sum(self.Hawkes_eta[p,p])/len(p)
                    new_eta = npr.normal(mu, self.eta_prior_sigma)
                if para_select == 2:
                    mu = np.sum(self.Hawkes_zeta[p,p])/len(p)
                    new_zeta = npr.normal(mu, self.zeta_prior_sigma)            
        
        new_b = np.clip(new_b,self.min,self.max)
        new_eta = np.clip(new_eta,self.min,self.max)
        new_zeta = np.clip(new_zeta,self.min,self.max)

        loglike = 0
        loglike_ = 0

        for i in self.entity_list[c_send]:
            for j in self.entity_list[c_receive]:

                if i!=j:

                    s_array, r_array, time = self.path_for_hawkes_scaling(self.path_entity[i],self.path_entity[j])
        
                    x_ij = self.observation_list[i][j]
                    x_ji = self.observation_list[j][i]

                    index_ij = np.searchsorted(time, x_ij, side = 'right')-1
                    index_ji = np.searchsorted(time, x_ji, side = 'right')-1

                    z_ij_1 = s_array[index_ij]
                    z_ij_2 = r_array[index_ij]

                    z_ji_1 = s_array[index_ji]
                    z_ji_2 = r_array[index_ji]

                    b = self.Hawkes_b[z_ij_1,z_ij_2]
                    eta = self.Hawkes_eta[z_ji_1,z_ji_2]
                    zeta = self.Hawkes_zeta[z_ji_1,z_ji_2]
                    b_base = self.Hawkes_b[s_array, r_array]

                    b_ = self.b * logit(b)
                    eta_ = self.eta * logit(eta)
                    zeta_ = self.zeta * logit(zeta)
                    b_base_ = self.b * logit(b_base)

                    s = self.cal_hawkes_likelihood_(time, b_, eta_, zeta_, b_base_, x_ij, x_ji)
                    # print(s,'s')
                    loglike = loglike + s

                    if para_select == 0:

                        for m in range(len(index_ij)):
                            if z_ij_1[m] == c_send and z_ij_2[m] == c_receive:
                                b_[m] = self.b * logit(new_b)

                        for m in range(len(r_array)):
                            if r_array[m] == c_send and s_array[m] == c_receive:
                                b_base_[m] = self.b * logit(new_b)
                    
                    else:

                        for m in range(len(index_ji)):
                            if z_ji_1[m] == c_send and z_ji_2[m] == c_receive:
                                if para_select == 1:
                                    eta_[m] = self.eta * logit(new_eta)
                                if para_select == 2:
                                    zeta_[m] = self.zeta * logit(new_zeta)

                    s = self.cal_hawkes_likelihood_(time, b_, eta_, zeta_, b_base_, x_ij, x_ji)

                    loglike_ = loglike_ + s
            
        ll = 0

        if para_select == 0:
            
            q_ = -(new_b-mu)**2/(2*self.b_prior_sigma)
            q = -(self.Hawkes_b[c_send,c_receive]-mu)**2/(2*self.b_prior_sigma)

            new_sample,ll = MH(loglike_, loglike, q_, q, new_b, self.Hawkes_b[c_send,c_receive])
            self.Hawkes_b[c_send,c_receive] = new_sample

        if para_select == 1:
            
            q_ = -(new_eta-mu)**2/(2*self.eta_prior_sigma)
            q = -(self.Hawkes_eta[c_send,c_receive]-mu)**2/(2*self.eta_prior_sigma)

            new_sample,ll = MH(loglike_, loglike, q_, q, new_eta, self.Hawkes_eta[c_send,c_receive])
            self.Hawkes_eta[c_send,c_receive] = new_sample

        if para_select == 2:
            
            q_ = -(new_zeta-mu)**2/(2*self.zeta_prior_sigma)
            q = -(self.Hawkes_zeta[c_send,c_receive]-mu)**2/(2*self.zeta_prior_sigma)

            new_sample,ll = MH(loglike_, loglike, q_, q, new_zeta, self.Hawkes_zeta[c_send,c_receive])
            self.Hawkes_zeta[c_send,c_receive] = new_sample

        # print(ll,'ll in sample_hawkes')
        return ll


    def get_fcp_from_c_time(self): # checked

        # function: correct the fcp, also correct the self.entity_number
        
        self.FCPList_exist = []
        self.FCPList_time = []
        self.FCPList_appear = []
        self.FCPList_disppear = []
        self.FCPList_remain = []
        
        self.children = [[] for i in range(self.c_max)]
        self.parents = [[] for i in range(self.c_max)]
        self.sliblings_depart =  np.zeros(self.c_max,dtype = int)+self.c_max
        self.sliblings_arrive = np.zeros(self.c_max,dtype = int)+self.c_max
        
        fcplist_time = np.unique(self.c_time[:,0])
        fcplist_time = np.sort(fcplist_time, kind = 'mergesort')
        if fcplist_time[0] == -1:
            self.FCPList_time = np.delete(fcplist_time, 0) # delete the first item, the value of first item is inactive
        else: 
            print('wrong!!!')
            exit()
        exist = np.where(self.c_time[:,0]==0)[0]

        for n in exist:
            if self.entity_num[n]==0:
                print(n, 'wrong self.entity_num be 0 in get fcp')
                exit()

        # print(exist.tolist(),'exist.tolist() in get fcp')
        self.FCPList_exist.append(exist.tolist())
        self.FCPList_appear.append(exist.tolist())
        self.FCPList_disppear.append([])
        self.FCPList_remain.append([])
        

        for t in range(1,len(self.FCPList_time)):
            c_appear = np.where(self.c_time[:,0] == self.FCPList_time[t])[0]
            c_disappear = np.where(self.c_time[:,1] == self.FCPList_time[t])[0]
            
            self.FCPList_appear.append(c_appear.tolist())
            self.FCPList_disppear.append(c_disappear.tolist())
            self.FCPList_remain.append(list(set(self.FCPList_exist[t-1])-set(c_disappear.tolist())))
            self.FCPList_exist.append(list(set(self.FCPList_remain[t])|set(c_appear.tolist())))
            # print(self.FCPList_exist[t],t,'self.FCPList_exist[t],t in get fcp')
            
            for p in c_disappear:
                for c in c_appear:
                    self.children[p].append(c)
                    self.parents[c].append(p)
            
            for c in c_appear:
                if len(c_appear)<3 and len(c_appear)>0:
                    s = 0 
                    for slibling in c_appear:
                        if c != slibling:
                            s = s + 1
                            if s == 2:
                                print('wrong slibling number in appear')
                                exit()
                            self.sliblings_depart[c]=slibling
                else:
                    print('wrong appear get fcp')
                    exit()
                        
            for c in c_disappear:
                if len(c_disappear)<3 and len(c_disappear)>0:
                    s = 0 
                    for slibling in c_disappear:
                        if c != slibling:
                            s = s +1
                            if s == 2:
                                print('wrong slibling number in disappear')
                                exit()
                            self.sliblings_arrive[c]=slibling
                else:
                    print('wrong disappear get fcp')
                    exit()

        for i in range(self.c_max):
            if self.c_time[i,0]>=0:
                self.entity_num[i] = len(self.entity_list[i])
            else:
                self.entity_num[i] = 0
                self.entity_list[i] = []

        dead_c = np.where(self.c_time[:,0]<0)[0]
        self.remove_hawkes_parameter(dead_c)

        live_index = np.where(self.c_time[:,0]>=0)[0]
        
        for p in live_index:
            p_set = []
            for m in self.FCPList_exist:
                if p in m:
                    p_set.extend(m)
            p_set = set(p_set)
            for q in live_index:
                if q not in p_set:
                    # print(p,q,'p,q remove')
                    self.remove_parameter(p,q)
   
                
    def preprocess_entity_path_i(self, i): # 
        
        
        # function: correct c_time
        # at time 0, if c is an empty community, then set prev_time be []
        
        prev_time = [] # in the prev_time, [] means no jump time.
        prev_jump = [] # community index

        path_i = copy.copy(self.path_entity[i])

        entity_num_array_ = np.zeros(len(path_i))
        for t in range(len(path_i)):
            entity_num_array_[t] = self.entity_num[path_i[t]]+0

        for j in self.path_entity[i]:
            self.entity_list[j].remove(i) # remove i in entity list
            self.entity_num[j] = self.entity_num[j]-1 # entity number - 1
        
        ori_path_i = copy.copy(path_i)
        ori_path_i = np.asarray(ori_path_i, dtype=int)

        path_time_begin = self.c_time[ori_path_i,0]
        path_time_end = self.c_time[ori_path_i,1]


        ori_path_i[np.where(self.entity_num[ori_path_i]==0)[0]] = -1 # record the empty community in path_i be -1

        c = path_i[0]
        
        if len(self.path_entity[i])!=1:
            
            if self.entity_num[c] == 0:
                    
                prev_time.append([])
                prev_jump.append(-1)

                self.set_slibling_arrive(path_i, 0, prev_time, self.sliblings_arrive[c],prev_jump,path_time_begin,path_time_end)
                self.transfer_parameter_before_sampling_arrive(self.path_entity[i][1], path_i[1], path_i[0])
                
            else:

                prev_time.append([])
                prev_jump.append(c)
                
            for t in range(1,len(self.path_entity[i])-1):
                
                c = path_i[t]
                if self.entity_num[c] == 0: 
                    
                    self.set_slibling_depart(path_i, t, prev_time, self.sliblings_depart[c],self.sliblings_arrive[c], prev_jump, path_time_begin, path_time_end)
                    self.transfer_parameter_before_sampling_depart(self.sliblings_depart[c], path_i[t-1], path_i[t])
                    
                    if self.sliblings_arrive[c]!=self.c_max:
                        self.set_slibling_arrive(path_i, t, prev_time, self.sliblings_arrive[c],prev_jump,path_time_begin,path_time_end)
                        self.transfer_parameter_before_sampling_arrive(self.path_entity[i][t+1], path_i[t+1], path_i[t])
                else: # check if necessary by checking set_slibling_arrive
                    if ori_path_i[t-1] == -1:
                        pass
                    else:
                        prev_time.append([])
                        prev_jump.append(c)
            
            c = path_i[-1]
            if self.entity_num[c] == 0: 
                if self.sliblings_depart[c]!=self.c_max:
                    self.set_slibling_depart(path_i, len(path_i)-1, prev_time, self.sliblings_depart[c], self.sliblings_arrive[c], prev_jump, path_time_begin, path_time_end)
                    self.transfer_parameter_before_sampling_depart(self.sliblings_depart[c], path_i[len(path_i)-2], path_i[-1])
                    
                else:
                    prev_time.append()
                    prev_jump.append(-1)
            else:
                if ori_path_i[len(path_i)-2] == -1:

                    pass
                else:
                    prev_time.append([])
                    prev_jump.append(c)
                
        else:
            prev_time.append([])
            if self.entity_num[c] == 0:
                self.c_time[c,0] = -1
                self.c_time[c,1] = self.T
                prev_jump.append(-1)
            else:
                prev_jump.append(c)
                
        return prev_time
                
                   
    def set_slibling_arrive(self, path_i, t, prev_time, arrive, prev_jump, path_time_begin, path_time_end): # 1 set slibling 2 set c_time
        
        prev_time.append(path_time_end[t]) # 1. satisfy when time = 0    2. check satisfy when time > 0
        
        self.c_time[arrive,1] = path_time_end[t+1] # set slibling end time
        
        self.c_time[path_i[t+1],0] = -1
        self.c_time[path_i[t+1],1] = self.T 
        
        self.c_time[path_i[t],0] = -1
        self.c_time[path_i[t],1] = self.T 
        
        path_i[t+1] = arrive
        
        prev_jump.append(arrive)
        
        
    def set_slibling_depart(self,path_i, t, prev_time, depart, arrive, prev_jump, path_time_begin, path_time_end): # 1 set slibling 2 set c_time 3 modify hawkes parameter
                
        prev_time.append(path_time_begin[t])
        prev_jump.append(-1)
        
        self.c_time[path_i[t-1],1] = self.c_time[depart,1] # set previous path end time
        
        self.c_time[depart,0] = -1
        self.c_time[depart,1] = self.T
        
        self.c_time[path_i[t],0] = -1
        self.c_time[path_i[t],1] = self.T 
        
        if depart == arrive:
            self.sliblings_arrive[path_i[t]] = path_i[t-1]

        if self.sliblings_arrive[depart]!=self.c_max:
            self.sliblings_arrive[path_i[t-1]] = self.sliblings_arrive[depart]
            self.sliblings_arrive[self.sliblings_arrive[depart]] = path_i[t-1]
            
            
    def process_entity_paths(self):
        
        root = np.where(self.c_time[:,0]==0)[0]
        path = []
        entity_list = []
        self.path_entity = [[] for i in range(self.d_num)]
        
        for c in root:
            self.iter_community_for_entity_paths(c, path, entity_list)
        
    
    def iter_community_for_entity_paths(self, node, path, entity_list):

        # should check carefully!!! not done yet!!!
        
        path.append(node)
        entity_list.append(self.entity_list[node])
        
        if self.children[node]==[]:
            
            entity_set = set(entity_list[0])
            for s in entity_list:
                entity_set = entity_set.intersection(set(s))
            
            entity_set = list(entity_set)
            for i in entity_set:
                self.path_entity[i].extend(copy.copy(path))
             
            del path[-1]
            del entity_list[-1]
        
        else:
            
            for i in self.children[node]:    
                self.iter_community_for_entity_paths(i, path, entity_list)
                
            del path[-1]
            del entity_list[-1]
        
        
    def extend_all_paths(self, i): # i is the entity for sampling!!!
        
        # extend all entity path based on the extended self.FCPList_time

        count_c_num = np.zeros(self.c_max,dtype=int)

        live_c_index = np.where(self.c_time[:,0]>-1)[0]

        for c in live_c_index:
            for t in self.FCPList_time:
                if t>=self.c_time[c,0] and t<self.c_time[c,1]:
                    count_c_num[c] = count_c_num[c]+1

        path_entity = copy.copy(self.path_entity)

        self.path_entity = [[] for n in range(self.d_num)] # community path for each entity

        for j in range(self.d_num):
            if i!=j:
                for c in path_entity[j]:
                    for n in range(count_c_num[int(c)]):
                        self.path_entity[j].append(int(c))

                    
    def transfer_parameter_before_sampling_depart(self, f, t, slibling):  
        

        para_f_index = np.where(self.Hawkes_b[f,:]>self.undefine_para_value)[0]
        # print(para_f_index,'para_f_index in transfer_parameter_before_sampling_depart')


        para_f_index = para_f_index.tolist()
        # print(f,t,slibling, 'f,t,slibling in transfer_parameter_before_sampling_depart')
        # print(para_f_index,'para_f_index in transfer_parameter_before_sampling_depart')

        para_f_index.remove(slibling)
        
        para_t_index = np.where(self.Hawkes_b[t]>self.undefine_para_value)[0]
        para_t_index = para_t_index.tolist()    
        
        diff = set(para_f_index)-set(para_t_index)
        diff = np.asarray(list(diff))
        
        if diff!=[]:
            self.copy_parameter_sampling(f, t, diff)
        
        self.remove_parameter(np.arange(self.c_max), slibling)
        self.remove_parameter(np.arange(self.c_max), f)
        
        
    def transfer_parameter_before_sampling_arrive(self, f, t, slibling): 
        # print(f,t,slibling, 'f,t,slibling in transfer_parameter_before_sampling_arrive')

        para_f_index = np.where(self.Hawkes_b[f,:]>self.undefine_para_value)[0]
        # print(para_f_index, 'para_f_index transfer_parameter_before_sampling_arrive')
        para_f_index = para_f_index.tolist()
        
        para_t_index = np.where(self.Hawkes_b[t]>self.undefine_para_value)[0]
        para_t_index = para_t_index.tolist()    
        
        diff = set(para_f_index)-set(para_t_index)
        diff = np.asarray(list(diff))
        
        if diff!=[]:
            self.copy_parameter_sampling(f, t, diff)
        
        self.remove_parameter(np.arange(self.c_max), slibling)
        self.remove_parameter(np.arange(self.c_max), f)
        
        
    def copy_parameter_sampling(self, f, t, diff): # checked
        # print(f,t,diff,self.Hawkes_b[f,diff],'f,t,diff,self.Hawkes_b[f,diff] in copy parameter sampling')
        self.Hawkes_b[t,diff] = np.copy(self.Hawkes_b[f,diff])
        self.Hawkes_b[diff,t] = np.copy(self.Hawkes_b[diff,f])
        self.Hawkes_eta[t,diff] = np.copy(self.Hawkes_eta[f,diff])
        self.Hawkes_eta[diff,t] = np.copy(self.Hawkes_eta[diff,f])
        self.Hawkes_zeta[t,diff] = np.copy(self.Hawkes_zeta[f,diff])
        self.Hawkes_zeta[diff,t] = np.copy(self.Hawkes_zeta[diff,f])
                
    
    def remove_parameter(self, index, c): # checked
        
        self.Hawkes_b[c,index] = self.undefine_para_value
        self.Hawkes_b[index,c] = self.undefine_para_value
        self.Hawkes_eta[c,index] = self.undefine_para_value
        self.Hawkes_eta[index,c] = self.undefine_para_value
        self.Hawkes_zeta[c,index] = self.undefine_para_value
        self.Hawkes_zeta[index,c] = self.undefine_para_value
            

    def smc_weight(self, t_begin_index, t_end_index, i, path_i, empty_para_array,d):  
        
        t_end_index = t_end_index+1
        time = copy.copy(self.FCPList_time)
        time.append(self.T)
        
        loglike = 0
        # print(t_begin_index, t_end_index,'t_begin_index, t_end_index')
        # print(path_i,'path_i in smc_weight')
        # print(t_begin_index,t_end_index,'t_begin_index,t_end_index smc_weight')
        for j in range(self.d_num):
            if j!=i:
                # print(j,'j in smc_weight')
                b, b_base, eta_ji_prior, zeta_ji_prior, eta_ji_like, zeta_ji_like, x_ij, x_ji_prior, x_ji_like = \
                self.derive_info_hawkes_smc(i, j, time, t_begin_index, t_end_index, path_i[0:t_end_index], self.path_entity[j][0:t_end_index], empty_para_array,d)
                
                loglike = loglike + self.calculate_hawkes_smc(time, t_begin_index, t_end_index, b, 
                                          eta_ji_prior, zeta_ji_prior, eta_ji_like, zeta_ji_like, x_ij, x_ji_prior, x_ji_like, empty_para_array, b_base)
                
                b, b_base, eta_ji_prior, zeta_ji_prior, eta_ji_like, zeta_ji_like, x_ij, x_ji_prior, x_ji_like = \
                self.derive_info_hawkes_smc(j, i, time, t_begin_index, t_end_index, self.path_entity[j][0:t_end_index], path_i[0:t_end_index],empty_para_array,d)
                
                loglike = loglike + self.calculate_hawkes_smc(time, t_begin_index, t_end_index, b, 
                                          eta_ji_prior, zeta_ji_prior, eta_ji_like, zeta_ji_like, x_ij, x_ji_prior, x_ji_like,empty_para_array, b_base)

        return loglike


    def derive_info_hawkes_smc(self, s, r, time, t_begin_index, t_end_index, path_s, path_r, empty_para_array,d):
        
        path_r = np.asarray(path_r, dtype = int)
        path_s = np.asarray(path_s, dtype = int)


        x_ij, x_ij_all, x_ij_like, x_ij_prior = derive_ob_info(self.observation_list[s][r], time, t_begin_index, t_end_index)
        x_ji, x_ji_all, x_ji_like, x_ji_prior = derive_ob_info(self.observation_list[r][s], time, t_begin_index, t_end_index)
                
        b = np.zeros(len(path_s))
        eta = np.zeros(len(path_s))
        zeta = np.zeros(len(path_s))
        
        # print(path_s, path_r,'path_s, path_r in derive_info_hawkes_smc')
        # print(t_begin_index,t_end_index,'t_begin_index,t_end_index in derive_info_hawkes_smc')
        # for m in self.FCPList_exist:
            # print(m, 'self.fcp_exist in derive_info_hawkes_smc')
        for t in range(len(path_s)):
            if path_s[t] == self.c_max and path_r[t] == self.c_max:

                if empty_para_array[t,self.c_max,0,0,d] == self.undefine_para_value:
                    print('wrong derive_info_hawkes_smc',t,'t1')
                    exit()

            elif path_s[t] == self.c_max:
                
                if empty_para_array[t,path_r[t],0,0,d] == self.undefine_para_value:
                    print('wrong derive_info_hawkes_smc',t,'t2',path_s[t],'path_s[t]',path_r[t],'path_r[t]')
                    exit()
                b[t] = empty_para_array[t,path_r[t],1,0,d]
                eta[t] = empty_para_array[t,path_r[t],1,1,d]
                zeta[t] = empty_para_array[t,path_r[t],1,2,d]

            elif path_r[t] == self.c_max:
                
                if empty_para_array[t,path_s[t],0,0,d] == self.undefine_para_value:
                    print('wrong derive_info_hawkes_smc',t,'t3')
                    exit()
                b[t] = empty_para_array[t,path_s[t],0,0,d]
                eta[t] = empty_para_array[t,path_s[t],0,1,d]
                zeta[t] = empty_para_array[t,path_s[t],0,2,d]

            else:

                if self.Hawkes_b[path_s[t],path_r[t]] == self.undefine_para_value:
                    print('wrong derive_info_hawkes_smc',t,'t4')
                    exit()
                b[t] = self.Hawkes_b[path_s[t],path_r[t]]
                eta[t] = self.Hawkes_eta[path_s[t],path_r[t]]
                zeta[t] = self.Hawkes_zeta[path_s[t],path_r[t]]

        b = self.b*logit(b)
        eta = self.eta*logit(eta)
        zeta = self.zeta*logit(zeta)
        
        z_ij_all_index = np.searchsorted(time,x_ij_all,side='right')-1

        b_all = b[z_ij_all_index]

        z_ji_prior_index = np.searchsorted(time,x_ji_prior,side = 'right')-1        
        z_ji_like_index = np.searchsorted(time,x_ji_like,side = 'right')-1

        eta_ji_prior = eta[z_ji_prior_index]
        zeta_ji_prior = zeta[z_ji_prior_index]
        
        eta_ji_like = eta[z_ji_like_index]
        zeta_ji_like = zeta[z_ji_like_index]
    
        return b_all, b, eta_ji_prior, zeta_ji_prior, eta_ji_like, zeta_ji_like, x_ij_like, x_ji_prior, x_ji_like
        
        
    def calculate_hawkes_smc(self, jump_time, t_begin_index, t_end_index, b, eta_ji_prior, 
    zeta_ji_prior, eta_ji_like, zeta_ji_like, x_ij, x_ji_prior, x_ji_like, empty_para_array, 
    b_base): 
        
        # smc_flag = 1 then the function works for smc, otherwise the function works for normal

        s = 0
               
        # calculate b likelihood in second part
        
        for t in range(t_begin_index,t_end_index):
            
            t_b = jump_time[t]
            t_e = jump_time[t+1]
            b_t = b_base[t]
            s = s-b_t*(t_e-t_b) 
            
        # calculate kernel likelihood in second part
        
    
            
        # here t_begin is larger than x_ji_prior
        t_begin = jump_time[t_begin_index]
        t_end = jump_time[t_end_index]
        if eta_ji_like!=[]:
            s = s-np.dot(eta_ji_like/zeta_ji_like, 1-np.exp(-zeta_ji_like*(t_end-x_ji_like))) 
        if eta_ji_prior!=[]:
            s = s-np.dot(eta_ji_prior/zeta_ji_prior,np.exp(-zeta_ji_prior*(t_begin-x_ji_prior))-
            np.exp(-zeta_ji_prior*(t_end-x_ji_prior)))

        
        # calculate the first part
        
        for t in range(len(x_ij)):

            l = b[t]
            if eta_ji_prior!=[]:
                l = l + np.dot(eta_ji_prior,np.exp(-zeta_ji_prior*(x_ij[t]-x_ji_prior)))
            index = np.where(x_ji_like<x_ij[t])[0]
            if index!=[]:
                l = l + np.dot(eta_ji_like[index],np.exp(-zeta_ji_like[index]*(x_ij[t]-x_ji_like[index])))
            s = s + np.log(l)
        
        return s 
                

    def smc_z(self, i, jump_array): 

        # set jump_array[0] == 1
        jump_array[0] = 1
        jump_array = np.asarray(jump_array)

        empty_para_array = np.zeros((len(self.FCPList_time),self.c_max+1, 2, 3, self.particle_num))+self.undefine_para_value
        particle_path = np.zeros((self.particle_num,len(self.FCPList_time)),dtype = int) - 1       
        
        jump_array_index = np.where(jump_array==1)[0]

        if len(jump_array) == len(self.FCPList_time):
            pass
        else: 
            print('wrong', len(jump_array), len(self.FCPList_time), 'in smc_z')

        for t in range(len(jump_array_index)):
            # print(t,'t in smc_z')
            
            if t == len(jump_array_index)-1:
                iter_len = len(jump_array) - jump_array_index[-1]-1
                
            else:
                iter_len = jump_array_index[t+1] - jump_array_index[t] - 1
        
            if t==0:

                root = np.where(self.c_time[:,0]==0)[0]
                root = np.append(root, self.c_max)
                total_probability = []
                total_path = []
                
                for node in root:

                    tran_probability = []
                    path = []
                    p = 0
                    if node != self.c_max:
                        p = self.entity_num[node]/(self.d_num-1+self.nu/self.xi)
                    else:
                        p = self.nu/self.xi/(self.d_num-1+self.nu/self.xi)

                    self.iter_path_smc(node, path, total_path, iter_len, tran_probability, 0, 0, total_probability,p)
                    
                particle_hawkes = np.zeros(self.particle_num)
                # print(total_probability,'total_probability in smc_z 1')
                # print(np.sum(total_probability),'np.sum(total_probability) in smc_z1')
                if np.abs(np.sum(total_probability)-1)>10**(-7):
                    print(np.sum(total_probability),'np.sum(total_probability)!=1 smc_z1')
                    exit()
                for d in range(self.particle_num):

                    index = np.argmax(npr.multinomial(1, total_probability/np.sum(total_probability)))
                    
                    # print(total_path[index], 'total path[index] smcz with t=0')
                    particle_path[d, jump_array_index[t]:(jump_array_index[t]+iter_len+1)] = np.copy(total_path[index]) 

                    # print(jump_array_index[t],(jump_array_index[t]+iter_len+1),'jump_array_index[t]:(jump_array_index[t]+iter_len+1) in smc_z')
                    self.sample_empty_parameter_smc(particle_path,empty_para_array,jump_array_index[t],jump_array_index[t]+iter_len+1,d,jump_array)
                    # print(np.where(empty_para_array[:,:,0,0,d]>self.undefine_para_value),'np.where(empty_para_array[:,:,0,0,d]>self.undefine_para_value) at t = 0 in smc_z')
                    # print(iter_len,'iter_len in smc_z')
                    # print(jump_array,'jump_array in smc_z',len(jump_array),'len(jump_array)s')
                    
                    particle_hawkes[d] = self.smc_weight(jump_array_index[t],jump_array_index[t]+iter_len, i, 
                    particle_path[d,:], empty_para_array, d)
                    # print(particle_hawkes[d],d,'particle_hawkes[d],d in smc_z')

                new_particle_path = np.copy(particle_path)
                particle_path[:,:] = -1
                new_empty_para_array = np.copy(empty_para_array)
                normalize_weight(particle_hawkes)
                # print(particle_hawkes,'particle_hawkes in smcz')
                # print(new_particle_path,'new_particle_path smc_z')
                for d in range(self.particle_num):
                    index = np.argmax(npr.multinomial(1, particle_hawkes))
                    particle_path[d,:] = np.copy(new_particle_path[index,:])
                    
                    # print(particle_hawkes,'particle_hawkes')
                    # print(particle_path[d,:],'particle_path[d,:] at t=0')
                    empty_para_array[:,:,:,:,d] = np.copy(new_empty_para_array[:,:,:,:,index])
                    for tl in range(len(self.FCPList_time)):
                        for cl in range(self.c_max):
                            if empty_para_array[tl,cl,0,0,d]>self.undefine_para_value:
                                pass
                                # print(tl,cl,'t,c in empty_para_array in smc_z')
                

            else:

                for d in range(self.particle_num):

                    total_probability = []
                    total_path = []
                
                    previous_node = particle_path[d,jump_array_index[t]-1]
                    node_array = []

                    if previous_node == self.c_max: 
                        node_array.extend(self.FCPList_exist[jump_array_index[t]-1])
                        node_array.append(self.c_max)
                    else: 
                        node_array.append(previous_node)
                        node_array.append(self.c_max)
                    # print(previous_node,'previous_node in smc_z')
                    # print(node_array,'node_array in smc_z')
                    for node in node_array:
                        tran_probability = []
                        p = 0
                        if previous_node == self.c_max and node == self.c_max:
                            live_c = np.asarray(node_array[0:-1],dtype = int)
                            p = 1 - np.sum(self.trans_matrix[jump_array_index[t],self.c_max,live_c])
                            # print(p,'p in smc_z')
                            if p == 0:
                                # print(t,'t in smc_z')
                                exit()
                        else:
                            
                            p = self.trans_matrix[jump_array_index[t],previous_node,node]+0
                        # print(node,'node',p,'p','in smc_z')
                        if p<=0 or p >=1:
                            print(p,'wrong smc_z p<=0 or p >=1')
                        self.iter_path_smc(node, path, total_path, iter_len, tran_probability,jump_array_index[t], 0,
                        total_probability,p)
                    if np.abs(np.sum(total_probability)-1)>10**(-7):
                        print(np.sum(total_probability),'np.sum(total_probability)!=1 smc_z2')
                        exit()
                    index = np.argmax(npr.multinomial(1, total_probability/np.sum(total_probability)))
                    particle_path[d, jump_array_index[t]:(jump_array_index[t]+iter_len+1)] = np.copy(np.asarray(total_path[index])) 
                # print(jump_array,'jump array smcz')
                # print(t,'t in smz_z')
                for d in range(self.particle_num):
                    
                    self.sample_empty_parameter_smc(particle_path,empty_para_array,jump_array_index[t],jump_array_index[t]+iter_len+1,d,jump_array)
                    # print(np.where(empty_para_array[:,:,0,0,d]>self.undefine_para_value),'np.where(empty_para_array[:,:,0,0,d]>self.undefine_para_value)')
                    particle_hawkes[d] = self.smc_weight(jump_array_index[t], jump_array_index[t]+iter_len, i, 
                    particle_path[d,:], empty_para_array,d)

                new_particle_path = np.copy(particle_path)
                particle_path = np.zeros((self.particle_num, len(self.FCPList_time)),dtype = int)-1
                new_empty_para_array = np.copy(empty_para_array)
                normalize_weight(particle_hawkes)
                # print(new_particle_path,'new_particle_path in smc_z 2')
                # print(particle_hawkes,'particle_hawkes')
                for d in range(self.particle_num):
                    index = np.argmax(npr.multinomial(1, particle_hawkes))

                    particle_path[d,:] = np.copy(new_particle_path[index,:])
                    # print(particle_path[d,:],'particle_path[d,:] at t=0')
                    # print(particle_path[d,:],'particle_path[d,:] in smc_z')
                    empty_para_array[:,:,:,:,d] = np.copy(new_empty_para_array[:,:,:,:,index]) 
                
                    tl,cl = np.where(empty_para_array[:,:,0,0,d]>self.undefine_para_value)
                                

                # exit()
                            
                    
        return particle_path[0,:], empty_para_array[:,:,:,:,0]


    def sample_empty_parameter_smc(self, particle_path, empty_para_array, t_begin, t_end, d, jump_array):

        # how to sample itself
        # print( t_begin, t_end, d, particle_path[d,:], ' t_begin, t_end, d, particle_path[d,:] in  sample empty')
        for t in range(t_begin, t_end):
            if particle_path[d,t] == self.c_max:
                # print(particle_path[d,t],particle_path[d,t-1],jump_array[t],particle_path[d,:],'particle_path[d,t],particle_path[d,t-1],jump_array[t],particle_path[d,:]')
                if t == 0:
                    # print(self.FCPList_exist[t],'self.FCPList_exist[t] in sample empty')
                    for c in self.FCPList_exist[t]:
                        empty_para_array[t,c,0,0,d] = np.clip(npr.normal(self.b_prior_mu,self.b_prior_sigma),self.min,self.max)
                        empty_para_array[t,c,1,0,d] = np.clip(npr.normal(self.b_prior_mu,self.b_prior_sigma),self.min,self.max)
                        empty_para_array[t,c,0,1,d] = np.clip(npr.normal(self.eta_prior_mu,self.eta_prior_sigma),self.min,self.max)
                        empty_para_array[t,c,1,1,d] = np.clip(npr.normal(self.eta_prior_mu,self.eta_prior_sigma),self.min,self.max)
                        empty_para_array[t,c,0,2,d] = np.clip(npr.normal(self.zeta_prior_mu,self.zeta_prior_sigma),self.min,self.max)
                        empty_para_array[t,c,1,2,d] = np.clip(npr.normal(self.zeta_prior_mu,self.zeta_prior_sigma),self.min,self.max)

                        '''
                        # fix eta zeta
                        empty_para_array[t,c,0,1,d] = 10
                        empty_para_array[t,c,1,1,d] = 10
                        empty_para_array[t,c,0,2,d] = 10
                        empty_para_array[t,c,1,2,d] = 10
                        '''

                    empty_para_array[t,self.c_max,0,0,d] = np.clip(npr.normal(self.b_prior_mu,self.b_prior_sigma),self.min,self.max)
                    empty_para_array[t,self.c_max,0,1,d] = np.clip(npr.normal(self.eta_prior_mu,self.eta_prior_sigma),self.min,self.max)
                    empty_para_array[t,self.c_max,0,2,d] = np.clip(npr.normal(self.zeta_prior_mu,self.zeta_prior_sigma),self.min,self.max)
                    '''
                    #fix eta
                    empty_para_array[t,self.c_max,0,1,d] = 10
                    empty_para_array[t,self.c_max,0,2,d] = 10
                    '''

                else:
                    if jump_array[t] == 1:
                        if particle_path[d,t-1] == particle_path[d,t]: # empty follow empty
                            for c in self.FCPList_exist[t]:
                                empty_para_array[t,c,0,0,d] = empty_para_array[t-1,c,0,0,d]
                                empty_para_array[t,c,1,0,d] = empty_para_array[t-1,c,1,0,d]
                                empty_para_array[t,c,0,1,d] = empty_para_array[t-1,c,0,1,d]
                                empty_para_array[t,c,1,1,d] = empty_para_array[t-1,c,1,1,d]
                                empty_para_array[t,c,0,2,d] = empty_para_array[t-1,c,0,2,d]
                                empty_para_array[t,c,1,2,d] = empty_para_array[t-1,c,1,2,d]
                            
                            
                        else: # genetate new --> split
                            ori_node = particle_path[d,t-1]
                            
                            for c in self.FCPList_exist[t]:
                                # print(ori_node,c,self.Hawkes_b[ori_node,c],'ori_node,c,self.Hawkes_b[ori_node,c]')
                                if c == ori_node:
                                    empty_para_array[t,self.c_max,0,0,d] = np.clip(npr.normal(self.Hawkes_b[c,ori_node],self.b_prior_sigma),self.min,self.max)
                                    empty_para_array[t,self.c_max,0,1,d] = np.clip(npr.normal(self.Hawkes_eta[c,ori_node],self.b_prior_sigma),self.min,self.max)
                                    empty_para_array[t,self.c_max,0,2,d] = np.clip(npr.normal(self.Hawkes_zeta[c,ori_node],self.b_prior_sigma),self.min,self.max)
                                
                                    '''
                                    # fix eta zeta

                                    empty_para_array[t,self.c_max,0,1,d] = 10
                                    empty_para_array[t,self.c_max,0,2,d] = 10
                                    '''
                                
                                # print(ori_node,c,self.Hawkes_b[ori_node,c],'ori_node,c,self.Hawkes_b[ori_node,c]')
                                # print(self.Hawkes_b[ori_node,c],c,'self.Hawkes_b[ori_node,c] in sample empty')
                                empty_para_array[t,c,0,0,d] = np.clip(npr.normal(self.Hawkes_b[c,ori_node],self.b_prior_sigma),self.min,self.max)
                                empty_para_array[t,c,1,0,d] = np.clip(npr.normal(self.Hawkes_b[ori_node,c],self.b_prior_sigma),self.min,self.max)
                                empty_para_array[t,c,0,1,d] = np.clip(npr.normal(self.Hawkes_eta[c,ori_node],self.eta_prior_sigma),self.min,self.max)
                                empty_para_array[t,c,1,1,d] = np.clip(npr.normal(self.Hawkes_eta[ori_node,c],self.eta_prior_sigma),self.min,self.max)
                                empty_para_array[t,c,0,2,d] = np.clip(npr.normal(self.Hawkes_zeta[c,ori_node],self.zeta_prior_sigma),self.min,self.max)
                                empty_para_array[t,c,1,2,d] = np.clip(npr.normal(self.Hawkes_zeta[ori_node,c],self.eta_prior_sigma),self.min,self.max)
                                '''
                                # fix eta,zeta
                                empty_para_array[t,c,0,1,d] = 10
                                empty_para_array[t,c,1,1,d] = 10
                                empty_para_array[t,c,0,2,d] = 10
                                empty_para_array[t,c,1,2,d] = 10
                                '''

                    else: # empty follow empty, but new community appears

                        appear = set(self.FCPList_exist[t])-set(self.FCPList_exist[t-1])
                        remain = set(self.FCPList_exist[t])-appear
                        disappear = set(self.FCPList_exist[t-1])-set(self.FCPList_exist[t])
                        # print(remain,appear,'remain,appear in sample empty parameter smc')
                        remain = list(remain)
                        appear = list(appear)
                        disappear = list(disappear)
                        disappear = np.asarray(disappear)

                        for c in remain:
                            # print(empty_para_array[t-1,c,0,0,d],c,'empty_para_array[t-1,c,0,0,d],c, in sample empty')
                            empty_para_array[t,c,0,0,d] = empty_para_array[t-1,c,0,0,d]
                            empty_para_array[t,c,1,0,d] = empty_para_array[t-1,c,1,0,d]
                            empty_para_array[t,c,0,1,d] = empty_para_array[t-1,c,0,1,d]
                            empty_para_array[t,c,1,1,d] = empty_para_array[t-1,c,1,1,d]
                            empty_para_array[t,c,0,2,d] = empty_para_array[t-1,c,0,2,d]
                            empty_para_array[t,c,1,2,d] = empty_para_array[t-1,c,1,2,d]
                            
                        for c in appear: #generate new

                            empty_para_array[t,c,0,0,d] = np.clip(npr.normal(np.sum(empty_para_array[t-1,disappear,0,0,d])/len(disappear),self.b_prior_sigma),self.min,self.max)
                            empty_para_array[t,c,1,0,d] = np.clip(npr.normal(np.sum(empty_para_array[t-1,disappear,1,0,d])/len(disappear),self.b_prior_sigma),self.min,self.max)
                            empty_para_array[t,c,0,1,d] = np.clip(npr.normal(np.sum(empty_para_array[t-1,disappear,0,1,d])/len(disappear),self.eta_prior_sigma),self.min,self.max)
                            empty_para_array[t,c,1,1,d] = np.clip(npr.normal(np.sum(empty_para_array[t-1,disappear,1,1,d])/len(disappear),self.eta_prior_sigma),self.min,self.max)
                            empty_para_array[t,c,0,2,d] = np.clip(npr.normal(np.sum(empty_para_array[t-1,disappear,0,2,d])/len(disappear),self.zeta_prior_sigma),self.min,self.max)
                            empty_para_array[t,c,1,2,d] = np.clip(npr.normal(np.sum(empty_para_array[t-1,disappear,1,2,d])/len(disappear),self.zeta_prior_sigma),self.min,self.max)
                            
                            '''
                            #fix eta, zeta
                            empty_para_array[t,c,0,1,d] = 10
                            empty_para_array[t,c,1,1,d] = 10
                            empty_para_array[t,c,0,2,d] = 10
                            empty_para_array[t,c,1,2,d] = 10
                            '''

    def iter_path_smc(self, node, path, total_path, iter_len, tran_probability, t, cur_length, total_probability,p):
    
        path.append(node)
        tran_probability.append(p)
        # print(tran_probability,'tran_probability in iter_path_smc')
        if cur_length == iter_len:
            total_path.append(copy.copy(path))
            total_probability.append(np.prod(tran_probability))
            # print(total_probability,'total_probability in iter_path_smc')
            # print(path,'path in iter_path_smc')
            # print(iter_len, len(tran_probability),'tran_probability length in iter_path_smc')
            del tran_probability[-1]
            del path[-1]
            # calculate transistion
        else:
            cur_length = cur_length+1
            if node == self.c_max: # empty node 
        
                self.iter_path_smc(node,path,total_path,iter_len,tran_probability,t,cur_length,total_probability,1)
            
            else:
                terminate_node = np.where(self.c_time[:,0]==self.FCPList_time[t+cur_length])[0]
                if set(self.children[node]) == set(terminate_node):
                    if len(self.children[node])== 2: # split
                        # print('children in iter_path_smc',)
                        for i in self.children[node]:
                            # print(i,node,'i,node in iter_path_smc')
                            # print(self.entity_num[i],self.entity_num[node],'self.entity_num[i],self.entity_num[node] in iter_path_smc')
                            p = self.entity_num[i]/self.entity_num[node]
                            # print(p, 'split child p in iter_path_smc')
                            
                            
                            self.iter_path_smc(i, path, total_path, iter_len, tran_probability, t, cur_length, total_probability,p)
                    else: # merge
                        
                        node = self.children[node][0]
                        self.iter_path_smc(node, path, total_path, iter_len, tran_probability, t, cur_length, total_probability,1)

                else:
                    self.iter_path_smc(node, path, total_path, iter_len, tran_probability, t, cur_length, total_probability,1)
            del tran_probability[-1]
            del path[-1]

    


    def postprocess_after_sample_z(self, jump_array, path_i, i, empty_para_array):

        
        path_i = np.asarray(path_i,dtype = int)
        new_c_flag = 0 # 1 shows the previous is a new community 
        split_flag = 0 # split:0 merge:1

        # check path_i length !!!
        
        _path_i = copy.copy(path_i)

        unused = np.where(self.c_time[:,0]<0)[0]
        unused = unused.tolist()
        t = 0

        while t < len(self.FCPList_time):
            if t == 0:
                if path_i[0]<self.c_max: # old community
                    new_c_flag = 0
                    self.entity_list[path_i[0]].append(i)
                    self.entity_num[path_i[0]] = self.entity_num[path_i[0]]+1
                    t = t+1
                    while t<len(self.FCPList_time):
                        if _path_i[t] == _path_i[t-1]:
                            t = t+1
                        else:
                            if jump_array[t] == 1: # no old split or merge, so new split!
                                split_flag = 0
                                new_c_flag = 1
                            else: 
                                new_c_flag = 0
                            break
                else:
                    new_c_flag = 1
                    split_flag = 1
                    split_new = unused.pop()
                    self.c_time[split_new,0] = 0
                    self.entity_list[split_new].append(i)
                    self.entity_num[split_new] = 1
                    # transfer parameter normal (include itself)
                    self.Hawkes_b[split_new,split_new] = empty_para_array[t,self.c_max,0,0]
                    self.Hawkes_eta[split_new,split_new] = empty_para_array[t,self.c_max,0,1]
                    self.Hawkes_zeta[split_new,split_new] = empty_para_array[t,self.c_max,0,2]              
                    for c in self.FCPList_exist[t]:
                        self.transfer_empty_para_postprocess(t,c,split_new,empty_para_array)
                    t = t+1
                    while t<len(self.FCPList_time):
                        if _path_i[t] == self.c_max:
                            # transfer_parameter normal
                            appear = set(self.FCPList_exist[t]) - set(self.FCPList_exist[t-1])
                            for c in appear:
                                self.transfer_empty_para_postprocess(t,c,split_new,empty_para_array)
                            t = t+1
                        else:
                            self.c_time[split_new,1] = self.FCPList_time[t]
                            break
            else: # indicate _path_i[t-1] != _path_i[t]

                if jump_array[t] == 1: # no split or merge, jump time! 
                    backup_t = t
                    if _path_i[t-1] < self.c_max: # split!
                        split_flag = 0
                        c_end = path_i[t-1]
                        split_old = unused.pop()
                        split_new = unused.pop()

                        self.c_time[split_old,0] = self.FCPList_time[t]
                        self.c_time[split_old,1] = self.c_time[c_end,1]
                        self.c_time[c_end,1] = self.FCPList_time[t]
                        self.c_time[split_new,0] = self.FCPList_time[t] 

                        self.entity_list[split_old]= copy.copy(self.entity_list[c_end])
                        self.entity_list[split_old].remove(i)
                        self.entity_num[split_old] = len(self.entity_list[split_old])

                        self.entity_list[split_new].append(i)
                        self.entity_num[split_new] = 1
                    
                        # transfer other and itself parameter for eplit_old
                        self.Hawkes_b[split_old,split_old] = self.Hawkes_b[c_end,c_end]
                        self.Hawkes_eta[split_old,split_old] = self.Hawkes_eta[c_end,c_end]
                        self.Hawkes_zeta[split_old,split_old] = self.Hawkes_zeta[c_end,c_end]

                        for ts in range(t,len(path_i)):
                            if path_i[ts] == c_end:
                                path_i[ts] = split_old

                        while backup_t<len(self.FCPList_time):
                            if c_end in self.FCPList_exist[backup_t]:
                                self.FCPList_exist[backup_t].remove(c_end)
                                for c in self.FCPList_exist[backup_t]:
                                    self.copy_parameter_sampling(c_end,split_old,c)
                                self.FCPList_exist[backup_t].append(split_old)
                                # transfer the parameter between empty and c_end to the parameter between empty and split_old
                                empty_para_array[backup_t,split_old,:,:] = np.copy(empty_para_array[backup_t,c_end,:,:])  
                                backup_t = backup_t+1
                            else:
                                break
                        
                        # transfer other and itself parameter for split_new
                        self.Hawkes_b[split_new,split_new] = empty_para_array[t,self.c_max,0,0]
                        self.Hawkes_eta[split_new,split_new] = empty_para_array[t,self.c_max,0,1]
                        self.Hawkes_zeta[split_new,split_new] = empty_para_array[t,self.c_max,0,2]              
                        for c in self.FCPList_exist[t]:
                            self.transfer_empty_para_postprocess(t,c,split_new,empty_para_array)
                        t = t+1
                        while t<len(self.FCPList_time):

                            if _path_i[t] == _path_i[t-1]:
                                appear = set(self.FCPList_exist[t]) - set(self.FCPList_exist[t-1])
                                for c in appear:
                                    self.transfer_empty_para_postprocess(t,c,split_new,empty_para_array)
                                t = t+1    
                            else:
                                self.c_time[split_new,1] = self.FCPList_time[t]
                                new_c_flag = 1
                                split_flag = 1
                                break

                    else: # merge
                        new_c_flag = 1
                        split_flag = 1
                        c_end = path_i[t]
                        merge_new = unused.pop()
                        
                        self.c_time[merge_new,1] = self.c_time[c_end,1]
                        self.c_time[c_end,1] = self.FCPList_time[t]
                        self.c_time[merge_new,0] = self.FCPList_time[t]

                        self.entity_list[merge_new] = copy.copy(self.entity_list[c_end])
                        self.entity_num[merge_new] = len(self.entity_list[merge_new])
                        self.entity_list[merge_new].append(i)
                        self.entity_num[merge_new] = self.entity_num[merge_new]+1

                        # transfer other and itself parameter for merge_new
                        self.Hawkes_b[merge_new,merge_new] = self.Hawkes_b[c_end,c_end]
                        self.Hawkes_eta[merge_new,merge_new] = self.Hawkes_eta[c_end,c_end]
                        self.Hawkes_zeta[merge_new,merge_new] = self.Hawkes_zeta[c_end,c_end]
                        
                        for ts in range(t,len(path_i)):
                            if path_i[ts] == c_end:
                                path_i[ts] = merge_new
                        
                        while backup_t<len(self.FCPList_time):
                            if c_end in self.FCPList_exist[backup_t]:
                                self.FCPList_exist[backup_t].remove(c_end)
                                for c in self.FCPList_exist[backup_t]:
                                    self.copy_parameter_sampling(c_end,merge_new,c)
                                self.FCPList_exist[backup_t].append(merge_new)
                                empty_para_array[backup_t,merge_new,:,:] = np.copy(empty_para_array[backup_t,c_end,:,:])
                                # transfer the parameter between empty and c_end to the parameter between empty and split_old
                                backup_t = backup_t+1
                            else:
                                break

                        t = t+1
                        while t<len(self.FCPList_time):
    
                            if _path_i[t] == _path_i[t-1]:
                                t = t+1    
                            else:
                                new_c_flag = 1
                                split_flag = 1
                                
                                break
                            
                else: # previous an old community, must have split or merge  
                    new_c_flag = 0          
                    self.entity_list[path_i[t]].append(i)
                    self.entity_num[path_i[t]] = self.entity_num[path_i[t]] + 1
                    t = t+1
                    while t<len(self.FCPList_time):
                        if _path_i[t] == _path_i[t-1]:
                            t = t+1
                        else:
                            break


    def sample_scaling(self, para_select):

        # para_select  0:b       1:eta        2:zeta

        loglike = 0
        loglike_ = 0     

        new_b = 0
        new_eta = 0
        new_zeta = 0

        if para_select == 0:
            new_b = npr.uniform(low = 0, high = self.b_max)
        if para_select == 1:
            new_eta = npr.uniform(low = 0, high = self.eta_max)
        if para_select == 2:
            new_zeta = npr.uniform(low = 0, high = self.zeta_max)

        for i in range(self.d_num):
            for j in range(self.d_num):
                if i!=j:
                    s_array, r_array, time = self.path_for_hawkes_scaling(self.path_entity[i],self.path_entity[j])
                    
                    x_ij = self.observation_list[i][j]
                    x_ji = self.observation_list[j][i]

                    index_ij = np.searchsorted(time, x_ij, side = 'right')-1
                    index_ji = np.searchsorted(time, x_ji, side = 'right')-1

                    z_ij_1 = s_array[index_ij]
                    z_ij_2 = r_array[index_ij]

                    z_ji_1 = s_array[index_ji]
                    z_ji_2 = r_array[index_ji]

                    b = self.Hawkes_b[z_ij_1,z_ij_2]
                    eta = self.Hawkes_eta[z_ji_1,z_ji_2]
                    zeta = self.Hawkes_zeta[z_ji_1,z_ji_2]
                    b_base = self.Hawkes_b[s_array, r_array]

                    b_ = self.b * logit(b)
                    eta_ = self.eta * logit(eta)
                    zeta_ = self.zeta * logit(zeta)
                    b_base_ = self.b * logit(b_base)

                    loglike = loglike + self.cal_hawkes_likelihood_(time, b_, eta_, zeta_, b_base_, x_ij, x_ji)
                    
                    if para_select == 0:
                        b_ = new_b * logit(b)
                        b_base_ = new_b * logit(b_base)
                    if para_select == 1:
                        eta_ = new_eta * logit(eta)
                    if para_select == 2:
                        zeta_ = new_zeta * logit(zeta)

                    loglike_ = loglike_ + self.cal_hawkes_likelihood_(time, b_, eta_, zeta_, b_base_, x_ij, x_ji)

        print(loglike_,loglike,'loglike_,loglike')
        print(self.b,self.eta,self.zeta,'self.b,self.eta,self.zeta')
        u = loglike_ - loglike
        if u>0:
            if para_select == 0:
                self.b = new_b
            if para_select == 1:
                self.eta = new_eta
            if para_select == 2:
                self.zeta = new_zeta
            print(self.b,self.eta,self.zeta,'self.b,self.eta,self.zeta')

            return loglike_
        else:
            u = np.exp(u)
            if npr.uniform() < u:
                if para_select == 0:
                    self.b = new_b
                if para_select == 1:
                    self.eta = new_eta
                if para_select == 2:
                    self.zeta = new_zeta
                print(self.b,self.eta,self.zeta,'self.b,self.eta,self.zeta')
                return loglike_
            else:
                print(self.b,self.eta,self.zeta,'self.b,self.eta,self.zeta')
                return loglike

    def cal_hawkes_likelihood_(self, time__, b, eta, zeta, b_base, t_base, t_trigger):
        
        time_ = np.copy(time__[1:])
        time_ = np.append(time_,self.T)
        interval_base = time_-time__

        t_base = np.asarray(t_base)
        t_trigger = np.asarray(t_trigger)
        '''
        # calculate the likelihood
        s = 0
        for i in range(len(t_base)):
            s_ = b[i]
            for j in range(len(t_trigger)):
                if t_base[i]>t_trigger[j]:
                    # print(eta[j]*np.exp(-zeta[j]*(t_base[i]-t_trigger[j])),'eta[j]*np.exp(-zeta[j]*(t_base[i]-t_trigger[j]))')
                    s_ = s_ + eta[j]*np.exp(-zeta[j]*(t_base[i]-t_trigger[j]))
            s = s + np.log(s_)

        for i in range(len(time__)):
            # print(-b_base[i]*interval_base[i],'-b_base[i]*interval_base[i]')
            s = s - b_base[i]*interval_base[i]
        for j in range(len(t_trigger)): 
            # print(- eta[j]*(1 - np.exp( -zeta[j]*(self.T-t_trigger[j])))/zeta[j],'- eta[j]*(1 - np.exp( -zeta[j]*(self.T-t_trigger[j])))/zeta[j]')
            s = s - eta[j]*(1 - np.exp( -zeta[j]*(self.T-t_trigger[j])))/zeta[j]
        '''

        q = 0

        trigger_index = 0
        for i in range(len(t_base)):
            q_ = b[i]
            if trigger_index == len(t_trigger):
                q_ = q_ + np.dot(eta,np.exp(-zeta*(t_base[i]-t_trigger)))
            else:
                while trigger_index<len(t_trigger):
                    if t_base[i]>t_trigger[trigger_index]:
                        trigger_index = trigger_index +1
                        if trigger_index == len(t_trigger):
                            q_ = q_ + np.dot(eta[0:trigger_index],np.exp(-zeta[0:trigger_index]*
                                (t_base[i]-t_trigger[0:trigger_index])))
                    else:
                        q_ = q_ + np.dot(eta[0:trigger_index],np.exp(-zeta[0:trigger_index]*
                        (t_base[i]-t_trigger[0:trigger_index])))
                        break
            
            q = q + np.log(q_)


        q = q - np.dot(b_base,interval_base)
    
        q = q - np.dot(eta/zeta,1 - np.exp(-zeta *(self.T - t_trigger))) 
            
        return q
            
    
    def path_for_hawkes_scaling(self, path_s, path_r):

        path_s_time = self.c_time[path_s,0]
        path_r_time = self.c_time[path_r,0]

        time = np.union1d(path_s_time,path_r_time)
        time = np.sort(time, kind = 'mergesort')

        s_array = np.zeros(len(time),dtype=int)-1
        r_array = np.zeros(len(time),dtype=int)-1

        s_index = np.searchsorted(time,path_s_time,side = 'right')-1
        s_array[s_index]=path_s[:]

        for i in range(len(s_array)):
            if s_array[i]==-1:
                s_array[i] = s_array[i-1]+0

        r_index = np.searchsorted(time,path_r_time,side = 'right')-1

        r_array[r_index]=path_r[:]

        for i in range(len(r_array)):
            if r_array[i]==-1:
                r_array[i] = r_array[i-1]+0

        return r_array, s_array, time


    def transfer_empty_para_postprocess(self, t, c, new_index, empty_para_array): # checked
        
        # print(t, c, new_index,'t, c, new_index, in transfer_empty_para_postprocess')
        # print(empty_para_array[t,c,0,0],'self.Hawkes_b[c,new_index] in transfer_empty_para_postprocess ')
        
        self.Hawkes_b[c,new_index] = empty_para_array[t,c,0,0]
        self.Hawkes_eta[c,new_index] = empty_para_array[t,c,0,1]
        self.Hawkes_zeta[c,new_index] = empty_para_array[t,c,0,2]
        self.Hawkes_b[new_index,c] = empty_para_array[t,c,1,0]
        self.Hawkes_eta[new_index,c] = empty_para_array[t,c,1,1]
        self.Hawkes_zeta[new_index,c] = empty_para_array[t,c,1,2]

                    
def logit(x): # checked

    return 1/(1+np.exp(-x))


def normalize_weight(weight): # checked

    # function: nomalize the log weight to 1 

    weight[:] = weight[:] - np.amax(weight)
    weight[:] = np.exp(weight[:])
    weight[:] = weight[:]/np.sum(weight)



def derive_ob_info(x, time, t_begin_index, t_end_index):  # checked

    # notice the t_end_index = original t_end_index +1

    x = np.asarray(x)

    x_all = []
    x_prior = []
    x_like = []

    if x!=[]:
        index = np.where(x<time[t_end_index])[0]
        if index!= []:
            x_all = x[index]
        index = np.where(x<time[t_begin_index])[0]
        if index!=[]:
            x_prior = x[index]

    if x_all == []:
        pass 
    else:
        x_like = x_all[len(x_prior):]

    x_all = np.asarray(x_all)
    x_like = np.asarray(x_like)
    x_prior = np.asarray(x_prior)

    return x, x_all, x_like, x_prior


def MH(f_, f, q_, q, x_, x): # checked

    u = npr.uniform(low = 0.0 , high = 1.0)
    
    u_ = f_ + q_ - f - q

    if u_>0:
        return x_,f_
    elif np.exp(u_)>u:
        return x_,f_
    else:
        return x,f
    





    
    
        
        
            
            
        
    
       
                    
                    
                    
                    
                    
            
            
            
        
        
        
        
        
            
        
                
        
        
        
        
        
          
        
        