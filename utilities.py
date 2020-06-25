from numpy import genfromtxt
import numpy as np
import numpy.random as rn
import pandas as pd
import pickle

    
    
def process_txt_asym(dat, time_begin = 0, scale_T = 100000000):
    
    print((dat.shape),'data_shape', type(dat))
    
    dat[:,3] = dat[:,3]/scale_T
    t_min = np.amin(dat[:,3])   
    dat[:,3] = dat[:,3] - t_min + time_begin
    observation_table = np.copy(dat)
    c_table = np.zeros((np.shape(dat)[0],2))
    observation_table = np.concatenate((observation_table,c_table), axis = 1)
    
    
    
    u_id1 = np.unique(dat[:,0])
    u_id2 = np.unique(dat[:,1])
    u_id = np.union1d(u_id1,u_id2)
    d_num = len(u_id)
    observation_list = []
    for i in range(d_num):
        observation_list.append([])
        for j in range(d_num):
            observation_list[i].append([])
            
            

    num = np.shape(dat)[0]
    
    ob_binary = np.zeros((num,num),dtype = int)
    
    for i in range(num):
        s1 = dat[i,0]
        s2 = dat[i,1]
        s1_index = np.where(u_id == s1)[0][0]
        s2_index = np.where(u_id == s2)[0][0]
        
        observation_list[s1_index][s2_index].append(dat[i,3])
        ob_binary[s1_index,s2_index] = 1
        
        observation_table[i,0] = s1_index
        observation_table[i,1] = s2_index
        
        
    binary_x, binary_y = np.where(ob_binary == 1)

    T = np.amax(dat[:,3])
        
    return d_num, u_id, observation_list, ob_binary, binary_x, binary_y, observation_table, T  




def process_txt_sym(dat, time_begin = 0, scale_T = 30000):
    
    
    print((dat.shape),'data_shape', type(dat))
    
    dat[:,3] = dat[:,3]/scale_T
    t_min = np.amin(dat[:,3])   
    dat[:,3] = dat[:,3] - t_min + time_begin  
    
    
    u_id1 = np.unique(dat[:,0])
    u_id2 = np.unique(dat[:,1])
    u_id = np.union1d(u_id1,u_id2)
    d_num = len(u_id)
    observation_list = []
    for i in range(d_num):
        observation_list.append([])
        for j in range(d_num):
            observation_list[i].append([])
            
            

    num = np.shape(dat)[0]
    
    ob_binary = np.zeros((num,num),dtype = int)
    
    for i in range(num):
        s1 = dat[i,0]
        s2 = dat[i,1]
        s1_index = np.where(u_id == s1)[0][0]
        s2_index = np.where(u_id == s2)[0][0]
        
        
        observation_list[s1_index][s2_index].append(dat[i,3])
        observation_list[s2_index][s1_index].append(dat[i,3])

        ob_binary[s1_index,s2_index] = 1
        ob_binary[s2_index,s1_index] = 1
        
    binary_x, binary_y = np.where(ob_binary == 1)

    T = np.amax(dat[:,3])
        
    return d_num, u_id, observation_list, ob_binary, binary_x, binary_y, T  


def preprocess_txt(file_name):

    dat = np.loadtxt(file_name, dtype=float)
    print(np.shape(dat),'shape of original data')
    dat = np.unique(dat, axis = 0)
    print(np.shape(dat),'shape of preprocessed data')


    return dat


def derive_ob_info(x, time, t_begin_index, t_end_index):   

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
        x_like = []
    else:
        x_like = x_all[len(x_prior):]

    return x_all, x_like, x_prior
    
    

    

def preprocess_undata(scaling = 1):

    df = pd.read_csv('UNdata.csv',encoding = "ISO-8859-1")

    me = df['co'].values.tolist()
    country = df['Country'].values.tolist()
    date = df['date'].values.tolist()
    vote = df['vote'].values.tolist()
    yes = df['yes'].values.tolist()
    no = df['no'].values.tolist()
    abstain = df['abstain'].values.tolist()
    member = df['member'].values.tolist()
    country_set = list(set(country))
    cty = []

    for i in range(197):
        if member[i] == 1 and vote[i]!=9:
            cty.append(country[i])

    '''
    cty_index = np.zeros(len(cty))
    for i in range(len(cty)):

        cty_index[i] = country_set.index(cty[i])
    '''
    print(country_set,'country_set')

    # print(cty_index,'cty_index')


    observation_list = []
    for i in range(len(cty)):
        observation_list.append([])
        for j in range(len(cty)):
            observation_list[i].append([])



    index = []
    for i in range(len(country)):
        if country[i] in cty and me[i] == 1:
            index.append(i)
            
    date_ = []

    for i in index:
        date_.append(date[i])

    print(len(date_))
    print(len(set(date_)))

    date_unique = list(set(date_))
    date_unique.sort()

    for i in date_unique:
        print(i)

    count = 0
    for t in date_unique:

        for i in index:

            if  date[i] == t :
                # print(t)
                abstain_ = abstain[i]
                yes_ = yes[i]
                no_ = no[i]

                flag_country = np.zeros(len(cty))
                ob = np.zeros(len(cty))
                for j in index:

                    if date[j] == t and abstain[j] == abstain_ and yes[j] == yes_ and no[j] == no_:
                        # print(country[j])
                        c_index = cty.index(country[j])
                        flag_country[c_index] = 1
                        ob[c_index] = vote[j]

                if np.sum(flag_country) == len(cty):
                    for m in range(len(cty)):
                        for n in range(m+1,len(cty)):
                            if ob[m] == ob[n] :
                                observation_list[m][n].append(t)
                                observation_list[n][m].append(t)
                    count = count +1
                else:
                    pass
                    # print('wrong')
                break


    print(count,'count')


    for m in range(len(cty)):
        for n in range(m+1,len(cty)):
            print(len(observation_list[m][n]),m,n)            

    print(len(date_unique))


    for m in range(len(cty)):
        for n in range(len(cty)):
            for i in range(len(observation_list[m][n])):
                index = date_unique.index(observation_list[m][n][i])
                observation_list[m][n][i] = index /scaling

    for m in range(len(cty)):
        for n in range(m+1,len(cty)):
            print(len(observation_list[m][n]),m,n)    


    for m in range(len(cty)):
        for n in range(m+1,len(cty)):
            s = np.asarray(observation_list[m][n])
            
            for i in range(1,len(s)):
                if s[i]>s[i-1]:
                    pass
                else:
                    print('wrong',s[i],s[i-1],m,n)

    d_num = len(cty)
    T = len(date_unique)
    return d_num, observation_list, T/scaling


def preprocess_undata_0323(scaling = 100):
    df = pd.read_csv('UNdata.csv',encoding = "ISO-8859-1")

    me = df['me'].values.tolist()
    country = df['Country'].values.tolist()
    date = df['date'].values.tolist()
    vote = df['vote'].values.tolist()
    yes = df['yes'].values.tolist()
    no = df['no'].values.tolist()
    abstain = df['abstain'].values.tolist()
    member = df['member'].values.tolist()
    cty= list(set(country))

    ob = []
    for i in range(len(cty)):
        ob.append([])
        for j in range(len(cty)):
            ob[i].append([])

    date_unique = []
    i = 0
    s = 0
    while i < len(date):
        cur_date = date[i]
        if me[i]==1 and cur_date not in date_unique:

            
            cur_yes = yes[i]
            cur_no = no[i]
            cur_abstain = abstain[i]
            print(cur_yes,cur_no,cur_abstain,cur_date)
            
            date_unique.append(cur_date)
            
            vote_number = [1,2,3,8,9]
            vote_group = [[] for i in range(len(vote_number))]

            while cur_yes == yes[i] and cur_no == no[i] and cur_abstain == abstain[i] and cur_date == date[i] and me[i]==1:
                
                v_index = vote_number.index(vote[i])
                c_index = cty.index(country[i])
                vote_group[v_index].append(c_index)

                if i < len(date):
                    i = i + 1
            

            for v in range(len(vote_number)-2):
                for p in vote_group[v]:
                    for q in vote_group[v]:
                        if p!=q:
                            ob[p][q].append(cur_date)
                            



            while date[i] == cur_date:

                if i < len(date):

                    i = i + 1


            
        else:
            i = i+1


    for i in range(len(cty)):
        for j in range(len(cty)):
            ob[i][j] = list(set(ob[i][j]))

    print('postprocess')
    print(len(date_unique))
    print(len(set(date_unique)))

    date_unique.sort()


    for m in range(len(cty)):
        for n in range(len(cty)):
            if len(ob[m][n]) != len(set(ob[m][n])):
                print('wrong',len(ob[m][n]),len(set(ob[m][n])))
                exit()
            for i in range(len(ob[m][n])):
                index = date_unique.index(ob[m][n][i])
                ob[m][n][i] = index/scaling
            ob[m][n].sort()

    for m in range(len(cty)):
        for n in range(m+1,len(cty)):
            s = np.asarray(ob[m][n])
            
            for i in range(1,len(s)):
                if s[i]>s[i-1]:
                    pass
                else:
                    print('wrong',s[i],s[i-1],m,n)



    with open("undata_me_0402_1.txt", "wb") as fp:   #Pickling
        pickle.dump(ob, fp)

    with open("un_country_0402_1.txt", "wb") as fp:   #Pickling
        pickle.dump(cty, fp)
    with open("undata_me_data.txt","wb") as fp: #Pickling
        pickle.dump(date_unique, fp)
    d_num = len(cty)
    T = len(date_unique)
    return d_num, ob, T/scaling



