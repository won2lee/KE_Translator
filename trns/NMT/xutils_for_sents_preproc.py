
import re
import numpy as np
from itertools import chain
from trns.NMT.counter_utils import normal_to_special, special_to_normal
from trns.NMT.utils_etc import counter_convert2, insert_voc, special_to_normal3
#from utils_for_sents import special_to_normal3


def save_sents(sentences):
    p=re.compile(r'\.$')
    q = re.compile(r'\s+')
    z = re.compile(r'_ ') #영어로 번역할 때는 '_' 무시
    u = re.compile(r'(?P<to_fix>[0-9]+)')
                 
    sents = []
    for s in sentences:
        s = [''.join([c for c in w if ord(c) < 55204]) for w in s ]
        sents.append(q.sub(' ',u.sub(' \g<to_fix> ',z.sub(' ',p.sub(' .',' '.join(s))))))
        
        #sents.append(z.sub(' ',q.sub(' ',p.sub(' .',' '.join(s)))))
        
    #to_save = '\n'.join(sents)   
    
    return sents

def log_prob_by_len3(vocabs):
    z = re.compile(r'걟+')
    #s_val = sum(vocabs.values())
    lh_len = {}
    for i in range(30):
        lh_len[i] = []
    for k,v in vocabs.items():
        zk = z.sub('',k)
        if len(zk) > 29:continue
        lh_len[len(zk)].append(k)
    lh_val ={}
    for i in range(1,30): 
        #lh_val[i] = sum([np.log(vocabs[s]/s_val) for s in lh_len[i]]) / len(lh_len[i])
        if len(lh_len[i]) == 0:continue
        lh_val[i] = sum([np.log(vocabs[s]) for s in lh_len[i] if vocabs[s]>1]) / len(lh_len[i])
        #print(i, lh_val[i])
    for_long = sum([v for k,v in lh_val.items() if k>24])/5.
    for i in range(30,60):
        lh_val[i] = for_long
    return lh_val


def dict_add(k,v,tgt):

    if k in tgt.keys():
        tgt[k] += v
    else:
        tgt[k] = v
    return tgt

def dict_subtract_part(to_del,vocX,rate):
    for k,v in to_del.items():
        if k in vocX.keys():
            vocX[k] = vocX[k]-int(v*rate)
    return vocX

def vocab_split(vocX,to_split,rate):
    
    to_del ={}
    X = {}
    kwds = to_split
    #to_skip = [normal_to_special(w,key_vars) for w in ['구한말']]
    for k,v in kwds.items():
        ks = normal_to_special(k,key_vars)
        vs = normal_to_special(v,key_vars)
        if ks in vocX.keys():
            for x in vs.split(' '):
                _ = dict_add(x,vocX[ks],X)  #k,v,tgt
            _ = dict_add(ks,vocX[ks],to_del)
    
    _ = dict_subtract_part(to_del,vocX,rate)
    
    return vocX


def vocab_adjust(vocX,to_adjust,rate,to_skip=[''], to_match = 0): #단어 첫 캐릭터 부터 검색하지 않는 경우의 수
                                                                 #to_match=0 의 의미는 '모두 첫 캐릭터 부터 검색'
    to_del ={}
    aa = {}
    bb = {}
    cc = {}

    kwds = to_adjust
    pre_kwds = ['']*to_match+['^']*(len(kwds[to_match:]))
    to_skip = [normal_to_special(w,key_vars) for w in to_skip]
    for i,wk in enumerate(kwds):
        wks = normal_to_special(wk,key_vars)
        p = re.compile(pre_kwds[i]+wks)
        nS = 0
        for k,v in vocX.items():
            
            if p.search(k) is not None and k not in to_skip:
                #print(special_to_normal3(k,key_vars,keep_double= False))
                sl = k.find(wks)
                el = sl +len(wks)                       
                sects = [k[:sl], k[sl:el],k[el:]]
                #print(special_to_normal3(k,key_vars,keep_double= False), sects)
                for ik,dt in enumerate([aa,bb,cc]):
                    _ = dict_add(sects[ik],v,dt)
                dict_add(k,v,to_del)
                nS += 1
        
        if nS==0:
             _ = dict_add(wks,20,bb)
                
    vocX = dict_subtract_part(to_del,vocX,rate) 
    
    return vocX, aa, bb, cc


def magic4(s, z, k_args):
    k=k_args[0]
    fX = k_args[1]
    tX = k_args[2]
    k2 = k_args[3]
    v=len(s)
    X = min(2*k*(v-fX),0.4*(v-k2),2*k) if (v<7) else min(-0.2*k*(v-tX),2*k)
    return X



def search_max_prob4(subword,sub_comb,vocabs,vocs_dict,logprob,sub_sum,z,k_args):

    for j in reversed(range(1,len(subword))):
        sb = [z.sub('',subword[:j]),z.sub('',subword[j:])]

        if (sb[0] in vocs_dict[0].keys()) and (sb[1] in vocs_dict[1].keys()):

            temp = [vocs_dict[0][sb[0]], vocs_dict[1][sb[1]]]
            temp_sum = [np.log(vocabs[vocs_dict[i][s]])-logprob[len(s)] + magic4(s,z,k_args) for i,s in enumerate(sb)]        
            #temp_sum = sum([np.log(vocabs[s])-logprob[len(s)]+min(8*np.log(len(s)/8),0) for s in temp])/2
            if sum(temp_sum)/len(temp_sum) > sum(sub_sum)/len(sub_sum): 
                sub_sum = temp_sum
                sub_comb = temp
                
    return sub_comb,sub_sum



def forward_check6(subword,vocabs,vocs_dict,z,logprob,k_args,recursive=False):
    
    zsub = z.sub('',subword)
    if zsub in vocs_dict[1].keys():
        ws_fw = [vocs_dict[1][zsub]]
    else:
        ws_fw = ['NotInVocabs']
        zsub = 'NotInVocabs'
    
    pre_sub_sum = sub_sum = [np.log(vocabs[vocs_dict[1][zsub]])-logprob[len(zsub)]+magic4(zsub,z,k_args)]
        
    if len(subword) > 1:        
        
        sub_comb = ws_fw
        
        #dir_to_check = reversed(range(1,len(subword))) #if forward else range(1,len(subword))
        
        sub_comb, sub_sum = search_max_prob4(subword,sub_comb, vocabs,vocs_dict,logprob,pre_sub_sum,z,k_args) 
        if recursive == True and len(sub_comb)>1:
            
            sub_comb1, sub_sum1 = forward_check6(sub_comb[0],vocabs,vocs_dict,z,logprob,k_args,recursive=False)
            sub_comb = sub_comb1 + sub_comb[1:]
            sub_sum = sub_sum1 + sub_sum[1:]            
            
            sub_comb2, sub_sum2 = forward_check6(sub_comb[-1],vocabs,vocs_dict,z,logprob,k_args,recursive=False)
            sub_comb = sub_comb[:-1] + sub_comb2
            sub_sum = sub_sum[:-1] + sub_sum2
        
        if (pre_sub_sum < sum(sub_sum)/len(sub_sum)):
            ws_fw = sub_comb 
        else:
            sub_comb = pre_sub_sum
            
    return ws_fw, sub_sum


def get_vocs_dict(vocabs):
    
    vocs_dict = [{},{}]
    z = re.compile(r'걟+')
    zz = re.compile(r'걟걟$')

    for k,v in vocabs.items():
        if zz.search(k) is not None:
            vocs_dict[1][zz.sub('',k)] = k
            continue    
        zk = z.sub('',k)
        if zk not in vocs_dict[0].keys():
            vocs_dict[0][zk] = [k]
        else:
            vocs_dict[0][zk] +=[k]
    for k,l in vocs_dict[0].items():
        vocs_dict[0][k] = sorted(l,key=lambda x:len(x), reverse=True)[0]  
        
    temp_dict = vocs_dict[0].copy()
    for k,v in vocs_dict[1].items():
        temp_dict[k] = v
    vocs_dict[1] = temp_dict
    
    return vocs_dict

def to_bpe_sents10(sentences, vocabs,vocs_dict, logprob, key_vars, sum_voc_vals,extracted_vocs,k_args, cut):
            
    
    BASE_CODE = key_vars['BASE_CODE']
    vocabs.pop('',1) 

    vocabs['NotInVocabs'] = 0.001
    josas =  list(chain(*[[k for k in extracted_vocs[kc].keys() if len(k)>0] for kc in ['mids','subs']]))
    verbs = [k for k in extracted_vocs['verbs'] if len(k)>0]    
    mids = ['었', '았', 'ㅓㅆ', 'ㅏㅆ', 'ㅣㅆ', 'ㅆ', '어', '아', 'ㅓ', 'ㅏ', 'ㅣ', 'ㄴ','ㄹ','ㅁ','게']
    mids = [normal_to_special(w,key_vars) for w in mids]
    vms = list(set(verbs+josas+mids))
    
    
    len_cut = cut[0]
    log_cut = cut[1]
    
    all_s = []
    #fre_sum = sum(vocabs.values())
    fre_sum = sum_voc_vals
    p = re.compile(r'[^가-힣ㄱ-ㅎㅏ-ㅣ_\-]+')
    p2 = re.compile(r'[가-힣]')
    q = re.compile(r'\.$')
    r = re.compile(r'\s+')
    #u = re.compile(r'(?P<to_fix>[,\(\)\'\"\<\>])')
    u = re.compile(r'(?P<to_fix>[^A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ0-9_\-\.])')
    uek = re.compile(r'(?P<en>[A-Za-z])(?P<ko>[가-힣ㄱ-ㅎ])')
    uke = re.compile(r'(?P<ko>[가-힣ㄱ-ㅎ])(?P<en>[A-Za-z])')
    uknk = re.compile(r'(?P<notko>[^가-힣ㄱ-ㅎㅏ-ㅣ])(?P<ko>[가-힣ㄱ-ㅎ])')
    ukkn = re.compile(r'(?P<ko>[가-힣ㄱ-ㅎ])(?P<notko>[^가-힣ㄱ-ㅎㅏ-ㅣ])')
    z = re.compile(r'걟+')
    z2 = re.compile(r'»')
    zz = re.compile(r'걟걟$')
    #u1 = re.compile(r'‘')
    #u2 = re.compile(r'”')
    #ord('“'),ord('‘'),ord('"'),ord("'")
    #(8220, 8216, 34, 39)

                   
    for ii,s in enumerate(sentences):
        st = []
        for c in s:
            if ord(c) == 8220 or ord(c) == 8221:
                st.append(chr(34))
            elif ord(c) == 8216 or ord(c) == 8217:
                st.append(chr(39))
            elif c == '»':
                st.append('>>')
            else:
                st.append(c)

        st = ''.join(st)
        st = r.sub(' ',st).strip().split(' ')
        st = '»'.join(st)
        st = uek.sub('\g<en> \g<ko>', st)
        st = uke.sub('\g<ko> \g<en>', st)
        st = uknk.sub('\g<notko> \g<ko>', st)
        st = ukkn.sub('\g<ko> \g<notko>', st)
        st = u.sub(' \g<to_fix> ',st)
        #st = u1.sub(' ‘',st)
        #st = u2.sub(' ”',st)
        st = q.sub(' .',st)
        st = r.sub(' ',st)
        st = [wd for wd in st.strip().split(' ')]
        """
        sent = []
        for wd in st.strip().split(' '):
            if wd[-1] == '_':
                sent += [wd]
            else:
                sent += [wd,'_']
        """
        #st = r.sub(' ',' '.join([u.sub(' \g<to_fix> ',wd) for wd in st])).split(' ')
            
        #if ii<5:
        #    print("s : {}".format(st))
        sentence = []
        for w in st:
            if (w == '') or (w ==' '):continue

            if p.search(w) is not None:                       
                #sentence += [w,'_']
                sentence += [z2.sub('__',w)]
                continue                    
            #print(w)
            n_fw = []
            n_bw = []

            bpe_w= ''.join([chr(i+BASE_CODE) for i in counter_convert2(w,key_vars)[1]])
            
            s_w = bpe_w + '걟걟'
            """ 
            if s_w in vocabs.keys():
                n_bf = [s_w]
            elif z.sub('',s_w[:-1]) in vocabs.keys():
                n_bf = [z.sub('',s_w[:-1])]
            else:
                n_bf = []
            """            
            #n_fw = forward_check3(s_w,vocabs,logprob,fre_sum)
            
            n_word = []  
            
            ifw = 0  
            pre_len = 0
            
            while len(z.sub('',s_w)):           
                #print(s_w)
                pre_sw = s_w
                #print("s_w : {}".format(s_w))
                

                n_fw,_ = forward_check6(s_w,vocabs,vocs_dict,z,logprob,k_args,recursive=True)
                if n_fw[0] != 'NotInVocabs':
                    n_word += n_fw
                    #sentence += n_word + ['걟'] if n_word[-1][-1] !='걟' else n_word
                    break

                #nj = len(s_w)-ifw+2
                #for i in reversed(range(1,len(s_w)-ifw+1)):
                #    if s_w[i:nj] in josas:
                #        nj = i
                
 
                for i in reversed(range(1,len(s_w)-ifw)): # len(s_w) -2 로 한 것은 n_bf와 중복을 피하기 위해 ...
                    #if (s_w[i:] == '걟갧') or (s_w[i:] == '걟'):continue
                    
 
                    if s_w[:i] in vocabs.keys():
                        res_w = special_to_normal(s_w[i:],key_vars)
                        if ord(res_w[0]) not in range(12593, 12644):
       
                            n_fw,_= forward_check6(s_w[:i],vocabs,vocs_dict,z,logprob,k_args,recursive=True)
                            n_word += n_fw

                            s_w = s_w[i:]
                            break 
                    
                        elif ((s_w[:i] in vms and res_w[0] in ['ㄴ','ㄹ','ㅁ','ㅆ','ㅏ','ㅓ'])
                              or (s_w[:i] in extracted_vocs['nouns'] and res_w[0] in ['ㅅ'])):
                            n_fw,_= forward_check6(s_w[:i],vocabs,vocs_dict,z,logprob,k_args,recursive=True)
                            n_word += n_fw

                            s_w = s_w[i:]
                            break                             
                    
                                       
                if s_w == pre_sw:
                    not_in_vocabs = 1
                    for i in range(len(s_w)):
                        if z.sub('',s_w[i:]) in vocs_dict[0].keys():
                            n_word += ["<" + s_w[:i] +">"] + forward_check6(s_w[i:],vocabs,vocs_dict,z,logprob,k_args,recursive=True)[0]
                            insert_voc(vocabs,"<" + s_w[:i] +">",1)
                            not_in_vocabs = 0                            
                            break
                    if not_in_vocabs == 1:
                        n_word.append("<" + s_w +">")
                        insert_voc(vocabs,"<" + s_w +">",1)
                    break
 

            n_w = [z.sub('걟',w) for w in n_word[:-1]] + [n_word[-1]] if len(n_word)>1 else n_word
    
            #if ii in range(3):
            #    print(ii, [special_to_normal(ss,key_vars) for ss in n_w])
                        
            if len(n_w)>1:
                window = 2
                to_continue = 1
                
                while to_continue:
                    
                    to_continue = 0
                
                    mx = 1.4 if n_w[-1] not in josas else 1.0
                    
                    for i in range(window,len(n_w)+1):
                        
                        to_ch = ''.join(n_w[i-window:i])
                        if to_ch not in vocabs:
                            continue
                        
                        nx = min(max(mx, 1.+(len(n_w)-i)*0.2), 1.4) # 조사의 잔존가능성은 유지하고 조사가 아닌 경우 가능한 결합
                            
                        temp_sum = [np.log(vocabs[ww])-logprob[len(ww)] + magic4(ww,z,k_args) for ww in n_w[i-window:i]]
                        if sum(temp_sum)/len(temp_sum) < np.log(vocabs[to_ch])*nx-logprob[len(to_ch)] + magic4(to_ch,z,k_args):
                            n_w = n_w[:i-window]+[to_ch]+n_w[i:]
                            to_continue = 1 
                            window = 2  # 결합된 것이 있으면 원도우는 다시 2로 세팅
                            break
                    
                    if to_continue == 0:
                        if window < len(n_w):
                            window += 1
                            to_continue = 1           
            
            """
            n_bw = []
            nj = 0
            
            #print([special_to_normal(ss,key_vars) for ss in n_w])

            for i, ww in enumerate(reversed(n_w)):
                
                if ww in mids:
                    temp = ww
                    nj = 1
                    
                elif nj==1:
                    if ww in verbs:
                        n_bw +=[temp,ww]
                        nj = 0
                    else:
                        break
                    
                elif ww in josas:
                    n_bw.append(ww)
                    
                elif i==0 and len(special_to_normal3(ww,key_vars,keep_double=False))<2:
                    break
                    
                elif ww in verbs:
                    wws = special_to_normal3(ww,key_vars,keep_double=False)
                    if len(wws)>1:
                        n_bw.append(ww)
                    elif wws in ['당','하','해','되','이','오','가','지']:
                        n_bw.append(ww)
                    else:
                        break
                        
                    
                else:
                    break
                    
            n_rw = n_w[:-len(n_bw)] if len(n_bw)>0 else n_w  
            
            #if normal_to_special('검',key_vars) in n_w: 
            #    print('\n\n',n_rw, [special_to_normal(wkk,key_vars) for wkk in n_w],'\n\n\n')

 
            if len(n_rw) > 1:
                
                len_avg = sum([len(ww) for ww in n_rw])/len(n_rw)
                log_avg = sum([np.log(vocabs[ww])*len(ww)/len_avg for ww in n_rw])/len(n_rw) * (len_avg/6.0)              
                
                if log_avg < log_cut and len_avg < len_cut: 
                    
                    #if ''.join(n_rw) in vocabs().keys(): # 추가된 부분
             
                    n_rw = [''.join(n_rw)]
                
                else:
                    n_temp = ['']
                    l = len(n_rw)
                    pre_pop = 0
                    for i,rw in enumerate(n_rw):
                        wn = special_to_normal3(rw,key_vars,keep_double=False)    
                        if pre_pop == 1:
                            n_temp[-1] = n_temp[-1]+rw
                            pre_pop = 0
                        
                        elif len(rw) < 2:
                            n_temp[-1] = n_temp[-1]+rw
                            pre_pop = 1                            
                            
                        elif len(wn) > 1:
                            if ord(wn[0]) in range(12593,12644) and ord(wn[-1]) in range(12593,12644):
                                n_temp[-1] = n_temp[-1]+rw
                                pre_pop = 1
                            elif ord(wn[0]) in range(12593,12644):
                                n_temp[-1] = n_temp[-1]+rw
                            elif ord(wn[-1]) in range(12593,12644):
                                n_temp.append(rw)
                                pre_pop = 1
                            else:
                                n_temp.append(rw)

                        else:
                              
                            w1 = ''.join(n_rw[-l+i-1:i])
                            w2 = ''.join(n_rw[i:i+1])
                            if w1 != rw and w1 in vocabs.keys():
                                if w2 in vocabs.keys():
                                    if vocabs[w1] > vocabs[w2]: 
                                        n_temp[-1] = n_temp[-1]+rw
                                    else:
                                        n_temp.append(rw)
                                        pre_pop = 1
                                else:
                                    n_temp[-1] = n_temp[-1]+rw
                            else:
                                if w2 in vocabs.keys():
                                    n_temp.append(rw)
                                    pre_pop = 1 
                                else:
                                    n_temp[-1] = n_temp[-1]+rw #양방향 모두 의미 없으면 3개 글자 모두 합쳐 하나로!
                                    pre_pop = 1 

                    n_rw = [ww for ww in n_temp if ww!='']

            n_w = n_rw + n_bw[::-1]                
            """            
            #if ii in range(3):
            #    print(ii,'last',[special_to_normal(w,key_vars) for w in n_w])
            #n_w = [z2.sub('_',chars) for chars in n_word]
            #sentence += n_w + ['걟'] if n_w[-1][-1] !='걟' else n_w
            
            sentence +=n_w
        #sentence = [sentence[0]] + [wds  for i,wds in enumerate(sentence[1:]) if sentence[i][-1] !='걟걟']
        all_s.append(sentence)

    sents = []
    for s in all_s:
        sent = []
        for w in s:
            #to_add = special_to_normal(w,key_vars) if p2.search(w) is not None else w
            to_add = w if p2.search(w) is None else special_to_normal3(w,key_vars,keep_double=False)
            """
            ######
            special_to_normal 을 고칠지 한글로 말들어진 후에 고칠 지 검토!!!
            count_utils 의 double_vowel_lookup2(JUNGSUNG_LIST) 와  
            counter_utils 의 counter_ind_char2(idx, key_vars,keep_double=True) 수정중
            ##########
            """
            sent.append(to_add)    
        #sents.append([special_to_normal(w,key_vars) for w in s])
        sents.append(sent)
    
    return sents 
