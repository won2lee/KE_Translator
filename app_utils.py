import re
from trns.NMT.xutils_for_key_vars import make_key_vars

def to_start(X):
    p = re.compile('\s+')
    p2 = re.compile('\_')
    q = re.compile('\:')
    q1 = re.compile('\[[\s\_0-9a-z]+\]')  #위키 등 주석 없애기
    q2 = re.compile('(?P<xCap>[^A-Z])\s*(?P<xdot>[\.\?\!])[\s\_]+')  # 앞에 대문자 오는 경우  J. F. Kennedy 같은 경우 제외
    q3 = re.compile('(?P<dotw>\.[a-zA-Z0-9가-힣]+)\.\_') #0519추가  ???
    #q4 = re.compile(r'\n')
    q4 = re.compile('\s{3,}')
    q5 = re.compile('\.\.') 
    q6 = re.compile('\.(?P<qmark>[\"\'\’\”])\s*(?P<enC>[A-Z][a-z]+)') ## 고민해야 할 부분
    q7 = re.compile('(?P<dotw>(Mr|[ap]\.m|Dr|Sen))\.\s')
    q8 = re.compile('Æ')
    q9 = re.compile('Ë')
    #p10 = re.compile('\s{3,}')
    Xa = q1.sub(' ',p.sub(' ',q4.sub('Ë',X))) #q4.sub('Ë',X)))
    XX = p.sub(' ',q9.sub('_Dawn of 2199._',q5.sub('.',q6.sub('.\g<qmark>_\g<enC>',q8.sub('.',q2.sub('\g<xCap>\g<xdot>_',
                        q7.sub('\g<dotw>Æ ',Xa))))))).split('_')
    return XX

def to_start_depricated(X):
    p = re.compile('\s+')
    p2 = re.compile('\_')
    q = re.compile('\:')
    q1 = re.compile('\[[\s\_0-9a-z]+\]')  #위키 등 주석 없애기
    q2 = re.compile('\.[\s\_]+')
    q3 = re.compile('(?P<dotw>\.[a-zA-Z0-9가-힣]+)\.\_') #0519추가
    q4 = re.compile('\\n')
    q5 = re.compile('\.\.')
    q6 = re.compile('\"\s*(?P<enC>[A-Z][a-z]+)')
    X = p.sub(' ',q5.sub('.',q3.sub('\g<dotw>. ',q6.sub('"_\g<enC>',q2.sub('._',q4.sub('_',q1.sub(' ',X))))))).split('_')
    return X

def rid_blank(snts):
    p = re.compile('\s+')
    p3 = re.compile('\<.*\>')
    p4 = re.compile('\<unk\>') #5.14 add
    p5 = re.compile('\?\s*\?')
    q2 = re.compile('\s+\,')
    q3 = re.compile('\s+\.')
    q4 = re.compile('\s+\)')
    q5 = re.compile('\(\s+')
    q6 = re.compile('(?P<num1>[0-9][\,\.]*)\s+(?P<num2>[0-9])')
    q7 = re.compile('\-\s+(?P<num3>[0-9])')
    q8 = re.compile('\_')
    q9 = re.compile('(?P<quo>[\‘\“])\s+')
    q10 = re.compile('(?P<num4>[0-9])\˅')
    
    z1 = re.compile('\s*(?P<qt1>[\'\"])[1]\s*')
    z2 = re.compile('\s*(?P<qt2>[\'\"])[0]\s*')
    z3 = re.compile("\s*\'[sS]\s+")
    z4 = re.compile("\s*Dawn of 2199.\s*")

    #z3 = re.compile("\s*\\\*\'s\s*")

    sents = []
    
    for s in snts:  
        n=0
        if p3.search(s) is not None:
            sent = []
            for w in s.split(' '):
                #print(w)
                if w =='<':
                    n =1 
                elif w =='>':
                    n=0
                elif n==1:
                    n +=1
                    sent.append(w)
                elif n>1:
                    sent[-1] += w
                else:
                    sent.append(w)
            s = ' '.join(sent)
            #sents.append(' '.join(sent))
        s = q7.sub('-\g<num3>',q6.sub('\g<num1>\g<num2>',q6.sub('\g<num1>\g<num2>',q5.sub('( ',
                                                        q4.sub(')',q3.sub('.',q2.sub(',',p4.sub('?',s))))))))
        #if len(s.split(' ')) < 1: continue
        #print("1", s)
        snt = q9.sub('\g<quo>',z3.sub("'s ",rid_sbol(s, p)))
        
        #print("3", s)
        sents.append(snt.strip())
        #sents.append(q8.sub('',snt).strip())
        
    sents = p.sub(' ',' '.join(sents)) 
    sents = q10.sub('\g<num4>,',z2.sub('\g<qt2> ',z1.sub(' \g<qt1>',sents)))
    #sents = z2.sub('\g<qt2> ',z2.sub('\g<qt2> ',z2.sub('\g<qt2> ',z2.sub(
    #    '\g<qt2> ',z1.sub(' \g<qt1>',z1.sub(' \g<qt1>',z1.sub(' \g<qt1>',z1.sub(' \g<qt1>',sents))))))))           
    return z4.sub('\n\n',p5.sub('',p5.sub('',p5.sub('',sents))))


def rid_sbol(X, p):
    

    s = ''.join(['Æ'+w if w[0] in ['_','^','`'] else w for w in p.sub(' ',X).strip().split(' ') if w != ''])
    #print("2", s)
    snt = []
    for w in s.split('Æ'):
        if len(w)>1:
            if w[0] == '_':
                if w[1] in [',','.','”','’']:
                    snt.append(w[1:])
                else:
                    snt.append(' '+w[1:])
            elif w[0] == '^':
                snt.append(' '+w[1:].title())
            else:
                snt.append(' '+w[1:].upper())  

        elif w != '':
            snt.append(w)
    sent = []
    n1 = 0
    n2 = 0
    for w in snt:
        if w[-1] == '"':  #   ,"'"]
            n1 += 1
            sent.append(w+str(n1%2))
        elif w[-1] == "'":
            n2 += 1
            sent.append(w+str(n2%2))   
        else:
            sent.append(w)

                        
    return ''.join(sent)
    

def preproc_num(X):

    p1 = re.compile('\(\s*\)')
    p2 = re.compile('\s+')
    q1 = re.compile('(?P<nA>[A-Za-z])')
    q2 = re.compile('[A-Za-z][a-z]+')
    q3 = re.compile('(?P<n1>[0-9])(?P<n2>[0-9][0-9])(?P<n3>[^0-9])')
    q4 = re.compile('\.\s*(?P<n4>[0-9][0-9])(?P<n5>[0-9])')
    q6 = re.compile('(?P<n6>[0-9])\s(?P<n7>[0-9])')
    q7 = re.compile('(?P<n8>[0-9])\s*\.\s*(?P<n9>[0-9])')

    q8 = re.compile('\<\s*(?P<n9>[0-9][0-9\s*\.]*[0-9\s]*)\>')
    q9 = re.compile('\(\s*\)')   # 빈괄호
    q10 = re.compile('\([\s\_]*\,')


    X = [p2.sub(' ',q3.sub('\g<n1> \g<n2> \g<n3>',q3.sub('\g<n1> \g<n2> \g<n3>',q3.sub('\g<n1> \g<n2> \g<n3>',
                            q7.sub('\g<n8> . \g<n9>',q4.sub('. \g<n4> \g<n5>',
                            q6.sub('\g<n6>\g<n7>',s))))))) for s in X]
    X = [s for s in X if len(s.split(' '))>1]

    #X = p2.sub(' ',q8.sub(' \g<n9> ',X)).strip()
    
    return X


def to_normal(sents):
    
    key_vars = make_key_vars()
        
    CHOSUNG = key_vars['CHOSUNG']
    JUNGSUNG = key_vars['JUNGSUNG']
    BASE_CODE = key_vars['BASE_CODE']
    #CHOSUNG_LIST  = key_vars['CHOSUNG_LIST'] 
    JUNGSUNG_LIST = key_vars['JUNGSUNG_LIST']
    JONGSUNG_LIST = key_vars['JONGSUNG_LIST'] 
    d_to_single = key_vars['d_to_single']
    k_alpha_to_num = key_vars['k_alpha_to_num']
    
    p1 = re.compile('\s+')
    p2 = re.compile('\<unk\>') 
    p = re.compile('[가-힣]')
    q = re.compile('[ㄱ-ㅎㅏ-ㅣ]')
    q1 = re.compile('\s+')
    q2 = re.compile('\_')
    q6 = re.compile('(?P<num1>[0-9][\,\.]*)\s+(?P<num2>[0-9])')
    
    q9 = re.compile('(?P<quo>[\‘\“])\s+')
    q10 = re.compile('(?P<num4>[0-9])\˅')
    
    z1 = re.compile('\s*(?P<qt1>[\'\"])[1]\s*')
    z2 = re.compile('\s*(?P<qt2>[\'\"])[0]\s*')
    z3 = re.compile("\s*'[sS]\s+")
    z4 = re.compile("\s*2199년의 새벽.\s*")
    #sent = q2.sub(' ',q1.sub('',sent))
    
    #sents = [list(q6.sub('\g<num1>\g<num2>',q2.sub(' ',q1.sub('',p2.sub('?',s))))) for s in sents]
    sents = [q6.sub('\g<num1>\g<num2>',q1.sub(' ',p2.sub('?',s))) for s in sents]
    sents = [list(q9.sub('\g<quo>',rid_sbol(s, p1))) for s in sents if len(s.split(' ')) > 0]
    
    snts = []
    for s in sents:
        n_check = 0  #  이전에 [가-힣]이 없었다 
        snt = []
        for keyword in s:
            if q.match(keyword) is None:
                snt.append(keyword)
                n_check = 1
            elif n_check == 1: 
                preW = snt[-1]
                k_num = k_alpha_to_num[1][keyword]-28

                char_code = ord(preW) - BASE_CODE
                char1 = int(char_code / CHOSUNG)
                char2 = int((char_code - (CHOSUNG * char1)) / JUNGSUNG)
                
                if keyword in JUNGSUNG_LIST:  
                    if (char2,k_num) in d_to_single.keys(): 
                        char2 = d_to_single[(char2,k_num)]
                        snt[-1] = chr(BASE_CODE + char1*CHOSUNG + char2*JUNGSUNG)
                    else:
                        snt.append(keyword)
                elif keyword in JONGSUNG_LIST:
                    snt[-1] = chr(BASE_CODE + char1*CHOSUNG + char2*JUNGSUNG + k_num-28)
                else:
                    snt.append(keyword)
                    n_check = 0
            else:
                snt.append(keyword)
        
        snts.append(''.join(snt)) 
        
    snts = p1.sub(' ',' '.join(snts)) 
    snts = q10.sub('\g<num4>,',z3.sub("'s ",z2.sub('\g<qt2> ',z1.sub(' \g<qt1>',snts))))   
    return z4.sub('\n\n',p1.sub(' ',snts))