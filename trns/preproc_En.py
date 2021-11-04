## 6.20 updated

import re
    
def preproc_en(X_sents,vocs):

    p = re.compile('(?P<to_fix>[^A-Za-z0-9\'\-\.\˅])')  
    p1 = re.compile('(?P<fix1>[A-Za-z]+)\’(?P<fix2>[sm])(?P<fix3>[^a-z0-1])')
    p2 = re.compile('(?P<fix4>[A-Za-z]+)\’(?P<fix5>[rl][el])(?P<fix6>[^a-z0-1])')
    p3 = re.compile('(?P<fix7>[A-Za-z]+)\’(?P<fix8>t)(?P<fix9>[^a-z0-1])')

    q1 = re.compile("\.\s*$")
    q2 = re.compile("\s+\'")
    q3 = re.compile("\'\s+")
    q4 = re.compile("\.\s*\'")
    q5 = re.compile('\.\s*\"')
    q6 = re.compile('\s+')

    u1 = re.compile('(?P<to_add1>\s[a-z]{2,})\.\s+(?P<to_add3>[A-Z][a-zA-Z]*)')
    u2 = re.compile('(?P<to_add2>\s[A-Z][a-z]{4,})\.\s+(?P<to_add3>[A-Z][a-zA-Z]*)')

    ut = re.compile('[0-9]*[A-Z][a-z]*[0-9]*')
    ui = re.compile('[0-9]*[A-Z][A-Z]+[0-9]*')

    z1 = re.compile('[^A-Za-z]')
    z2 = re.compile('\s+')
    z3 = re.compile('(?P<num1>[0-9]+[\.\˅]{,1}[0-9]*)')
    z4 = re.compile('(?P<num2>[0-9])\,(?P<num3>[0-9])')

    X_sents = [q6.sub(' ',u1.sub('\g<to_add1> . \g<to_add3>',u2.sub('\g<to_add2> . \g<to_add3>',q5.sub(' . "',
                             q4.sub(" . '",q3.sub(" ' ",q2.sub(" ' ",p.sub(' \g<to_fix> ',
                            z4.sub('\g<num2>˅\g<num3>',(q1.sub(' .',p3.sub("\g<fix7>'\g<fix8>\g<fix9>",
                            p2.sub("\g<fix4>'\g<fix5>\g<fix6>",p1.sub("\g<fix1>'\g<fix2>\g<fix3>",s))))))))))))))
                for s in X_sents]

    sents =[]

    for s in X_sents:
        sent = []
        #for w in z2.sub(' ',s.lower()).strip().split(' '):
        for ws in z2.sub(' ',s).strip().split(' '):

            if ui.match(ws) is not None:
                wtag = chr(96) #initial
            elif ut.match(ws) is not None:
                wtag = '^' #upper
            else:
                wtag = '_' #lower

            wx = []
            for w in z3.sub(' \g<num1> ',ws).strip().split(' '):

                if z1.match(w) is not None:
                    wx += [w]
                else:
                    wx += w_recursive(w.lower(),vocs ,iter_first=True) 
            sent.append(' '.join([wtag] + wx))

        sents.append(q6.sub(' ',' '.join(sent)).strip())
        sents = [s for s in sents if len(s)>0]

    return sents


def w_recursive(w, X, iter_first=False):

    if w in X.keys():
        return [w]  

    if w == '':
        return ['']

    if iter_first and len(w) > 4:

        if w[-2:] in ["'s","’s"]:
            if w[:-2] in X.keys(): 
                return [w[:-2],w[-2:]]
            else:
                return w_recursive(w[:-2],X, iter_first=True)+[w[-2:]]

        if len(w) > 6:

            if w[-1:] =='s' and w[:-1] in X.keys():
                return [w[:-1],w[-1:]] 

            if w[-2:] =='ly' and w[:-2] in X.keys():
                return [w[:-2],w[-2:]]

            elif len(w) >6 and w[-3:] in ['ion', 'ing','ers'] and w[:-3] in X.keys():
                return [w[:-3],w[-3:]]         

            elif len(w) >7 and w[-4:] in ['ness','ment','ions','ings'] and w[:-4] in X.keys():
                return [w[:-4],w[-4:]]            

            elif len(w) > 8 and w[-5:] in ['ation','ments'] and w[:-5] in X.keys():
                return [w[:-5],w[-5:]]   

            elif len(w) > 9 and w[-6:] in ['ations','nesses'] and w[:-6] in X.keys():
                return [w[:-6],w[-6:]]                

            elif w[-1] =="s" and w[-2:] not in ["es","ss"] and w[:-1] in X.keys():
                return [w[:-1],w[-1]]            

            elif w[-2:] in ['ed','es','er'] and w[:-2] in X.keys():
                return [w[:-2],w[-2:]]

    if iter_first:     #6.20 updated
        for l in reversed(range(len(w))):
            if w[:l] in X.keys():    # or w[i:i+l+1].lower() in X.keys():
                char_last = w_recursive(w[l:],X)    #,re_class=r1)
                char_mid = w[:l]     # if w[i:i+l+1] in X.keys() else w[i:i+l+1].lower()

                return [char_mid]+char_last            

    for l in reversed(range(len(w))):
        for i in range(len(w)-l):
            if w[i:i+l+1] in X.keys():    # or w[i:i+l+1].lower() in X.keys():
                char_first = [''] if i == 0 else w_recursive(w[:i],X)   #,re_class=r1)
                char_last = [''] if i+l+1 == len(w) else w_recursive(w[i+l+1:],X)    #,re_class=r1)
                char_mid = w[i:i+l+1]     # if w[i:i+l+1] in X.keys() else w[i:i+l+1].lower()

                return char_first+[char_mid]+char_last

    return [w] # ['<unk>']  #6.20 changed

    


def preproc_num(X):

    p1 = re.compile('\(\s*\)')
    p2 = re.compile('\s+')
    q1 = re.compile('(?P<nA>[A-Za-z])')
    q2 = re.compile('[A-Za-z][a-z]+')
    q3 = re.compile('(?P<n1>[0-9])(?P<n2>[0-9][0-9])(?P<n3>[^0-9])')
    q4 = re.compile('\.\s*(?P<n4>[0-9][0-9])(?P<n5>[0-9])')
    q6 = re.compile('(?P<n6>[0-9])\s(?P<n7>[0-9])')
    q7 = re.compile('(?P<n8>[0-9])\s*(?P<n10>[\.\˅])\s*(?P<n9>[0-9])')

    q8 = re.compile('\<\s*(?P<n11>[0-9][0-9\s*\.]*[0-9\s]*)\>')
    q9 = re.compile('\(\s*\)')   # 빈괄호
    q10 = re.compile('\([\s\_]*\,')


    X = [p2.sub(' ',q3.sub('\g<n1> \g<n2> \g<n3>',q3.sub('\g<n1> \g<n2> \g<n3>',q3.sub('\g<n1> \g<n2> \g<n3>',
                            q7.sub('\g<n8> \g<n10> \g<n9>',q4.sub('. \g<n4> \g<n5>',
                            q6.sub('\g<n6>\g<n7>',s))))))) for s in X]

    #X = p2.sub(' ',q8.sub(' \g<n9> ',X)).strip()
    
    return X

def json_read(f_name):
    import json
    with open(f_name) as f:
        return json.load(f)
    
class Pre_en(object):
    
    def __init__(self):
        self.vocs = json_read("trns/NMT/Data/en_vocabs_to_apply_0615.json")
    
    def forward(self, X): 
        return preproc_en(X,self.vocs)


"""    
def pre_en(X): 
    #vocs = json_read("trns/Data/en_vocabs_0409.json")
    vocs = json_read("trns/NMT/Data/en_vocabs_to_apply_0615.json")
    X = preproc_en(X,vocs)
    
    return X
"""

def save_en_sents(sentences,fn,save_op):
    
    sentences = preproc_num(sentences)   
    to_save = '\n'.join(sentences)   

    with open(fn,save_op) as f:
        f.write(to_save+'\n')  
    
    return to_save