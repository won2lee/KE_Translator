#updated0620 #updated 0618

import re

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

    #r1 = 'first' if (re_class=='first' or iter_first) else 'mid'
    #r2 = 'last' if (re_class=='last' or iter_first) else 'mid' 

    if iter_first:   #6.20 updated
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
    
    return [w]  #['<unk>']

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

    #X = p2.sub(' ',q8.sub(' \g<n9> ',X)).strip()
    
    return X

def get_tag(ws, ut, ui): 
    
    if ui.match(ws) is not None:
        wtag = chr(96) #initial
    elif ut.match(ws) is not None:
        wtag = '^' #upper
    else:
        wtag = '_' #lower
    return [wtag]


def save_sents(sentences,en_vocs):
    
    p = re.compile(r'\.$')
    q = re.compile(r'\s+')
    z = re.compile(r'__') #영어로 번역할 때는 '_' 무시
    u = re.compile(r'(?P<to_fix>[0-9]+)')

    q1 = re.compile('[A-Za-z]')
    #q2 = re.compile('(?P<c1>[A-Z])(?P<c2>[A-Z])')  
    
    ut = re.compile('[0-9]*[A-Z][a-z]*[0-9]*')
    ui = re.compile('[0-9]*[A-Z][A-Z]+[0-9]*')
    uc = re.compile('_+\s+(?P<tag>[`^_])')
    uw = re.compile('_\s+_')

    p1 = re.compile('(?P<to_fix>[^A-Za-z0-9ㄱ-ㅎㅏ-ㅣ가-힣\'\-\.\˅\`\^\_\s])')
    q7 = re.compile("\.\s*$")
    q2 = re.compile("\s+\'")
    q3 = re.compile("\'\s+")
    q4 = re.compile("\.\s*\'")
    q5 = re.compile('\.\s*\"')
    q6 = re.compile('\s+')

    z4 = re.compile('(?P<num2>[0-9])\s*\,\s*(?P<num3>[0-9][0-9][0-9])')    
      
    sents = []
    for s in sentences:
        s = [''.join([c for c in w if ord(c) < 55204]) for w in s ]
        s = [w if q1.match(w) is None else ' '.join(
            get_tag(w, ut, ui) + w_recursive(w.lower(), en_vocs, iter_first=True)) for w in s]  
        
        s = p1.sub('_ \g<to_fix> _', z4.sub('\g<num2>˅\g<num3>',
                                            z4.sub('\g<num2>˅\g<num3>',(q7.sub('_ .',' '.join(s))))))           
        s = q6.sub(' ',q5.sub('_ . _ " _', q4.sub("_ . _ ' _",q3.sub("_ ' _",q2.sub("_ ' _",s)))))
        s = q.sub(' ',u.sub(' \g<to_fix> ',z.sub('_',uc.sub('\g<tag>',uw.sub('_',uw.sub('_',s))))))
        s = '_ ' + s
        
        sents.append(s)
        
        #sents.append(z.sub(' ',q.sub(' ',p.sub(' .',' '.join(s)))))
        
    to_save = sents
    #sents = preproc_num(sents) 
        
    #to_save = '\n'.join(sents)   

    #with open(fn,save_op) as f:
    #    f.write(to_save+'\n')  
    
    return to_save
