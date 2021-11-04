#app.py
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

from app_utils import to_start, rid_blank, preproc_num, to_normal
from trns.get_model import trns_model
import random

from multiprocessing import Pool
from itertools import chain
import time

pre_en, pre_ko, trns = trns_model()

def mp(X):
    X = pre_en.forward(X) #for x in X]
    X = preproc_num(X) #for x in X]
    X = [s.split(' ') for s in X if s.strip() !="''"] #for x in X] #'"'
    X = trns.translate(X,'ko') 
    return X

def mpko(X):
    X = pre_ko.forward(X) #for x in X]
    X = preproc_num(X) #for x in X]
    X = [s.split(' ') for s in X if s.strip() !="''"] #for x in X]
    X = trns.translate(X,'en') 
    return X

def nmt(X, to_start, pre_ko, pre_en, trns, Pool):

    enko_count = sum([1 if ord(c) in range(65,123) else -1 for c in X])
    
    XX = to_start(X)
    XX = 'Æ'.join(XX) #.split('Ë')
    
    if enko_count > 0:
        """
        X = XX.split('Æ') #for x in XX]
        no_split = str(len(X))
        start = time.time()

        X = pre_en.forward(X) #for x in X]
        X = preproc_num(X) #for x in X]
        X = [s.split(' ') for s in X if s.strip() !="''"] #for x in X] #'"'
        X = trns.translate(X,'ko')         
        """
        XX = [[x] for x in XX.split('Æ')] #or x in XX] 
        no_split = str(len(XX))
        start = time.time()
        
        with Pool(2) as p:
            X = p.map(mp, XX)
        #X = map(mp, XX)
        X = list(chain(*X))
        
        tt = time.time() - start
        
        X = to_normal(X) #for x in X]
        """
        if x == 'Ë':
            X = '\n\n'
        else:
            X = [x]           
            X = pre_en.forward(X)
            X = preproc_num(X)
            X = [s.split(' ') for s in X if s.strip() !="''"] #'"'
            X = trns.translate(X,'ko')
            X = to_normal(X)
        """
        Xout = X.strip() #for x in X]
        #Xout = no_split + str(tt) + X.strip() #for x in X]
        #nmt_sub(Xout=Xout)
        #redirect(url_for('nmt_sub',xout=Xout))

        #render_template('nmt_sub',tested=Xout)
        
    else:
        
        """
        X = XX.split('Æ') #for x in XX]
        
        start = time.time() 
        X = pre_ko.forward(X) #for x in X]
        X = preproc_num(X) #for x in X]
        X = [s.split(' ') for s in X if s.strip() !="''"] #for x in X]
        X = trns.translate(X,'en') 
        """
        XX = [[x] for x in XX.split('Æ')] #or x in XX]
        start = time.time()
        with Pool(2) as p:
            X = p.map(mpko, XX)
        #X = map(mp, XX)
        X = list(chain(*X))
        
        tt = time.time() - start
        
        X = rid_blank(X) #for x in X]            
        """
        if x == 'Ë':
            X = '\n\n'
        else:
            X = [x]
            X = pre_ko.forward(X)
            X = preproc_num(X)
            X = [s.split(' ') for s in X if s.strip() !="''"]
            X = trns.translate(X,'en')
            X = rid_blank(X)  
        """
        Xout = X.strip() #for x in X]
        #Xout = str(tt) + X.strip() #for x in X]
        #return nmt_sub(Xout=Xout)
        #redirect(url_for('nmt_sub',xout=Xout))
        #render_template('nmt_sub',tested=Xout)

    return Xout #'\n\n'.join(Xout)
    #return '\n\n'.join([s.strip() for s in ' '.join(Xout).split('\n')])
    #return ' '.join(XX) + Xout.strip()

@app.route('/')
def root():    
    X = ["<Example 1>\n\nJoseph Robinette Biden Jr. (born November 20, 1942) is an American politician and the president-elect of the United States. Having defeated incumbent Donald Trump in the 2020 United States presidential election, he will be inaugurated as the 46th president on January 20, 2021. A member of the Democratic Party, Biden served as the 47th vice president from 2009 to 2017 and a United States senator for Delaware from 1973 to 2009.", "<Example 2>\n\nThe number of Korean domestic tourists has increased since 2010. The number of people who participated in domestic travel (which includes one-day trips) was about 238.3 million (in 2015). It increased by 4.9% compared to 2014 (227.1 million).[7] In 2014, Korean's domestic tourism expenditure was ₩14.4 trillion.[8]\n\nAlso, Korean oversea tourists keep increasing since 2010. From 2012 to 2014, the number of people travelling overseas has risen by about 8.2% on average. In 2014, number of Korean oversea tourists was about 16.1 million.", 'Example 3.\n\n조 바이든 미국 대통령 당선인은 16일(현지시간) 미국 내 신종 코로나바이러스 감염증(코로나19) 재확산과 관련해 "지금 조율하지 않으면 더 많은 사람이 죽을 것"이라며 도널드 트럼프 대통령이 정권 인수·인계 작업에 협력할 것을 촉구했다.\n\n바이든 당선인은 이날 자택이 있는 델라웨어주 윌밍턴에서 카멀라 해리 부통령 당선인과 함께 당선 후 처음으로 경제 구상을 밝히는 기자회견을 열었다.\n\n그는 이 자리에서 경제적 어려움에 부닥친 근로자와 기업, 주 정부를 지원할 수 있도록 의회가 코로나19 경기 부양 패키지를 신속하게 통과시켜야 한다고 말했다.', "네이버 뉴스, 위키피디아 등 복사해서 여기에 붙이고 아래 translate 버튼 누르면 됩니다.\n물론 문장을 직접 타이프해도 됩니다.\n\n한국 뉴스와 위키피디아 문장으로 주로 학습을 했기 때문에 의문문과 명령문 번역이 약합니다.\n\n가끔 어색한 번역들도 나올 수 있습니다."]
    ix = random.randint(0,len(X)-1)
    return render_template('nmt.html', to_test = X[ix]) #'Welcome to Translation!'

#@main.route('/test')
#@app.route('/nmt')
#def test():
#    return render_template('nmt.html')

#@app.route('/nmt', methods=['POST'])
def nmt_sub(Xout=None):
    Xs = request.form['nmt']
    return render_template('nmt.html', to_test = Xs, tested = Xout)

@app.route('/nmt', methods=['POST'])
def post():
    X = request.form['nmt']
    Y = nmt(X, to_start, pre_ko, pre_en, trns, Pool)
    return render_template('nmt.html',to_test = X, tested = Y)



if __name__ == '__main__':
    app.run(debug=True)
