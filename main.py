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

        XX = [[x] for x in XX.split('Æ')] #or x in XX] 
        no_split = str(len(XX))
        start = time.time()
        
        with Pool(2) as p:
            X = p.map(mp, XX)

        X = list(chain(*X))       
        tt = time.time() - start       
        X = to_normal(X) #for x in X]
        Xout = X.strip() #for x in X]
        
    else:

        XX = [[x] for x in XX.split('Æ')] #or x in XX]
        start = time.time()
        with Pool(2) as p:
            X = p.map(mpko, XX)

        X = list(chain(*X))       
        tt = time.time() - start      
        X = rid_blank(X) #for x in X]            
        Xout = X.strip() #for x in X]

    return Xout #'\n\n'.join(Xout)

@app.route('/')
def root():    
    X = ["네이버 뉴스, 위키피디아 등 복사해서 여기에 붙이고 아래 translate 버튼 누르면 됩니다.\n물론 문장을 직접 타이프해도 됩니다."]
    ix = random.randint(0,len(X)-1)
    return render_template('nmt.html', to_test = X[ix]) #'Welcome to Translation!'

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
