import sys
sys.path.append("..")
from dmsc_1nn.trainer import *
from data.data_util import *
import shutil
import time
import jieba
import os
import pandas
from random import choice
class Predictor:
    def __init__(self):
        self.vector_len=0
        self.syspath=get_data_path()
        if not os.path.exists(self.syspath+r"/online"):
            #os.makedirs(self.syspath+r'/online')
            shutil.copytree(self.syspath+'/train',self.syspath+'/online')
        self.stops=stop_set(self.syspath+r"/online/stop_words")
        self.trainer = word_frq(self.syspath+r'/online')
        self.loss=0
        self.num=0

    def get_score(self,words,model):
        if self.vector_len==0:
            sc=0
            for word in words:
                if word in model.index:
                    sc+=model.loc[word]['freq']
            return sc
        else:
            model.sort_values('freq', ascending=False, inplace=True)
            model = model[:self.vector_len]
            sc=0
            for word in words:
                if word in model.index:
                    sc+=1
            return sc

    def predect(self,comment,film):
        film_path=self.syspath+"/online/"+film
        words=jieba.lcut(comment)
        scores=[0,0,0,0,0]
        if os.path.exists(film_path):
            for i in range(5):
                model=pandas.read_csv(film_path+r'/{}.csv'.format(i),index_col='Word', encoding='utf-8')
                scores[i]=self.get_score(words,model)
        max_score=max(scores)
        for i in range(5):
            if scores[i]==max_score:
                model = pandas.read_csv(self.syspath + r'/online/{}.csv'.format(i), index_col='Word', encoding='utf-8')
                scores[i] = self.get_score(words, model)
            else:
                scores[i]=-1
        max_score = max(scores)
        max_stars=[]
        for i in range(5):
            if max_score==scores[i]:
                max_stars.append(i)
        if len(max_stars)==1:
            return max_stars[0]+1
        else:
            #print(comment,film)
            return choice(max_stars)+1

    def receive_data(self,comment,film,star=0,like=0,update=True):
        '''
        此处star是[1,5]
        :param comment:
        :param film:
        :param star:
        :return:
        '''
        pre_star=self.predect(comment,film)
        if star!=0:
            if update:
                self.trainer.add_comment(comment,film,star)
                for i in range(like):
                    self.trainer.add_comment(comment, film, star)
            loss=pre_star-star
            self.num+=1
            self.loss+=loss*loss

    def cal_MSE(self):
        if self.num!=0:
            return self.loss/self.num
        else:
            return -1

    def end(self):
        self.trainer.end()

if __name__ == '__main__':
    test_data=pandas.read_csv(get_data_path()+r'/rest_data.csv')
    print(time.strftime("%H:%M:%S", time.localtime()))
    print("load_data complete")
    predictor=Predictor()
    idxs=[]
    MSEs=[]
    print(time.strftime("%H:%M:%S", time.localtime()))
    print("start:")
    for idx,row in test_data.iterrows():
        predictor.receive_data(row['Comment'],row['Movie_Name_EN'],row['Star'],row['Like'])
        if idx%100000==0 and idx!=0:
            idxs.append(idx)
            mse=predictor.cal_MSE()
            MSEs.append(mse)
            print(time.strftime("%H:%M:%S", time.localtime()))
            print("{}:{}\n".format(idx,mse))
    print(predictor.cal_MSE())
    predictor.end()

    with open("MSE.txt",'w') as f:
        f.write(idxs)
        f.write('\n')
        f.write(MSEs)