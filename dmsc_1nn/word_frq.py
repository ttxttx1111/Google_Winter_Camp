import sys
sys.path.append("..")
import pandas
import jieba
import os
import platform
from dmsc_1nn.tfidf import tf_idf
from functools import wraps
from data.data_util import *
import time
MAX_STAR = 5


def singleton(cls):
    instances = {}

    @wraps(cls)
    def getinstance(*args, **kw):
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]

    return getinstance


@singleton
class word_frq:
    def __init__(self, syspath=""):
        self.UPDATE_THRESHOLD = 500000
        self.MIN_freq = 150
        if not syspath:
            if platform.system().find("Windows") >= 0:
                self.syspath = r'/Google_Winter_Camp/data'
            else:
                self.syspath = r'/home/smalljust19/Google_Winter_Camp/data'
        else:
            self.syspath = syspath
        self.stops = stop_set(self.syspath + r'/stop_words')
        self.tot_init()
        self.temp_init()


    def tot_init(self):
        # self.tot_film_len = {}

        self.tot_star_data = [{}, {}, {}, {}, {}]
        self.tot_star_len = [0, 0, 0, 0, 0]
        self.tot_star_film_data = [{}, {}, {}, {}, {}]
        self.tot_star_film_len = [{}, {}, {}, {}, {}]
        if os.path.exists(self.syspath):
            for i in range(MAX_STAR):
                with open(self.syspath+r'/{}'.format(i)) as f:
                    buf=f.read()
                    self.tot_star_len[i]=str(buf)
                existed_data=pandas.read_csv(self.syspath+r'/{}.csv'.format(i),index_col='Word', encoding='utf-8')
                for word in existed_data.index:
                    self.tot_star_data[i][word]=existed_data[word]['num']
            for filename in os.listdir(self.syspath):
                pathname = os.path.join(self.syspath, filename)
                if (os.path.isdir(pathname)):
                    for i in range(MAX_STAR):
                        with open(pathname + r'/{}'.format(i)) as f:
                            buf = f.read()
                            self.tot_star_film_len[i][filename] = str(buf)
                        self.tot_star_film_data[i][filename]={}
                        existed_data = pandas.read_csv(pathname + r'/{}.csv'.format(i), index_col='Word', encoding='utf-8')
                        for word in existed_data.index:
                            self.tot_star_film_data[i][filename][word] = existed_data[word]['num']

        print(time.strftime("%H:%M:%S", time.localtime()))
        print("read_saved complete")
        # self.tot_len = 0  # 总字数

    def temp_init(self):
        self.film_data = {}
        # self.film_len = {}  # 文本字数
        self.star_data = [{}, {}, {}, {}, {}]
        self.star_len = [0, 0, 0, 0, 0]
        self.star_film_data = [{}, {}, {}, {}, {}]
        self.star_film_len = [{}, {}, {}, {}, {}]
        self.temp_data = {}
        # self.temp_len = 0
        self.temp_sum = 0

    def cat_file(self, file_name):
        if os.path.exists(file_name):
            pass
        else:
            if not os.path.exists(file_name[:file_name.rfind(r'/')]):
                os.makedirs(file_name[:file_name.rfind(r'/')])
            temp = pandas.DataFrame(data=None, columns=['Word', 'num', 'freq'])
            temp.to_csv(file_name, index=False, encoding='utf-8')

    def update_file(self, file_name, data, filter, father_data, cur_len, all_len):
        local_data = pandas.read_csv(file_name, index_col='Word', encoding='utf-8')
        if filter is None:
            filter = local_data
        for word, num in data.items():
            if word in filter.index or num >= self.MIN_freq:
                if word in local_data.index:
                    local_data.loc[word]['num'] += num
                    if all_len != 0:
                        local_data.loc[word]['freq'] = tf_idf(cur_times=local_data.loc[word]['num'],
                                                              all_times=father_data.loc[word]['num'],
                                                              cur_len=cur_len,
                                                              all_len=all_len).tfidf()
                else:
                    if all_len != 0:
                        freq = tf_idf(cur_times=num,
                                      all_times=father_data.loc[word]['num'],
                                      cur_len=cur_len,
                                      all_len=all_len).tfidf()
                        local_data.loc[word] = [num, freq]
                    else:
                        local_data.loc[word] = [num, 0]

        local_data.to_csv(file_name, index='Word', encoding='utf-8')
        return local_data

    def update_data(self):

        def update(file_name, data, filter, father_data, cur_len, all_len):
            self.cat_file(file_name)
            return self.update_file(file_name, data, filter, father_data, cur_len, all_len)

        new_all_data = update(self.syspath + r'/all.csv', self.temp_data, None,
                              None, 0, 0)

        new_film_data = {}
        for film, film_data in self.film_data.items():
            new_film_data[film] = update(self.syspath + r'/' + film + r'/all.csv', film_data, new_all_data,
                                         None, 0, 0)

        for i in range(MAX_STAR):
            self.tot_star_len[i] += self.star_len[i]
            with open(self.syspath + r'/star_{}'.format(i),'w') as f:
                f.write(str(self.tot_star_len[i]))
            self.tot_star_data[i] = update(self.syspath + r'/star_{}.csv'.format(i), self.star_data[i], new_all_data,
                                           new_all_data, self.tot_star_len[i], MAX_STAR)

            for film, film_data in self.star_film_data[i].items():
                self.tot_star_film_len[i][film] = self.tot_star_film_len[i].get(film, 0) + self.star_film_len[i].get(
                    film, 0)
                with open(self.syspath + r'/' + film + r'/{}'.format(i), 'w') as f:
                    f.write(str(self.tot_star_film_len[i][film]))
                self.tot_star_film_data[i][film] = update(self.syspath + r'/' + film + r'/{}.csv'.format(i), film_data,
                                                          new_all_data, new_film_data[film],
                                                          self.tot_star_film_len[i][film], MAX_STAR)

        print(time.strftime("%H:%M:%S", time.localtime()))
        print("update on going")

    def temp_data_update(self, words, film, star):
        self.temp_sum += 1

        if not self.film_data.get(film):
            self.film_data[film] = {}
            for i in range(MAX_STAR):
                self.star_film_data[i][film] = {}
        word_dict = {}
        for word in words:
            if '\u4e00' <= word <= '\u9fff' and self.stops.count(word) < 1:
                word_dict[word] = word_dict.get(word, 0) + 1
        for word, num in word_dict.items():
            self.star_len[star] += num
            self.star_film_len[star][film] = self.star_film_len[star].get(film, 0) + num

            self.temp_data[word] = self.temp_data.get(word, 0) + num
            self.film_data[film][word] = self.film_data[film].get(word, 0) + num
            self.star_data[star][word] = self.star_data[star].get(word, 0) + num
            self.star_film_data[star][film][word] = self.star_film_data[star][film].get(word, 0) + num

    def add_comment(self, comment, film, star):
        star -= 1
        word_list = jieba.lcut(comment)
        self.temp_data_update(words=word_list, film=film, star=star)
        if self.temp_sum >= self.UPDATE_THRESHOLD:
            self.update_data()
            self.temp_init()

    def end(self):
        self.update_data()
        print(time.strftime("%H:%M:%S", time.localtime()))
        print("update complete")


if __name__ == '__main__':
    t = word_frq()
    data = [["算法bfUI欧版搜上帝防晒服", 'aaaa', 1],
            ["沙发你啥都啥佛教那是", 'aaaa', 5],
            ["撒娇地佛扫附近", "bbb", 3]]
    for d in data:
        t.add_comment(d[0], d[1], d[2])
    t.end()
