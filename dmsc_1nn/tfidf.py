import math
class tf_idf():
    def __init__(self,cur_times,all_times,cur_len,all_len):
        '''
        需要word在当前文本和全体文本出现次数，当前文本字数，文本数
        全体文本默认该电影或该星级，
        '''
        self.cur_times=cur_times
        self.all_times=all_times
        self.cur_len=cur_len
        self.all_len=all_len

    def tf(self):
        '''
        计算当前文本word出现次数/文本字数
        :param word:
        :param count:
        :return:count[word] / sum(count.values())
        '''
        return self.cur_len/self.cur_len

    def n_containing(self):
        """
        计算word在全体中出现次数
        :param word:
        :param count_list:
        :return:sum(1 for count in count_list if word in count)
        """
        return self.all_times

    def idf(self):
        '''
        计算 句子数/....
        :param word:
        :param count_list:
        :return:
        '''
        return math.log(self.all_len) / (1 + self.n_containing())

    def tfidf(self):
        return self.tf() * self.idf()
