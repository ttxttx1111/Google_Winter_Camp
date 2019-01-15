# coding = utf-8

# from gensim.models.word2vec import Word2Vec
# import numpy as np
import csv
import jieba
import multiprocessing


def get_comment_list():
    with open('DMSC.csv', encoding="utf8") as f:
        reader = csv.reader(f)
        comment_list = []
        for item in list(reader):
            comment = item[8]
            comment_list.append(comment)
    return comment_list[1:]


def tokenizer(document):
    text = jieba.lcut(document.replace('\n', '').replace(' ', ''))
    return text


def create_comment_file():
    comment_list = get_comment_list()
    fp = open("data.txt", "w", encoding="utf8")
    new_comment_list = []
    for comment in comment_list:
        new_comment_list.append(tokenizer(comment))
    for new_comment in new_comment_list:
        try:
            fp.write(new_comment[0])
            for word in new_comment[1:]:
                fp.write(" " + word)
            fp.write("\n")
        except:
            print(new_comment)
    fp.close()



if __name__ == '__main__':
    create_comment_file()
