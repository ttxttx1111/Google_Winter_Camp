import pandas
import jieba
def filter_comment_by_like(data,min_like=10):
    return data[data['Like']>min_like]

def filter_comment_by_lenth(data,min_len=50):
    return data[data['Comment'].map(len)>min_len]

def group_by_user(data):
    return data.groupby('Movie_Name_CN')['Comment'].size()

def filter_user_group(raw_data,min_films=5):
    """

    :param raw_data:
    :param min_films: 每个用户最少看过的电影
    :return:
    """
    user_data = group_by_user(raw_data)
    return user_data[user_data >= min_films]

def stop_set():
    stops=[]
    with open('stop_words','r',encoding='utf-8') as f:
        for line in f.readlines():
            line=line.strip()
            stops.append(line)
    return stops

def gen_dict(raw_data):
    stops=stop_set()
    word_map = {}
    for idx, row in raw_data.iterrows():
        buf = row['Comment']
        word_list = jieba.lcut(buf)
        for word in word_list:
            if '\u4e00' <= word <= '\u9fff' and stops.count(word)<1:
                if word_map.get(word):
                    word_map[word] += 1
                else:
                    word_map[word] = 1
    words = sorted(word_map.items(), key=lambda x: x[1], reverse=True)
    with open('voc', 'w', encoding='utf-8') as f:
        for word in words:
            if word[1]<=100:
                break
            f.write('{} {}\n'.format(word[0], word[1]))

def re_gen_data():
    raw_data = pandas.read_csv('DMSC.csv', encoding='utf-8', sep=',')
    raw_data.drop(['ID','Movie_Name_EN','Crawl_Date','Number','Date'],axis=1,inplace=True)
    raw_data.to_csv('Data.csv',index=False,encoding='utf-8')

def group_by_stars(raw_data):
    raw_data.drop(['Movie_Name_CN','Username','Like'],axis=1)
    for star in range(1,6,1):
        star_data=raw_data[raw_data['Star']==star]
        mp = {}
        for idx,row in star_data.iterrows():
            buf=row['Comment']
            if mp.get(len(buf)):
                mp[len(buf)]+=1
            else:
                mp[len(buf)]=1
        with open('star'+str(star),'w',encoding='utf-8') as f:
            sorted_mp=sorted(mp.items(),key=lambda x:x[0])
            for length in sorted_mp:
                f.write("{} {}\n".format(length[0],length[1]))

def get_score(comment,star_key_words):
    words=[]
    sim=0
    for key_word in star_key_words:
        if comment.find(key_word)>0:
            words.append(1)
            sim+=1
        else:
            words.append(0)
    return sim

def get_all_key_words():
    key_words = []
    for i in range(1, 6, 1):
        with open("keyword" + str(i), 'r', encoding='utf-8') as f:
            line = f.readline()
        key_words.append(line.split(','))
    return key_words

def get_sample():
    raw_data = pandas.read_csv('Data.csv', encoding='utf-8', sep=',')
    raw_data.sample(n=10,replace=True)
    return raw_data['Comment'].tolist()

def classify():
    key_words=get_all_key_words()
    samples=get_sample()
    results=[]
    for sample in samples:
        mx=0
        max_sc=0
        for i in range(5):
            sc=get_score(sample,key_words[i])
            if sc>max_sc:
                mx=i+1
                max_sc=sc
        results.append(mx)
    return results

def draw_star():
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(10,6))
    colors=['red','green','blue','orange','black']
    for i in range(5):
        data=pandas.read_csv('star'+str(i+1),header=None,names=['length','number'],sep=' ')
        plt.plot(data['length'],data['number'],c=colors[i],label='star'+str(i+1))
    plt.legend(loc='upper right')
    plt.show()

def get_key_words_by_star(raw_data):
    import jieba.analyse
    raw_data.drop(['Movie_Name_CN', 'Username', 'Like'], axis=1)
    jieba.analyse.set_stop_words("stop_words")
    for star in [5,1,3,2,4]:
        star_data = raw_data[raw_data['Star'] == star]
        buf=""
        for idx,row in star_data.iterrows():
            buf+=row['Comment']
        tags=jieba.analyse.extract_tags(buf,topK=60)
        with open("keyword"+str(star),'w',encoding='utf-8') as f:
            f.write(",".join(tags))

if __name__ == '__main__':
    #raw_data=pandas.read_csv('Data.csv',encoding='utf-8',sep=',')
    #gen_dict(raw_data)
    #raw_data=raw_data.head(10)
    #print(filter_comment_by_like(raw_data).shape[0])
    print(classify())
    #draw_star()
    #print(group_by_user(raw_data))