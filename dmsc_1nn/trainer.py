import pandas
import platform
from dmsc_1nn.word_frq import word_frq
import time
import traceback


def get_data_path():
    if platform.system().find("Windows") >= 0:
        syspath = r'/Google_Winter_Camp/data'
    else:
        syspath = r'/home/smalljust19/Google_Winter_Camp/data'
    return syspath

def data_separate(syspath):
    raw_data=pandas.read_csv(syspath+r'/DMSC.csv',encoding='utf-8', sep=',')
    raw_data.drop(['ID','Movie_Name_CN','Crawl_Date','Number','Username'],axis=1,inplace=True)
    raw_data.sort_values('Date',inplace=True)
    raw_data.to_csv(syspath+'/Data_by_date.csv',index=False,encoding='utf-8')
    raw_data[:1000000].to_csv(syspath+'/train_data.csv',index=False,encoding='utf-8')
    raw_data[1000000:].to_csv(syspath + '/rest_data.csv', index=False, encoding='utf-8')
    return raw_data[:1000000]

def pre_train():
    syspath=get_data_path()
    train_data=data_separate(syspath)
    print(time.strftime("%H:%M:%S", time.localtime()))
    print("data ready")
    trainer = word_frq(syspath+r'/train')
    for idx,row in train_data.iterrows():
        try:
            weight=row['Like']
            for i in range(weight+1):
                trainer.add_comment(row['Comment'],row['Movie_Name_EN'],row['Star'])
        except Exception as e:
            print(row)
            traceback.print_exc()
            print(time.strftime("%H:%M:%S", time.localtime()))
            return
    trainer.end()
    print(time.strftime("%H:%M:%S", time.localtime()))

if __name__ == '__main__':
    #pre_train()
    syspath = get_data_path()
    data_separate(syspath)