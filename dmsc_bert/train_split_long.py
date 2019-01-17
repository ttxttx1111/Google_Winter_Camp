import pandas as pd
basedir = "/home/ttxttx1111/Google_Winter_Camp/dmsc_bert/"
# basedir = "/home/ttx/Google_Winter_Camp/dmsc_bert/"

train_full = pd.read_csv("input/DMSC.csv")
train_full.head()


train_new = train_full[train_full["Comment"].apply(lambda x:len(x) > 100)]

train = train_new.iloc[:int(train_new.shape[0]*80/100),:]
pd.DataFrame.to_csv(train, basedir + "/input/train.csv",index=False)

val = train_new.iloc[int(train_new.shape[0]*80/100):int(train_new.shape[0]*90/100),:]
pd.DataFrame.to_csv(val, basedir + "/input/val.csv",index=False)


test = train_new.iloc[int(train_new.shape[0]*90/100):,:]
pd.DataFrame.to_csv(test, basedir + "/input/test.csv",index=False)

