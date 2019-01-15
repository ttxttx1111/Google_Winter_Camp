import pandas as pd
# basedir = "/home/ttxttx1111/Google_Winter_Camp/dmsc_bert/"
basedir = "/home/ttx/Google_Winter_Camp/dmsc_bert/"

train_full = pd.read_csv("input/DMSC.csv")
train_full.head()


train_new = train_full[train_full["Comment"].apply(lambda x:len(x) < 30)]

train = train_new.loc[:train_full.shape[0]*80/100,:]
pd.DataFrame.to_csv(train, basedir + "/input/train.csv",index=False)

train = train_new.loc[train_full.shape[0]*80/100:train_full.shape[0]*90/100,:]
pd.DataFrame.to_csv(train, basedir + "/input/val.csv",index=False)


train = train_new.loc[train_full.shape[0]*90/100:,:]
pd.DataFrame.to_csv(train, basedir + "/input/test.csv",index=False)

