# Google_Winter_Camp
# 懂你
1. 在豆瓣影评数据集上，使用中文bert模型进行sequence regression的微调，在测试集上达到了0.74的RMSE。

    一般而言，bert训练参数中的sequence长度为输入数据中最大值。
    但当数据训练量较大且数据长度不均时，可以将数据按长度进行简单划分，分别输入对应sequence长度的模型进行训练，可以节约大量训练时间。
    比如，200w条sequnce最大长度为200的数据在v100上需要48小时，而取出其中100w条长度小于30的数据进行训练，则只需要6小时。
    同时，可能在短数据上获得性能的提升。
    
    该部分感谢 Google AI Language 和 hugging face inc 提供的数据和基础代码

    https://github.com/huggingface/pytorch-pretrained-BERT
  
    https://github.com/google-research/bert
    
   
2. 同时，由于数据集中只用28部电影，数量和种类都存在缺陷，所以提供一个基于统计学的增量方法，该方法利用电影历史影评和当前影评进行评分的分类预测，使用的思路大致为tf_idf + knn。测试中，面对新电影时的RMSE为1.34, 随着电影影评的增加，RMSE会逐渐降低至1.1，证明其确实学习到了新电影影评的模式
3. 完整的前端后台对接，可以直接在网页上输入评论，会获得两个系统给出的评分


