# Google_Winter_Camp
# 懂你
1. 在豆瓣影评数据集上，使用中文bert模型进行sequence regression的微调，在测试集上达到了0.74的RMSE。
该部分感谢 Google AI Language 和 hugging face inc 提供的数据和基础代码

    https://github.com/huggingface/pytorch-pretrained-BERT
  
    https://github.com/google-research/bert

2. 同时，由于数据集中只用28部电影，数量和种类都存在缺陷，所以提供一个基于统计学的增量方法，该方法利用电影历史影评和当前影评进行评分的分类预测，使用的思路大致为tf_idf + knn。测试中，面对新电影时的RMSE为1.34, 随着电影影评的增加，RMSE会逐渐降低至1.1，证明其确实学习到了新电影影评的模式
3. 完整的前端后台对接，可以直接在网页上输入评论，会获得两个系统给出的评分
4. 微调bert需要大量的计算， 因此提供一个可选择的方案为使用bert作为固定特征抽取器，可以输入文本后得到新特征再经过其他网络得到最终分数。
最为方便的bert特征抽取代码为:
https://github.com/hanxiao/bert-as-service#faq
5. 一般而言，bert训练参数中的sequence长度为输入数据中最大值。
但当数据训练量较大且数据长度不均时，可以将数据按长度进行简单划分，分别输入对应sequence长度的模型进行训练，可以节约大量训练时间。
同时，可能在短数据上获得性能的提升。

