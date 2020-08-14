---
Word2Vec学习
---

Word2Vec模型是Google公司在2013年开源的一种将词语转化为向量表示的模型。

word2vec作为神经概率语言模型的输入，其本身其实是神经概率模型的副产品，是为了通过神经网络学习某个语言模型而产生的中间结果。具体来说，“某个语言模型”指的是“**CBOW**”和“**Skip-gram**”。具体学习过程会用到两个降低复杂度的近似方法——**Hierarchical Softmax**或**Negative Sampling**。两个模型乘以两种方法，一共有**四种**实现。


## CBOW

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217105702468.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)

CBOW 是 Continuous Bag-of-Words Model 的缩写，是一种根据上下文的词语预测当前词语的出现概率的模型。其图示如上图。

CBOW是已知上下文，估算当前词语的语言模型。其学习目标是最大化对数似然函数：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217105736924.png)

其中，w表示语料库C中任意一个词。从上图可以看出，对于CBOW，

**输入层**是上下文的词语的词向量（什么！我们不是在训练词向量吗？不不不，我们是在训练CBOW模型，词向量只是个副产品，确切来说，是CBOW模型的一个参数。训练开始的时候，词向量是个随机值，随着训练的进行不断被更新）。

**投影层**对其求和，所谓求和，就是简单的向量加法。

**输出层**输出最可能的w。由于语料库中词汇量是固定的|C|个，所以上述过程其实可以看做一个多分类问题。给定特征，从|C|个分类中挑一个。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217104423784.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)

上图输出层的树形结构即为Hierarchical Softmax。

非叶子节点相当于一个神经元（感知机，我认为逻辑斯谛回归就是感知机的输出代入f(x)=1/(1+e^x)），二分类决策输出1或0，分别代表向下左转或向下右转；每个叶子节点代表语料库中的一个词语，于是每个词语都可以被01唯一地编码，并且其编码序列对应一个事件序列，于是我们可以计算条件概率p(w|Context(w))。

在开始计算之前，引入一些符号：
1.$p^w$从根结点出发到达w对应叶子结点的路径.

2.$l^w$路径中包含结点的个数

3.$p^w_{1}$,$p^w_{2}$,...,$p^w_{l^w}$路径$p^w$中的各个节点

4.$d^w_{2}$,$d^w_{3}$,...,$d^w_{l^w}$$\in${0,1}词w的编码，$d^w_{j}$表示路径$p^w$第j个节点对应的编码（根节点无编码）

5.$θ^w_{1}$,$θ^w_{2}$,...,$θ^w_{l^w-1}$$\in${0,1}路径$p^w$中非叶节点对应的参数向量

于是可以给出w的条件概率：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217111109707.png)

这是个简单明了的式子，从根节点到叶节点经过了屏幕快照 2016-07-17 上午10.00.06.png-1个节点，编码从下标2开始（根节点无编码），对应的参数向量下标从1开始（根节点为1）。

其中，每一项是一个逻辑斯谛回归：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217111238209.png)

考虑到d只有0和1两种取值，我们可以用指数形式方便地将其写到一起：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217111247369.png)

我们的目标函数取对数似然：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217111325614.png)
将p(w|Context(w))代入上式，有
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217111351347.png)
这也很直白，连乘的对数换成求和。不过还是有点长，我们把每一项简记为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217111420696.png)
怎么最大化对数似然函数呢？分别最大化每一项即可（这应该是一种近似，最大化某一项不一定使整体增大，具体收敛的证明还不清楚）。怎么最大化每一项呢？先求函数对每个变量的偏导数，对每一个样本，代入偏导数表达式得到函数在该维度的增长梯度，然后让对应参数加上这个梯度，函数在这个维度上就增长了。这种白话描述的算法在学术上叫随机梯度上升法，详见[更规范的描述](https://www.hankcs.com/ml/the-logistic-regression-and-the-maximum-entropy-model.html#h3-6)。


每一项有两个参数，一个是每个节点的参数向量$θ^w_{j-1}$，另一个是输出层的输入$x_{w}$，我们分别对其求偏导数：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217111719449.png)
因为sigmoid函数的导数有个很棒的形式：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217111936832.png)
于是代入上上式得到：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217111953874.png)
合并同类型得到：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217112009696.png)
于是$θ^w_{j-1}$的更新表达式就得到了：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217112020410.png)

其中，$\eta$是机器学习的老相好——学习率，通常取0-1之间的一个值。学习率越大训练速度越快，但目标函数容易在局部区域来回抖动。


在L(w,j)中$x_w$和$θ^w_{j-1}$是对称的，故得到$x_w$的更新表达式：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217112512946.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217112522398.png)
其中，v($\tilde{w}$)代表上下文中某一个单词的词向量。




## Skip-gram
Skip-gram模型同样是一个三层神经网络. skip-gram模型的结构与CBOW模型正好相反，skip-gram模型输入某个单词输出对它上下文词向量的预测。

输入一个单词, 输出对上下文的预测。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217104746884.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)

Skip-gram的核心同样是一个哈夫曼树, 每一个单词从树根开始到达叶节点可以预测出它上下文中的一个单词。

上图与CBOW的两个不同在于：

1. 输入层不再是多个词向量，而是一个词向量

2. 投影层其实什么事情都没干，直接将输入层的词向量传递给输出层

在对其推导之前需要引入一个新的记号：

u：表示w的上下文中的一个词语。

于是语言模型的概率函数可以写作：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217112836354.png)

注意这是一个词袋模型，所以每个u是无序的，或者说，互相独立的。

在Hierarchical Softmax思想下，每个u都可以编码为一条01路径：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217112906223.png)
类似的，每一项都是如下简写：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217112915466.png)
把他们写到一起，得到目标函数：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217121641361.png)
类似CBOW的做法，将每一项简记为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217121654835.png)

虽然上式对比CBOW多了一个u，但给定训练实例（一个词w和它的上下文{u}），u也是固定的。所以上式其实依然只有两个变量$x_w$和$θ^w_{j-1}$，对其求偏导数：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217121912130.png)
具体求导过程类似CBOW，略过。

于是得到$θ^w_{j-1}$的更新表达式：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217121920798.png)
同理利用对称性得到对$x_w$的偏导数：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217121935312.png)
于是得到$x_w$的更新表达式：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217121959679.png)


无论是CBOW还是Skip-gram模型，其实都是分类模型。对于机器学习中的分类任务，在训练的时候不但要给正例，还要给负例。对于Hierarchical Softmax，负例是二叉树的其他路径。对于Negative Sampling，负例是随机挑选出来的。

## word2vec词向量与传统的one-hot词向量相比，主要有以下两个优势

**1.低维稠密**

一般来说分布式词向量的维度设置成100-500就足够使用，而one-hot类型的词向量维度与词表的大小成正比，是一种**高维稀疏**的表示方法，这种表示方法导致其在计算上具有比较低效率。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191219094917501.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)

**2.蕴含语义信息**

one-hot这种表示方式使得每一个词映射到高维空间中都是互相正交的，也就是说one-hot向量空间中词与词之间没有任何关联关系，这显然与实际情况不符合，因为实际中词与词之间有近义、反义等多种关系。Word2vec虽然学习不到反义这种高层次语义信息，但它巧妙的运用了一种思想：“**具有相同上下文的词语包含相似的语义**”，使得语义相近的词在映射到欧式空间后中具有较高的余弦相似度。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191219095007318.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191219095017422.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)



## 其他资源

1.论文推荐：[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)

2.CS224N笔记：词向量的表示：[word2vec](https://www.hankcs.com/nlp/word-vector-representations-word2vec.html)

3.[word2vec源码解析](https://blog.csdn.net/google19890102/article/details/51887344)

4.[Python版本Word2vec和Doc2vec](https://radimrehurek.com/gensim/models/word2vec.html)

5.[漫谈Word2vec之skip-gram模型](https://mp.weixin.qq.com/s/reT4lAjwo4fHV4ctR9zbxQ)



