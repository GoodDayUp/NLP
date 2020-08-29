---
GloVe学习：Global Vectors for Word Representation
---

[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)


## 什么是GloVe？
正如[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)论文而言，GloVe的全称叫Global Vectors for Word Representation，它是一个基于**全局词频统计**（count-based & overall statistics）的词表征（word representation）工具，它可以把一个单词表达成一个由实数组成的向量，这些向量捕捉到了单词之间一些语义特性，比如相似性（similarity）、类比性（analogy）等。我们通过对向量的运算，比如欧几里得距离或者cosine相似度，可以计算出两个单词之间的语义相似性。

- 模型目标：进行词的向量化表示，使得向量之间尽可能多地蕴含语义和语法的信息。

- 输入：语料库

- 输出：词向量

- 方法概述：首先基于语料库构建词的共现矩阵，然后基于共现矩阵和GloVe模型学习词向量。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191218201803776.png)

## 统计共现矩阵

设共现矩阵为$X$，其元素为$X_{i,j}$。 
$X_{i,j}$的意义为：在整个语料库中，单词i和单词j共同出现在一个窗口中的次数。 

举个例子： 
设有语料库：

```
i love you but you love him i am sad
```

这个小小的语料库只有1个句子，涉及到7个单词：i、love、you、but、him、am、sad。 
如果我们采用一个窗口宽度为5（左右长度都为2）的统计窗口，那么就有以下窗口内容：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191218202002648.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)

窗口0、1长度小于5是因为中心词左侧内容少于2个，同理窗口8、9长度也小于5。 
以窗口5为例说明如何构造共现矩阵： 
中心词为love，语境词为but、you、him、i；则执行： 

$X_{love,but} +=1$
$X_{love,you} +=1$
$X_{love,him} +=1$
$X_{love,i} +=1$

使用窗口将整个语料库遍历一遍，即可得到共现矩阵$X$。


## GloVe是如何实现的？

GloVe的实现分为以下三步：

 - 根据语料库（corpus）构建一个共现矩阵（Co-ocurrence Matrix）$X$（什么是共现矩阵？），**矩阵中的每一个元素$X_{ij}$代表单词和上下文单词$j$在特定大小的上下文窗口（context window）内共同出现的次数**。一般而言，这个次数的最小单位是1，但是GloVe不这么认为：它根据两个单词在上下文窗口的距离$d$，提出了一个衰减函数（decreasing weighting）：$deacy=1/d$ 用于计算权重，也就是说**距离越远的两个单词所占总计数（total count）的权重越小**。

```
In all cases we use a decreasing weighting function, so that word pairs that are d words apart contribute 1/d to the total count.
```

 - 构建词向量（Word Vector）和共现矩阵（Co-ocurrence Matrix）之间的近似关系，论文的作者提出以下的公式可以近似地表达两者之间的关系：

$w^T_{i}\tilde{w}_j+b_i+b_j=log(X_{ij})$      **(1)**

其中，**$w^T_{i}$和$\tilde{w_j}$是我们最终要求解的词向量**；$b_{i}$和$\tilde{b_j}$分别是两个词向量的bias term。当然你对这个公式一定有非常多的疑问，比如它到底是怎么来的，为什么要使用这个公式，为什么要构造两个词向量$w^T_{i}$和$\tilde{w_j}$？下文我们会详细介绍。

 - 有了公式1之后我们就可以构造它的loss function了：

$J=$$\sum_{i,j=1}^V$$f(X_{ij})(w^T_{i}\tilde{w}_j+b_i+b_j-log(X_{ij}))^2$

这个loss function的基本形式就是最简单的mean square loss，只不过在此基础上加了一个权重函数$f(X_{ij})$，那么这个函数起了什么作用，为什么要添加这个函数呢？我们知道在一个语料库中，肯定存在很多单词他们在一起出现的次数是很多的（frequent co-occurrences），那么我们希望：

 
- 1.这些单词的权重要大于那些很少在一起出现的单词（rare co-occurrences），所以这个函数要是非递减函数（non-decreasing）；
- 2.但我们也不希望这个权重过大（overweighted）当到达一定程度之后应该不再增加；
- 3.如果两个单词没有在一起出现，也就是$X_{ij}=0$，那么他们应该不参与到loss function的计算当中去，也就是$f(x)$要满足$f(0)=0$

满足以上两个条件的函数有很多，作者采用了如下形式的分段函数：

$$f(x)=\begin{cases} (x/x_{max})^α,&\text{if $x<x_{max}$ }\\1,&\text{otherwise}\end{cases}$$   

这个函数图像如下所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191218192912780.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)

这篇论文中的所有实验，α的取值都是0.75，而$x_{max}$取值都是100。以上就是GloVe的实现细节，那么GloVe是如何训练的呢？


## GloVe是如何训练的？

虽然很多人声称GloVe是一种无监督（unsupervised learing）的学习方式（因为它确实不需要人工标注label），但其实它还是有label的，这个label就是公式2中的log($X_{ij}$)，而公式2中的向量$w$和$\tilde{w}$就是要不断更新/学习的参数，所以本质上它的训练方式跟监督学习的训练方法没什么不一样，都是基于梯度下降的。具体地，这篇论文里的实验是这么做的：**采用了AdaGrad的梯度下降算法，对矩阵$X$中的所有非零元素进行随机采样，学习曲率（learning rate）设为0.05，在vector size小于300的情况下迭代了50次，其他大小的vectors上迭代了100次，直至收敛**。最终学习得到的是两个vector是$w$和$\tilde{w}$，因为$X$是对称（symmetric），所以从原理上讲$w$和$\tilde{w}$是也是对称的，他们唯一的区别是初始化的值不一样，而导致最终的值不一样。所以这两者其实是等价的，都可以当成最终的结果来使用。**但是为了提高鲁棒性，我们最终会选择两者之和$w$ + $\tilde{w}$作为最终的vector（两者的初始化不同相当于加了不同的随机噪声，所以能提高鲁棒性）**。在训练了400亿个token组成的语料后，得到的实验结果如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191218193446715.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)

这个图一共采用了三个指标：语义准确度，语法准确度以及总体准确度。那么我们不难发现Vector Dimension在300时能达到最佳，而context Windows size大致在6到10之间。

## Glove与LSA、word2vec的比较

LSA（Latent Semantic Analysis）是一种比较早的count-based的词向量表征工具，它也是基于co-occurance matrix的，只不过采用了基于奇异值分解（SVD）的矩阵分解技术对大矩阵进行降维，而我们知道SVD的复杂度是很高的，所以它的计算代价比较大。还有一点是它对所有单词的统计权重都是一致的。而这些缺点在GloVe中被一一克服了。而word2vec最大的缺点则是没有充分利用所有的语料，所以GloVe其实是把两者的优点结合了起来。从这篇论文给出的实验结果来看，GloVe的性能是远超LSA和word2vec的，但网上也有人说GloVe和word2vec实际表现其实差不多。

## 公式推导
写到这里GloVe的内容基本就讲完了，唯一的一个疑惑就是公式1到底是怎么来的？如果你有兴趣可以继续看下去，如果没有，可以关掉浏览器窗口了。为了把这个问题说清楚，我们先定义一些变量：

- $X_{ij}$表示单词$j$出现在单词$i$的上下文中的次数；

- $X_{i}$表示单词$i$的上下文中所有单词出现的总次数，即$X_{i}=\sum^kX_{ik}$;

- $P_{ij}=P(j|i)=X_{ij}/X_{i}$即表示单词$j$出现在单词$i$的上下文中的概率；


有了这些定义之后，我们来看一个表格：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191218194252360.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)

理解这个表格的重点在最后一行，它表示的是两个概率的比值（ratio），**我们可以使用它观察出两个单词$i$和$j$相对于单词$k$哪个更相关（relevant）**。比如，ice和solid更相关，而stream和solid明显不相关，于是我们会发现$P(solid|ice)/P(solid|steam)$ 比1大得多。同样的gas和steam更相关，而和ice不相关，那么$P(solid|ice)/P(solid|steam)$ 就远小于1；当都有关（比如water）或者都没有关(fashion)的时候，两者的比例接近于1；这个是很直观的。因此，**以上推断可以说明通过概率的比例而不是概率本身去学习词向量可能是一个更恰当的方法**，因此下文所有内容都围绕这一点展开。

于是为了捕捉上面提到的概率比例，我们可以构造如下函数：

$F(w_{i},w_j,\tilde{w}_k)=P_{ik}/P_{jk}$

其中，函数$F$的参数和具体形式未定，它有三个参数$w_{i},w_j和\tilde{w}_k$，$w和\tilde{w}$是不同的向量；
因为向量空间是线性结构的，所以要表达出两个概率的比例差，最简单的办法是作差，于是我们得到：

$F(w_{i}-w_j,\tilde{w}_k)=\frac{P_{ik}}{P_{jk}}$

这时我们发现公式5的右侧是一个数量，而左侧则是一个向量，于是我们把左侧转换成两个向量的内积形式：

$F((w_{i}-w_j)^T\tilde{w}_k)=\frac{P_{ik}}{P_{jk}}$   (6)

我们知道$X$是个对称矩阵，单词和上下文单词其实是相对的，也就是如果我们做如下交换：$w$$\leftrightarrow$$\tilde{w}_k$，$X$$\leftrightarrow$$X^T$ 公式6应该保持不变，那么很显然，现在的公式是不满足的。为了满足这个条件，首先，我们要求函数$F$要满足同态特性（homomorphism）：

$F((w_{i}-w_j)^T\tilde{w}_k)=\frac{F(w^T_{i}\tilde{w}_k)}{F(w^T_{j}\tilde{w}_k)}$

结合公式6，我们可以得到：

$F(w^T_{i}\tilde{w}_k)=P_{ik}=\frac{X_{ik}}{X_i}$

然后令F = exp，于是我们有：

$w^T_{i}\tilde{w}_k=log(P_{ik})=log(X_{ik})-log(X_i)$   (9)

因为等号右侧$log(X_i)$的存在，公式9不满足对称（symmetry）的，而且这个$log(X_i)$其实是跟$k$独立的，它只跟$i$有关，于是我们可以针对$w_i$增加一个bias term $b_i$把它替换掉，于是我们有：

$w^T_{i}\tilde{w}_k+b_i=log(X_{ik})$    (10)


但是公式10还是不满足对称性，于是我们针对$w_k$增加一个bias term $b_k$，从而得到公式1的形式：

$w^T_{i}\tilde{w}_k+b_i+b_k=log(X_{ik})$    (1)

以上内容其实不能完全称之为推导，因为有很多不严谨的地方，只能说是解释作者如何一步一步构造出这个公式的，仅此而已。

## 代码实现
https://github.com/NLP-LOVE/ML-NLP/blob/master/NLP/16.3%20GloVe/GloVe.ipynb

参考：

1.论文：[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)。
2.[GloVe详解](http://www.fanyeong.com/2018/02/19/glove-in-detail/)
3.[理解GloVe模型（Global vectors for word representation）](https://blog.csdn.net/coderTC/article/details/73864097)
