---
多领域中文分词工具：pkuseg
---


pkuseg 是基于论文[\[PKUSEG: A Toolkit for Multi-Domain Chinese Word Segmentation\]](http://cn.arxiv.org/pdf/1906.11455.pdf)的工具包。其简单易用，支持细分领域分词，有效提升了分词准确度。

## 特点

pkuseg具有如下几个特点：

1. 多领域分词。不同于以往的通用中文分词工具，此工具包同时致力于为不同领域的数据提供个性化的预训练模型。根据待分词文本的领域特点，用户可以自由地选择不同的模型。 我们目前支持了新闻领域，网络领域，医药领域，旅游领域，以及混合领域的分词预训练模型。在使用中，如果用户明确待分词的领域，可加载对应的模型进行分词。如果用户无法确定具体领域，推荐使用在混合领域上训练的通用模型。各领域分词样例可参考 [example.txt](https://github.com/lancopku/pkuseg-python/blob/master/example.txt) 。

2. 更高的分词准确率。相比于其他的分词工具包，当使用相同的训练数据和测试数据，pkuseg可以取得更高的分词准确率。

3. 支持用户自训练模型。支持用户使用全新的标注数据进行训练。

4. 支持词性标注。

## 安装方式

```
pip3 install pkuseg
```


**预训练模型**可详见[release](https://github.com/lancopku/pkuseg-python/releases)。使用时需设定"model_name"为模型文件。

## 各类分词工具包性能的对比


**细领域训练及测试结果**

以下是在不同数据集上的对比结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191223141553282.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)

**默认模型在不同领域的测试效果**

考虑到很多用户在尝试分词工具的时候，大多数时候会使用工具包自带模型测试。为了直接对比“初始”性能，我们也比较了各个工具包的默认模型在不同领域的测试效果。请注意，这样的比较只是为了说明默认情况下的效果，并不一定是公平的。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191223141738527.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)

其中，All Average 显示的是在所有测试集上 F-score 的平均。

## 参数说明与代码示例
**模型配置**
```
pkuseg.pkuseg(model_name = "default", user_dict = "default", postag = False)
	model_name		模型路径。
			        "default"，默认参数，表示使用我们预训练好的混合领域模型(仅对pip下载的用户)。
			    	"news", 使用新闻领域模型。
			    	"web", 使用网络领域模型。
			    	"medicine", 使用医药领域模型。
			    	"tourism", 使用旅游领域模型。
			         model_path, 从用户指定路径加载模型。
	user_dict	    设置用户词典。
				    "default", 默认参数，使用我们提供的词典。
				    None, 不使用词典。
					dict_path, 在使用默认词典的同时会额外使用用户自定义词典，可以填自己的用户词典的路径，词典格式为一行一个词（如果选择进行词性标注并且已知该词的词性，则在该行写下词和词性，中间用tab字符隔开）。
	postag		    是否进行词性分析。
					False, 默认参数，只进行分词，不进行词性标注。
					True, 会在分词的同时进行词性标注。
```

**对文件进行分词**

```
pkuseg.test(readFile, outputFile, model_name = "default", user_dict = "default", postag = False, nthread = 10)
	readFile		输入文件路径。
	outputFile		输出文件路径。
	model_name		模型路径。同pkuseg.pkuseg
	user_dict		设置用户词典。同pkuseg.pkuseg
	postag			设置是否开启词性分析功能。同pkuseg.pkuseg
	nthread			测试时开的进程数。
```

**模型训练**

```
pkuseg.train(trainFile, testFile, savedir, train_iter = 20, init_model = None)
	trainFile		训练文件路径。
	testFile		测试文件路径。
	savedir			训练模型的保存路径。
	train_iter		训练轮数。
	init_model		初始化模型，默认为None表示使用默认初始化，用户可以填自己想要初始化的模型的路径如init_model='./models/'。
```



代码示例1：使用默认配置进行分词（如果用户无法确定分词领域，推荐使用默认模型分词）

```
import pkuseg

seg = pkuseg.pkuseg()           # 以默认配置加载模型
text = seg.cut('我爱北京天安门')  # 进行分词
print(text)
```
代码示例2：细领域分词（如果用户明确分词领域，推荐使用细领域模型分词）
```
import pkuseg

seg = pkuseg.pkuseg(model_name='medicine')  # 程序会自动下载所对应的细领域模型
text = seg.cut('我爱北京天安门')              # 进行分词
print(text)

```

代码示例3：分词同时进行词性标注，各词性标签的详细含义可参考 [tags.txt](https://github.com/lancopku/pkuseg-python/blob/master/tags.txt)

```
import pkuseg

seg = pkuseg.pkuseg(postag=True)  # 开启词性标注功能
text = seg.cut('我爱北京天安门')    # 进行分词和词性标注
print(text)

```

代码示例4：对文件分词
```
import pkuseg

# 对input.txt的文件分词输出到output.txt中
# 开20个进程
pkuseg.test('input.txt', 'output.txt', nthread=20)     
```

代码示例5：额外使用用户自定义词典

```
import pkuseg

seg = pkuseg.pkuseg(user_dict='my_dict.txt')  # 给定用户词典为当前目录下的"my_dict.txt"
text = seg.cut('我爱北京天安门')                # 进行分词
print(text)
```

代码示例6：使用自训练模型分词（以CTB8模型为例）

```
import pkuseg

seg = pkuseg.pkuseg(model_name='./ctb8')  # 假设用户已经下载好了ctb8的模型并放在了'./ctb8'目录下，通过设置model_name加载该模型
text = seg.cut('我爱北京天安门')            # 进行分词
print(text)
```

代码示例7：训练新模型 （模型随机初始化）

```
import pkuseg

# 训练文件为'msr_training.utf8'
# 测试文件为'msr_test_gold.utf8'
# 训练好的模型存到'./models'目录下
# 训练模式下会保存最后一轮模型作为最终模型
# 目前仅支持utf-8编码，训练集和测试集要求所有单词以单个或多个空格分开
pkuseg.train('msr_training.utf8', 'msr_test_gold.utf8', './models')	
```

代码示例8：fine-tune训练（从预加载的模型继续训练）

```
import pkuseg

# 训练文件为'train.txt'
# 测试文件为'test.txt'
# 加载'./pretrained'目录下的模型，训练好的模型保存在'./models'，训练10轮
pkuseg.train('train.txt', 'test.txt', './models', train_iter=10, init_model='./pretrained')
```
## 预训练模型

从pip安装的用户在使用细领域分词功能时，只需要设置model_name字段为对应的领域即可，会自动下载对应的细领域模型。

从github下载的用户则需要自己下载对应的预训练模型，并设置model_name字段为预训练模型路径。预训练模型可以在[release](https://github.com/lancopku/pkuseg-python/releases)部分下载。以下是对预训练模型的说明：

- news: 在MSRA（新闻语料）上训练的模型。

- web: 在微博（网络文本语料）上训练的模型。

- medicine: 在医药领域上训练的模型。

- tourism: 在旅游领域上训练的模型。

- mixed: 混合数据集训练的通用模型。随pip包附带的是此模型。


## 其他相关论文

- Xu Sun, Houfeng Wang, Wenjie Li. Fast Online Training with Frequency-Adaptive Learning Rates for Chinese Word Segmentation and New Word Detection. ACL. 2012.

- Jingjing Xu and Xu Sun. Dependency-based gated recursive neural network for chinese word segmentation. ACL. 2016.

- Jingjing Xu and Xu Sun. Transfer learning for low-resource chinese word segmentation with a novel neural network. NLPCC. 2017.

## 常见问题及解答
[1. 为什么要发布pkuseg？
2. pkuseg使用了哪些技术？
3. 无法使用多进程分词和训练功能，提示RuntimeError和BrokenPipeError。
4. 是如何跟其它工具包在细领域数据上进行比较的？
5. 在黑盒测试集上进行比较的话，效果如何？
6. 如果我不了解待分词语料的所属领域呢？
7. 如何看待在一些特定样例上的分词结果？
8. 关于运行速度问题？
9. 关于多进程速度问题？](https://github.com/lancopku/pkuseg-python/wiki/FAQ)




