#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

import time
#set display defaults
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
#显示的图表大小为 10,图形的插值是以最近为原则,图像颜色是灰色

# Make sure that caffe is on the python path:
caffe_root = 'C:\Users\Administrator\Desktop\Caffe_Using\caffe-master'  
# this file is expected to be in {caffe_root}/examples
#这里注意路径一定要设置正确,记得前后可能都有“/”,路径的使用是
#{caffe_root}/examples,记得 caffe-root 中的 python 文件夹需要包括 caffe 文件夹。

#caffe_root = '/home/bids/caffer-root/' #为何设置为具体路径反而不能运行呢

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe 



#import
import os
if not os.path.isfile(caffe_root + 'myself/my_train_val.caffemodel'):
    print("Downloading pre-trained CaffeNet model...")
    

#设置网络为测试阶段，并加载网络模型prototxt和数据平均值mean_npy
caffe.set_mode_gpu()# 采用GPU运算
model_def = caffe_root +'/myself/my_deploy.prototxt'
model_weights = caffe_root +'/myself/snopshot/_iter_89000.caffemodel'

net = caffe.Net(model_def,model_weights,caffe.TEST)  #用caffe的测试模式，即只是提取特征，不训练

#定义转换-预处理函数
#caffe中用的图像是BGR空间，但是matplotlib用的是RGB空间；
#caffe的数值空间是[0,255]但是matplotlib的空间是[0,1]
#载入imagenet的均值，实际图像要剪掉这个均值，从而减少噪声的影响
mu = np.load(caffe_root + '/myself/classification_test/meannpy.npy')
mu = mu.mean(1).mean(1)
print 'mean-subtracted values:', zip('BGR', mu)     #打印B、G、R的平均像素值

# 定义转换输入的data数值函数
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  #分离图像的RGB三通道
transformer.set_mean('data', mu)            # 减去平均像素值
transformer.set_raw_scale('data', 255)      #将0-1空间变为0-255；mat->caffe
transformer.set_channel_swap('data', (2,1,0)) #交换RGB空间到BGR空间
#transformer.set_mean('data', np.load(caffe_root + '/myself/classification_test/meannpy.npy').mean(1).mean(1)) 
# mean pixel，ImageNet的均值
# the reference model operates on images in [0,255] range instead of [0,1]。参考模型运行在【0,255】的灰度,而不是【0,1】
# the reference model has channels in BGR order instead of RGB，因为参考模型本来频道是 BGR,所以要将RGB转换

# set net to batch size of 50
net.blobs['data'].reshape(50,3,256,256)  #batchsize = 50,三通道，图像大小是256*256


img = caffe.io.load_image(caffe_root +'\\myself\\classification_test\\501.jpg')
print('load_image:',img.shape)
traformed_image = transformer.preprocess('data',img)
#print("preprocess:",traformed_image.shape)
plt.imshow(img)
plt.show()


#net.blobs['data'].data[...] = transformer.preprocess('data', image)

#将图像数据拷贝到内存中并分配给网络net
net.blobs['data'].data[...] = traformed_image

out = net.forward()
output_prob = out['prob'][0]        #这里是输出softmax回归向量
#print("Predicted class is #{}.".format(output_prob.argmax()))
plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))
print "Predicted class is # {}.".format(output_prob.argmax())

# load labels,加载标签，并输出top_k
imagenet_labels_filename = caffe_root + '/data/ilsvrc12/synset_words.txt'

try:
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
except:
    print "OOOOOOOOO"
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
# sort top k predictions from softmax output
top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
#net.blobs['prob'].data[0]输出的就是分类的概率
print net.blobs['prob'].data[0]
print labels[top_k]


# # CPU 与 GPU 比较运算时间
# # CPU mode
# start = time.clock()
# net.forward()  # call once for allocation
# end = time.clock()
# print end-start



# start2 = time.clock()
# # GPU mode
# caffe.set_device(0)
# caffe.set_mode_gpu()
# net.forward()  # call once for allocation
# end2 = time.clock()
# print end2-start2


#****提取特征并可视化****
#net.blobs.items()存储了预测图片的网络中各层的feature map的数据
#显示各个层的参数和输出类型,输出分别是（batchsize,通道数或者feature map数目，输出image高，输出image宽）
for layer_name,blob in net.blobs.iteritems():
    print layer_name +'\t' + str(blob.data.shape)

#net.params.items()存储了训练结束后学习好的网络参数
#查看参数，存放参数的数据结构是输出的feature-map数量，输入的feature-map数量，卷积核大小
#这里conv3和conv4分开了，分别是192，则192*2=384
#后面只有一个参数的表示偏置b数量
for layer_name,param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape),str(param[1].data.shape)

#网络的特征存储在net.blobs，参数和bias存储在net.params，以下代码输出每一层的名称和大小。这里亦可手动把它们存储下来。
#[(k, v.data.shape) for k, v in net.blobs.items()]
#显示出各层的参数和形状,第一个是批次,第二个 feature map 数目,第三和第四是每个神经元中图片的长和宽,可以看出,输入是 227*227 的图片,三个频道,卷积是 32 个卷积核卷三个频道,因此有 96 个 feature map
#print [(k, v[0].data.shape) for k, v in net.params.items()]
#输出:一些网络的参数

#**可视化的辅助函数**
# take an array of shape (n, height, width) or (n, height, width, channels)用一个格式是(数量,高,宽)或(数量,高,宽,频道)的阵列
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)每个可视化的都是在一个由一个个网格组成
def vis_square(data, padsize=1, padval=0):
    #标准化数据normalize
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0]))) #ceil是接近于正向-1.2=-1；0.2=1
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    # add some space between filters
    # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))# pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])


    #plt.imshow(data)
    #plt.show()
    #plt.axis('on')


#根据每一层的名称，选择需要可视化的层，可以可视化filter（参数）和output（特征）
# the parameters are a list of [weights, biases],各层的特征，第一个卷积层，共96个过滤器
#权值参数和偏置项参数分别用params["conv1"][0]和params["conv1"][1]
filters = net.params['conv1'][0].data
#print filters.shape
vis_square(filters.transpose(0, 2, 3, 1))#对filters 4维数组进行位置对换，主要是为了将rgb放在最后一维
#plt.show()



#过滤后的输出,96 张 featuremap
feat = net.blobs['conv1'].data[4, :96]   #表示conv1层学习的feature map，显示第4个crop image 的top 96个feature map
#print feat.shape
vis_square(feat, padval=1)
#plt.show()

feat = net.blobs['conv1'].data[0, :36]
vis_square(feat, padval=1)
#plt.show()

feat = net.blobs['pool1'].data[0, :36]
vis_square(feat, padval=1)
#plt.show()

#第二个卷积层:有 128 个滤波器,每个尺寸为 5X5X48。我们只显示前面 48 个滤波器,每一个滤波器为一行。输入:
filters = net.params['conv2'][0].data
vis_square(filters[:48].reshape(48**2, 5, 5))#对conv2层参数w显示，conv2：256*48*5*5，这里显示头48个filters，reshape是为了显示的时候把48个

#5*5的kernel放在一行显示，共48*48的方格显示
feat = net.blobs['pool2'].data[0, :36]
vis_square(feat, padval=1)
#plt.show()

#第二层输出 256 张 feature,这里显示 36 张。输入:
feat = net.blobs['conv2'].data[4, :36]
vis_square(feat, padval=1)

feat = net.blobs['conv2'].data[0, :36]
vis_square(feat, padval=1)

#第三个卷积层:全部 384 个 feature map,输入:
feat = net.blobs['conv3'].data[4]
vis_square(feat, padval=0.5)

#第四个卷积层:全部 384 个 feature map,输入:
feat = net.blobs['conv4'].data[4]
vis_square(feat, padval=0.5)


#第五个卷积层:全部 256 个 feature map,输入:
feat = net.blobs['conv5'].data[4]
vis_square(feat, padval=0.5)

#第五个 pooling 层:我们也可以观察 pooling 层,输入:
feat = net.blobs['pool5'].data[4]
vis_square(feat, padval=1)
#plt.show()

#用caffe 的python接口提取和保存特征比较方便。
features = net.blobs['conv5'].data  # 提取卷积层 5 的特征
vis_square(feat, padval=1)
#print "hahahhahahaa:",features.shape
#np.savez("001",features) # 将特征存储到本文文件中
#print np.load("001.npz")

#plt.show()


#然后我们看看第六层(第一个全连接层)输出后的直方分布:
feat = net.blobs['fc6'].data[4]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)


#plt.show(_ )
#第七层（第二个全连接层）输出后的直方分布:可以看出值的分布没有这么平均了。
feat = net.blobs['fc7'].data[4]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)

#plt.show(_ )


#The final probability output, prob
feat = net.blobs['prob'].data[0]
a = plt.plot(feat.flat)
#plt.show(a)



