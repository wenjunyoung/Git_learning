

``` 
https://www.cnblogs.com/q735613050/p/9996706.html
```

OpenCV 中默认的图像的表示确实反过来的，也就是 BGR

H, W, C = img.shape()

## Opencv 可视化

1. 画线： cv.line(image, (Start,Start), (end,end), (R,G,B))
2. 画矩形： cv2.rectangle( imge, (左上角起点坐标), (右下角终点坐标), (颜色RGB) )

3. 画圆圈： cv2.circle( image, (圆心坐标),  R, (颜色RGB)  )

## 图像转换

仿射变换

```javascript
cv2.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]])

其中：

src – 输入图像。
 M – 变换矩阵。
 dsize – 输出图像的大小。
 flags – 插值方法的组合（int 类型！）
 borderMode – 边界像素模式（int 类型！）
 borderValue – （重点！）边界填充值; 默认情况下，它为0。

上述参数中：

M作为仿射变换矩阵，一般反映平移或旋转的关系，为InputArray类型的2×3的变换矩阵。

flages表示插值方式，默认为 flags=cv2.INTER_LINEAR，表示线性插值，

此外还有：cv2.INTER_NEAREST（最近邻插值）

		cv2.INTER_AREA （区域插值）

		cv2.INTER_CUBIC（三次样条插值）

		cv2.INTER_LANCZOS4（Lanczos插值）


```

1. 平移

```
定义平移矩阵，需要是numpy的float32类型
# x轴平移200，y轴平移100, 2*3矩阵
M = np.float32([[1, 0, 200], [0, 1, 100]])
# 用仿射变换实现平移
img_s = cv2.warpAffine(img, M, (cols, rows), borderValue=(155, 150, 200))

```

2. 旋转

   ​	1.rot_mat = cv2.getRotationMatrix2D( center,  angle,  scale)

   参数说明：

   - center：图片的旋转中心
   - angle：旋转角度
   - scale：缩放比例，该例中0.5表示我们缩小一半

   2. cv2.warpAffine(img,  rot_mat,   (img.shape[1], img.shape[0]) )

   参数说明： img表示输入的图片，rot_mat表示仿射变化矩阵，(image.shape[1], image.shape[0])表示变换后的图片大小

   

3. 调整大小

![image-20210707161532763](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210707161532763.png)

![image-20210707163422271](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210707163422271.png)

``` 
# 缩放成200x200的方形图像
img_200x200 = cv2.resize(img, (200, 200))

# 不直接指定缩放后大小，通过fx和fy指定缩放比例，0.5则长宽都为原来一半
# 等效于img_200x300 = cv2.resize(img, (300, 200))，注意指定大小的格式是(宽度,高度)
# 插值方法默认是cv2.INTER_LINEAR，这里指定为最近邻插值
img_200x300 = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, 
                              interpolation=cv2.INTER_NEAREST)
```

2. 翻转

![image-20210708095702707](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708095702707.png)



2. 设置边界框、补边

cv2.copyMakeBorder( )

![img](https://img-blog.csdnimg.cn/20200409200348787.png)

src ： 输入的图片
top, bottom, left, right ：相应方向上的边框宽度
borderType：定义要添加边框的类型，它可以是以下的一种：
		cv2.BORDER_CONSTANT：添加的边界框像素值为常数（需要额外再给定一个参数）
		cv2.BORDER_REFLECT：添加的边框像素将是边界元素的镜面反射，类似       gfedcb|abcdefgh|gfedcba
		cv2.BORDER_REFLECT_101 or cv2.BORDER_DEFAULT：和上面类似，但是有一些细微的不同，类似于gfedcb|abcdefgh|gfedcba
		cv2.BORDER_REPLICATE：使用最边界的像素值代替，类似于aaaaaa|abcdefgh|hhhhhhh
		cv2.BORDER_WRAP：不知道怎么解释，直接看吧，cdefgh|abcdefgh|abcdefg
value：如果borderType为cv2.BORDER_CONSTANT时需要填充的常数值。

## 图像加减法

![image-20210708101316102](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708101316102.png)加法 cv2.add()

​		使用opencv的函数，`cv2.add(img1, img2)`,也可以使用numpy数组的加法操作，`res = img1+img2`，两幅图像大小、类型必须一致，或者第二个图像是一个简单的标量值。

![image-20210708101506358](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708101506358.png)

## 图像叠加or图像混合加权实现

​			不是直接相加而是有一个权重，这样可以塑造混合或者透明的效果。

![image-20210708102512872](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708102512872.png)

cv2.addWeighted(img1, Alpha, img2, 1-Alpha, 0)

![image-20210708103644324](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708103644324.png)



## 位运算  

AND， OR， NOT， XOR

![image-20210708103815048](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708103815048.png)

 cv2.bitwise_or()

cv2.bitwise_xor()

cv2.bitwise_not()

## 掩膜

![image-20210708105246552](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708105246552.png)

![image-20210708105355873](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708105355873.png)

## 通道分离 与合并

分离

cv2.split（）函数的使用

``` 
import numpy as np;
import cv2;             #导入opencv模块
 
image=cv2.imread("/home/zje/Pictures/lena.jpeg");#读取要处理的图片
B,G,R = cv2.split(image);                       #分离出图片的B，R，G颜色通道
cv2.imshow("RED",R);                            #显示三通道的值都为R值时d图片
cv2.imshow("GREEN",G);                          #显示三通道的值都为G值时d图片
cv2.imshow("BLUE",B);                           #显示三通道的值都为B值时d图片
cv2.waitKey(0);                                 #不让程序突然结束
```

![image-20210708110150806](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708110150806.png)

merge（）函数的使用：

   将某一颜色通道（如R）与零矩阵合并，形成（R，0，0）从而显示只有红色通道的图

``` 

import numpy as np;
import cv2;             #导入opencv模块
 
image=cv2.imread("/home/zje/Pictures/lena.jpeg");#读取要处理的图片
B,G,R = cv2.split(image);                       #分离出图片的B，R，G颜色通道
zeros = np.zeros(image.shape[:2],dtype="uint8");#创建与image相同大小的零矩阵
cv2.imshow("BLUE",cv2.merge([B,zeros,zeros]));#显示 （B，0，0）图像
cv2.imshow("GREEN",cv2.merge([zeros,G,zeros]));#显示（0，G，0）图像
cv2.imshow("RED",cv2.merge([zeros,zeros,R]));#显示（0，0，R）图像
cv2.waitKey(0)
```

![img](https://img-blog.csdn.net/20180616212500239?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQ0NTM4OTg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

![img](https://img-blog.csdn.net/20180616212608215?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQ0NTM4OTg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

![img](https://img-blog.csdn.net/20180616212634631?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQ0NTM4OTg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## 颜色空间

· 图像处理中有多种色彩空间，例如 RGB、HLS、HSV、HSB、YCrCb、CIE XYZ、CIE Lab 等，经常要遇到色彩空间的转化，以便生成 mask 图等操作.

· 颜色空间也称彩色模型彩色空间，彩色系统彩色空间，彩色系统，主要是在某些标准下用通常可接受的方式对彩色加以说明.

https://www.aiuai.cn/aifarm365.html

![image-20210708111023030](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708111023030.png)

![image-20210708111034912](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708111034912.png)

## 直方图

``` 
1. 使用灰度直方图进行阈值处理；
2. 使用直方图进行白平衡；
3. 使用颜色直方图来跟踪图像中的对象，例如使用 CamShift 算法；
4. 使用颜色直方图作为特征——包括多维的颜色直方图；
5. 使用图像梯度的直方图来形成 HOG 和 SIFT 描述符；
6. 在图像搜索引擎和机器学习中使用的极受欢迎的视觉词袋表示也是直方图！
```

cv2.calcHist(images, channels, mask, histSize, ranges)

``` images 要计算直方图的原始图像
images 要计算直方图的原始图像
channels 通道，[0]:灰度直方图，[0,1,2]BGR彩色直方图
mask 要为某个蒙版像素计算直方图，没有可设置为None.
histSize x轴要分的bins个数[32,32,32]，每个通道分为32个区间
range 可能的像素范围[0,256] 因为calHist不包含后者
```

![image-20210708112931758](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708112931758.png)

![image-20210708112949771](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708112949771.png)

![image-20210708113000807](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708113000807.png)

直方图均衡化

​		图像的直方图是对图像对比度效果上的一种处理，旨在使得图像整体效果均匀，黑与白之间的各个像素级之间的点更均匀一点。 

​		通过这种方法，亮度可以更好地在直方图上分布。这样就可以用于增强局部的对比度而不影响整体的对比度，直方图均衡化通过有效地扩展常用的亮度来实现这种功能。

​		这种方法对于背景和前景都太亮或者太暗的图像非常有用，这种方法尤其是可以带来X光图像中更好的骨骼结构显示以及曝光过度或者曝光不足照片中更好的细节。

``` 
cv2.equalizeHist(img)，将要均衡化的原图像【要求是灰度图像】作为参数传入，则返回值即为均衡化后的图像。
```

## 图像平滑与模糊

均值模糊

![image-20210708115252862](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708115252862.png)

方框滤波

![image-20210708115846201](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708115846201.png)

高斯滤波

![image-20210708115606871](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708115606871.png)

中值滤波

![image-20210708115648210](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708115648210.png)

双边滤波

![image-20210708115745389](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708115745389.png)

2D卷积

![image-20210708115911560](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708115911560.png)

## 阈值/二值化

简单阈值

```
cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
 - 第一个原图像，第二个进行分类的阈值，第三个是高于（低于）阈值时赋予的新值，第四个是一个方法选择参数
 - 函数有两个返回值，第一个retVal（得到的阈值值（在后面一个方法中会用到）），第二个就是阈值化后的图像
```

![image-20210708142016775](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708142016775.png)

自适应阈值

``` 
cv2.adaptiveThreshold（）
第一个原始图像
第二个像素值上限
第三个自适应方法Adaptive Method:
			— cv2.ADAPTIVE_THRESH_MEAN_C ：领域内均值
			—cv2.ADAPTIVE_THRESH_GAUSSIAN_C ：领域内像素点加权和，权 重为一个高斯窗口
第四个值的赋值方法：只有cv2.THRESH_BINARY 和cv2.THRESH_BINARY_INV
第五个Block size: 规定领域大小（一个正方形的领域）
第六个常数C， 阈值等于均值或者加权值减去这个常数（为0相当于阈值 就是求得领域内均值或者加权值）
			这种方法理论上得到的效果更好，相当于在动态自适应的调整属于自己像素点的阈值，而不是整幅图像都用一个阈值。
```

![image-20210708142628411](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708142628411.png)

Otsu’s二值化

![image-20210708142953831](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708142953831.png)

![image-20210708143002779](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708143002779.png)

![image-20210708143026020](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708143026020.png)

![image-20210708143036882](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708143036882.png)

print(ret2) 得到的结果为144。可以看出似乎两个结果并没有很明显差别，主要是两个阈值（127与144）太相近了，如果这两个隔得很远那么会很明显的



## 梯度 边缘检测

``` 
https://blog.csdn.net/on2way/article/details/46851451
```



![image-20210708145235714](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708145235714.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200228112159196.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTAzMDcwNDg=,size_16,color_FFFFFF,t_70)

山脊边缘 Ridge edge：图像灰度值突然变化，然后在很短的距离内回到开始的值；通常由图像中的线产生

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200228112527966.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTAzMDcwNDg=,size_16,color_FFFFFF,t_70)

![img](https://img-blog.csdnimg.cn/20200228121926124.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTAzMDcwNDg=,size_16,color_FFFFFF,t_70)

屋顶边缘：灰度变化不是瞬间的而是在有限距离内发生的一种山脊边缘；通常在曲面相交处产生

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200228122221511.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTAzMDcwNDg=,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200228122257170.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTAzMDcwNDg=,size_16,color_FFFFFF,t_70)

阶跃/斜坡边缘术语

边缘描述子
——**边缘法向**：最大灰度变化方向的单位矢量。
——**边缘方向**：沿着边缘的单位矢量（垂直于边缘法线）。
——**边缘位置/中心**：边缘所在图像中的位置
——**边缘强度/幅值**：沿着边缘法向的局部图像对比度

![img](https://img-blog.csdnimg.cn/20200228122952453.png)

- 梯度简单来说就是求导，在图像上表现出来的就是提取图像的边缘（不管是横向的、纵向的、斜方向的等等），所需要的无非也是一个核模板，模板的不同结果也不同。所以可以看到，所有的这些个算子函数，归结到底都可以用函数cv2.filter2D()来表示，不同的方法给予不同的核模板，然后演化为不同的算子而已。

![image-20210708152601173](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708152601173.png)

![image-20210708152623783](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708152623783.png)

![image-20210708152810205](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708152810205.png)

![image-20210708152952754](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708152952754.png)

<img src="https://img-blog.csdn.net/20130627154241703?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc3VubnkyMDM4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center" alt="img" style="zoom:33%;" /><img src="https://img-blog.csdn.net/20130627154403390?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc3VubnkyMDM4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center" alt="img" style="zoom: 33%;" />

Laplacian算子

![image-20210708153645625](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708153645625.png)

``` 
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('flower.jpg',0)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)#默认ksize=3
sobely = cv2.Sobel(img,cv2.CV_64F,0,1)
sobelxy = cv2.Sobel(img,cv2.CV_64F,1,1)
laplacian = cv2.Laplacian(img,cv2.CV_64F)#默认ksize=3
#人工生成一个高斯核，去和函数生成的比较
kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]],np.float32)#
img1 = np.float64(img)#转化为浮点型的
img_filter = cv2.filter2D(img1,-1,kernel)
sobelxy1 = cv2.Sobel(img1,-1,1,1)

plt.subplot(221),plt.imshow(sobelx,'gray')
plt.subplot(222),plt.imshow(sobely,'gray')
plt.subplot(223),plt.imshow(sobelxy,'gray')
plt.subplot(224),plt.imshow(laplacian,'gray')

plt.figure()
plt.imshow(img_filter,'gray')
```

![img](https://img-blog.csdn.net/20150712165425613)

![这里写图片描述](https://img-blog.csdn.net/20150712165438843)上述一个很重要的问题需要明白的就是，在滤波函数第二个参数，当我们使用-1表示输出图像与输入图像的数据类型一致时，如果原始图像是uint8型的，那么在经过算子计算以后，得到的图像可能会有负值，如果与原图像数据类型一致，那么负值就会被截断变成0或者255，使得结果错误，那么针对这种问题有两种方式改变（上述程序中都有）：一种就是改变输出图像的数据类型（第二个参数cv2.CV_64F），另一种就是改变原始图像的数据类型（此时第二个参数可以为-1，与原始图像一致)。
上述程序从结果上也说明使用函数cv2.filter2D也能达到相同的效果。

![image-20210708153936713](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708153936713.png)

![image-20210708153950335](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708153950335.png)

``` 

#coding=utf-8
import cv2
import numpy as np  
 
img = cv2.imread("D:/lion.jpg", 0)
 
gray_lap = cv2.Laplacian(img,cv2.CV_16S,ksize = 3)
dst = cv2.convertScaleAbs(gray_lap)
 
cv2.imshow('laplacian',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

![img](https://img-blog.csdn.net/20130628170607187?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc3VubnkyMDM4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)



``` 
https://blog.csdn.net/xiaojiegege123456/article/details/7714897 
```



![image-20210708154141722](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708154141722.png)

![image-20210708154204341](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708154204341.png)

![image-20210708154220520](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708154220520.png)

![image-20210708154248110](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708154248110.png)

![image-20210708154306127](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708154306127.png)

## 边缘检测与梯度算子

### 边缘检测

边缘是指图象中灰度发生急剧变化的区域。图象[灰度](https://baike.baidu.com/item/灰度)的变化情况可以用灰度分布的梯度来反映，给定连续图象f(x，y)，其[方向导数](https://baike.baidu.com/item/方向导数)在边缘法线方向上取得局部最大值。

图象中一点的边缘被定义为一个矢量，模为当前点最人的方向导数，方向为该角度代表的方向。通常我们只考虑其模，而不关心方向。

### 梯度算子

**(一)梯度算子可分为3类：**

1、使用差分近似图像函数导数的算子。有些是具有旋转不变性的(如：[Laplacian算子](https://baike.baidu.com/item/Laplacian算子))，因此只需要一个卷积掩模来计算。其它近似[一阶导数](https://baike.baidu.com/item/一阶导数)的算子使用几个掩模。

2、基于图像函数二阶导数过零点的算子(如：M arr—Hild reth或Canny边缘检测算子。

3、试图将图像函数与边缘的参数模型相匹配的箅子。

 

**(二)第一类梯度算子**

[拉普拉斯](https://baike.baidu.com/item/拉普拉斯)(Laplace)算子通常使用3×3的掩模，有时也使用强调中心象素或其邻接性的[拉普拉斯算子](https://baike.baidu.com/item/拉普拉斯算子)(这种近似不再具有旋转不变性)。

拉普拉斯算子的缺点：它对图像中的某些边缘产生双重响应。

图像锐化(shapeening)

图像锐化的目的是图像的边缘更陡峭、清晰。[锐化](https://baike.baidu.com/item/锐化)的输出图像f是根据下式从输入图像g得到的：f(i，j)=g(i，j)-c s(i，j)，其中c是反映锐化程度的正系数，s(i，j)是图像函数锐化程度的度量，用梯度箅子来计算，Laplacian箅子常被用于这一目的。

Prewitt边缘检测算子

Sobel边缘检测算子

 

**(三)第二类梯度算子--二阶导数过零点算子**

根据图象边缘处的一阶微分(梯度)应该是极值点的事实，图象边缘处的二阶微分应为零，确定过零点的位置要比确定极值点容易得多也比较精确。右侧是Lena的过零点检测结果。

为抑制噪声，可先作平滑滤波然后再作二次微分，通常采用高斯函数作平滑滤波，故有LoG(Laplacian of Gaussian)算子。

高斯-拉普拉斯(LoG，Laplacian of Gaussian)算子。

噪声点对边缘检测有较大的影响，效果更好的边缘检测器是高斯-拉普拉斯(Lo G)算子。它把高斯平滑滤波器和拉普拉斯[锐化](https://baike.baidu.com/item/锐化)滤波器结合起来，先平滑掉噪声，再进行[边缘检测](https://baike.baidu.com/item/边缘检测)，所以效果更好。

 

过零点检测

在实现时一般用两个不同参数的高斯函数的差DoG(Difference ofGaussians)对图象作卷积来近似，这样检测来的边缘点称为f(x，y)的过零点(Zero—crossing)。

与前面的微分算子出仅采用很小的邻域来检测边缘不同，过零点(Zero-crossing)的检测所依赖的范闱与参数。有关，但边缘位置与0的选择无关，若只关心全局性的边缘可以选取比较大的[邻域](https://baike.baidu.com/item/邻域)(如0=4时，邻域接近40个象素宽)来获取明显的边缘。过零点检测更可靠，不易受噪声影响，但．缺点是对形状作了过分的平滑，例如会丢失欠明显的[角点](https://baike.baidu.com/item/角点)；还有产生环行边缘的倾向。

产生环行边缘的原因是：图象的边缘多出现于亮度呈现突起或凹陷的位置上，其附近边缘法向线条上一阶微分会出现两个极值点，也就是会出现两个过零点。其整体结果是边缘呈现环行状态。

 

**(四)Canny边缘提取（或边缘检测Edge Detection)**

在如下的三个标准意义下，Canny边缘检测算子对受闩噪声影响的阶跃型边缘是最优的：

1)检测标准--不丢失重要的边缘，不应有虚假的边缘；

2)定位标准--实际边缘与检测到的边缘位置之间的偏差最小；

3)单响应标准--将多个响应降低为单个边缘响应。

 

Canny边缘检测算子的提出是基于以下概念：

(1)边缘检测算子是针对一维信号和前两个最优标准(即检测标准和定位标准)表达的，用微积分方法可以得到完整的解；

(2)如果考虑第三个标准(多个响应)，需要通过数值优化的办法得到最优解，该最优滤波器可以有效地近似为标准差为(的高斯平滑滤波器的一阶微分，其误差小于20%，这是为了便于实现；这与M ar—Hild reth边缘检测算子很相似；它是基于LoG边缘检测算子的；

(3)将边缘检测箅子推广到两维情况。阶跃边缘由位置、方向和可能的幅度(强度)来确定。可以证明将图象与一对称2 D Gaussian做[卷积](https://baike.baidu.com/item/卷积)后再沿梯度方向微分，就构成了一个简单而有效的方向[算子](https://baike.baidu.com/item/算子)(回想一下，LoG过零点算子并不能提供边缘方向的信息，因为它使用了Laplacian滤波器)。

(4)由于噪声引起的对单个边缘的(多个)虚假响应通常造成所谓的“纹状(streaking)"问题。一般而言，该问题在[边缘检测](https://baike.baidu.com/item/边缘检测)中是非常普遍的。

边缘检测算子的输出通常要做阈值化处理，以确定哪些边缘是突出的。

纹状是指边缘轮廓断开的情形，是由算子输出超出或低于阈值的波动引起的。纹状现象可以通过带滞后的[阈值](https://baike.baidu.com/item/阈值)处理(thresh01ding withhysteresis)来消除；

如果边缘响应超过一给定高阈值时，这些象素点构成了某个尺度下的边缘检测算子的确定的输出。

个别的弱响应通常对应于噪声，但是如果这些点是与某些具有强响应的点连接时，它们很可能是图象中真实的边缘。这些连接的象素点在当其响应超过一给定的低阈值时，就被当作边缘象素。

这里的低阈值和高阈值需要根据对信噪比的估计来确定。

(5)算子的合适尺度取决于图象中所含的物体情况。解决该[未知数](https://baike.baidu.com/item/未知数)的方法是使用多个尺度，将所得信息收集起来。不同尺度的Canny检测[算子](https://baike.baidu.com/item/算子)由高斯的不同的标准差(来表示。有可能存在几个尺度的算子对边缘都给出突出的响应(即信噪比超过[阈值](https://baike.baidu.com/item/阈值))；在这种情况下，选择具有最小尺度的算子，因为它定位最准确。

特征综合方法(Feature synthesis appmach)

首先标记出所有由最小尺度算子得到的突出边缘。具有较大尺度(的算子产生的边缘根据它们(标记出的边缘)合成得到(即，根据从较小的尺度收集到的证据来预测较大尺度(应具有的作用效果)。然后将合成得到的边缘响应与较大尺度(的实际边缘响应作比较。仅当它们比通过合成预测的响应显著地强时，才将其标记为边缘。

这一过程可以对一个尺度序列(从小到大)重复进行，通过不断加入较小的尺度中没有的边缘点的方式累积起来生成边缘图。

 

**Canny边缘检测算法**

一般其工作原理如下：

**1）**使用高斯滤波器对图像进行平滑去噪；

**2）**计算输入图像梯度；

**3）**在边缘上使用非极大值抑制（NMS）进行过滤；

**4）**在检测到的边缘上使用双阈值法去除假阳性；

**5）**分析所有的边缘及其之间的连接，以保留真正的边缘并消除不明显的边缘；

``` 
Canny(image, threshold1, threshold2, edges=None, apertureSize=None, L2gradient=None)
 - image：输入的8bit图像

 - threshold1：阈值1

 - threshold2：阈值2

 - edges：输出的边缘图像，8bit单通道，宽高与输入图像一致

 - apertureSize：Sober算子核大小

 - L2gradient：是否使用更精确的方式计算梯度，True使用，否则不用
```

![image-20210708155242624](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708155242624.png)

## 轮廓检测

![image-20210708160102771](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708160102771.png)

**contour返回值 **

cv2.findContours()函数首先返回一个list，list中每个元素都是图像中的一个轮廓，用numpy中的ndarray表示。

- 输出两个轮廓中存储的点的个数，可以看到，第一个轮廓中只有4个元素，这是因为轮廓中并不是存储轮廓上所有的点，而是只存储可以用直线描述轮廓的点的个数，比如一个“正立”的矩形，只需4个顶点就能描述轮廓了。

**hierarchy返回值 **

- 此外，该函数还可返回一个可选的hiararchy结果，这是一个ndarr-ay，其中的元素个数和轮廓个数相同，每个轮廓contours[i]对应4个hierarchy元素hierarchy[i][0] ~hierarchy[i][3]，分别表示后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号，如果没有对应项，则该值为负数。

- hierarchy本身包含两个ndarray，每个ndarray对应一个轮廓，每个轮廓有四个属性。

![image-20210708161336357](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708161336357.png)



## 色调  明暗  Gamma曲线

除了区域，图像本身的属性操作也非常多，比如可以通过 HSV 空间对色调和明暗进行调节。HSV空间是由美国的图形学专家 A. R. Smith 提出的一种颜色空间，HSV 分别是色调（Hue），饱和度（Saturation）和明度（Value）。在 HSV 空间中进行调节就避免了直接在 RGB 空间中调节是还需要考虑三个通道的相关性。OpenCV中H的取值是 [0,180)[0,180)，其他两个通道的取值都是 [0,256)[0,256)，下面例子接着上面例子代码，通过 HSV 空间对图像进行调整：

``` 
# 通过cv2.cvtColor把图像从BGR转换到HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# H空间中，绿色比黄色的值高一点，所以给每个像素+15，黄色的树叶就会变绿
turn_green_hsv = img_hsv.copy()
turn_green_hsv[:, :, 0] = (turn_green_hsv[:, :, 0]+15) % 180
turn_green_img = cv2.cvtColor(turn_green_hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite('turn_green.jpg', turn_green_img)

# 减小饱和度会让图像损失鲜艳，变得更灰
colorless_hsv = img_hsv.copy()
colorless_hsv[:, :, 1] = 0.5 * colorless_hsv[:, :, 1]
colorless_img = cv2.cvtColor(colorless_hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite('colorless.jpg', colorless_img)

# 减小明度为原来一半
darker_hsv = img_hsv.copy()
darker_hsv[:, :, 2] = 0.5 * darker_hsv[:, :, 2]
darker_img = cv2.cvtColor(darker_hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite('darker.jpg', darker_img)

for i, filename in enumerate(['turn_green.jpg', 'colorless.jpg', 'darker.jpg']):
    plt.subplot(221 + i)
    img1 = plt.imread(filename)
    plt.imshow(img1)
    plt.title(filename)
    plt.axis('off')
plt.show()
```

![image-20210708163827903](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708163827903.png)

无论是 HSV 还是 RGB，我们都较难一眼就对像素中值的分布有细致的了解，这时候就需要直方图。

如果直方图中的成分过于靠近 00 或者 255，可能就出现了暗部细节不足或者亮部细节丢失的情况。一个常用方法是考虑**用 Gamma 变换来提升暗部细节**。**Gamma变换**是矫正相机直接成像和人眼感受图像差别的一种常用手段，简单来说就是通过非线性变换让图像从对曝光强度的线性响应变得更接近人眼感受到的响应。具体的定义和实现，还是接着上面代码中读取的图片，执行计算直方图和 Gamma 变换的代码如下：

``` 
import numpy as np

# 分通道计算每个通道的直方图
hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])

*********************************************************************************
# 定义Gamma矫正的函数
def gamma_trans(img, gamma):
    # 具体做法是先归一化到1，然后gamma作为指数值求出新的像素值再还原
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)

    # 实现这个映射用的是OpenCV的查表函数
    return cv2.LUT(img, gamma_table)
*********************************************************************************

# 执行Gamma矫正，小于1的值让暗部细节大量提升，同时亮部细节少量提升
img_corrected = gamma_trans(img, 0.5)
cv2.imwrite('gamma_corrected.jpg', img_corrected)

# 分通道计算Gamma矫正后的直方图
hist_b_corrected = cv2.calcHist([img_corrected], [0], None, [256], [0, 256])
hist_g_corrected = cv2.calcHist([img_corrected], [1], None, [256], [0, 256])
hist_r_corrected = cv2.calcHist([img_corrected], [2], None, [256], [0, 256])

# 将直方图进行可视化
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

pix_hists = [[hist_b, hist_g, hist_r],
             [hist_b_corrected, hist_g_corrected, hist_r_corrected]]

pix_vals = np.arange(256).reshape((-1, 1))
for sub_plt, pix_hist in zip([121, 122], pix_hists):
    ax = fig.add_subplot(sub_plt, projection='3d')
    for c, z, channel_hist in zip(['b', 'g', 'r'], [20, 10, 0], pix_hist):
        cs = [c] * 256
        ax.bar(
            pix_vals,
            channel_hist,
            zs=z,
            zdir='y',
            color=cs,
            alpha=0.618,
            edgecolor='none',
            lw=0)

    ax.set_xlabel('Pixel Values')
    ax.set_xlim([0, 256])
    ax.set_ylabel('Channels')
    ax.set_zlabel('Counts')
plt.show()
```

![image-20210708164631953](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708164631953.png)

## 仿射变换

图像的仿射变换涉及到图像的形状位置角度的变化，是深度学习预处理中常到的功能。仿射变换具体到图像中的应用，主要是对图像的**缩放**，**旋转**，**剪切**，翻转和平移的组合。在 OpenCV 中，仿射变换的矩阵是一个 2×32×3 的矩阵，其中左边的 2×22×2 子矩阵是线性变换矩阵，右边的 2×12×1 的两项是平移项：

![image-20210708164721686](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708164721686.png)

![image-20210708164734480](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708164734480.png)

需要注意的是，对于图像而言，宽度方向是 xx，高度方向是 yy，坐标的顺序和图像像素对应下标一致。所以原点的位置不是左下角而是右上角，y 的方向也不是向上，而是向下。在 OpenCV 中实现仿射变换是通过仿射变换矩阵和 `cv2.warpAffine()` 这个函数，还是通过代码来理解一下：

```
import cv2
import numpy as np

filename = 'D:/images/cartoon/j.jpeg'
# 读取一张照片
img = cv2.imread(filename)

# 沿着横纵轴放大1.6倍，然后平移(-150,-240)，最后沿原图大小截取，等效于裁剪并放大
M_crop_elephant = np.array([[1.6, 0, -150], [0, 1.6, -240]], dtype=np.float32)

img_elephant = cv2.warpAffine(img, M_crop_elephant, (1400, 2000))
cv2.imwrite('cartoon.jpg', img_elephant)

# x轴的剪切变换，角度15°
theta = 15 * np.pi / 180
M_shear = np.array([[1, np.tan(theta), 0], [0, 1, 0]], dtype=np.float32)

img_sheared = cv2.warpAffine(img, M_shear, (1400, 2000))
cv2.imwrite('cartoon_sheared.jpg', img_sheared)

# 顺时针旋转，角度15°
M_rotate = np.array(
    [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta),
                                          np.cos(theta), 0]],
    dtype=np.float32)

img_rotated = cv2.warpAffine(img, M_rotate, (1400, 2000))
cv2.imwrite('cartooni_rotated.jpg', img_rotated)

# 某种变换，具体旋转+缩放+旋转组合可以通过SVD分解理解
M = np.array([[1, 1.5, -400], [0.5, 2, -100]], dtype=np.float32)

img_transformed = cv2.warpAffine(img, M, (1400, 2000))
cv2.imwrite('cartoon_transformed.jpg', img_transformed)

for i, filename in enumerate([
        'cartoon.jpg', 'cartoon_sheared.jpg', 'cartooni_rotated.jpg',
        'cartoon_transformed.jpg'
]):
    plt.subplot(221 + i)
    img1 = plt.imread(filename)
    plt.imshow(img1)
    plt.title(filename)
    plt.axis('off')
plt.show()
```

![image-20210708164823519](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210708164823519.png)

