本项目是利用AlexNet对ImageNet数据集（1TB）进行分类，因数据集太大，未能进行试验。
理论上迭代450000步之后，训练损失值为0.78，验证损失可达到0.98，验证集准确率73.83%

AlexNet的改进（ZFNet）：
使用较小的卷积核（第一次卷积），使用单片GPU,
conv1: ksize 7*7， stride 2	11*11	4             
pool1：ksize 3*3， stride 2	3*3	2

conv2: ksize 5*5， stride 2	5*5	1
pool2：ksize 3*3， stride 2	3*3	2

conv3: ksize 3*3， stride 1	3*3	1

conv4: ksize 3*3， stride 1	3*3	1

conv5: ksize 3*3， stride 1	3*3	1
pool2：ksize 3*3， stride 2	3*3	2