# Convolution

## 相关Pytorch操作

```Python
import torch
from torch import nn
layer = nn.Conv2d(1,3,kernel_size=3,stride=1,padding=0)
# 输入通道为1（灰白图像） kernel个数为3 kernel_size=3,stride=1,padding=0
```

- 第一个参数为通道数量
- 第二个为kernel个数（卷积后有几个通道）

### 测试卷积

```Python
x = torch.rand(1,1,28,28)
# 生成1张1通道的28*28的图片

out.shape
Out[10]: torch.Size([1, 3, 26, 26])
```

#### 测试不同的卷积核

```Python
layer = nn.Conv2d(1,3,kernel_size=3,stride=1,padding=1)
out = layer.forward(x)
out.shape
Out[13]: torch.Size([1, 3, 28, 28])

layer = nn.Conv2d(1,3,kernel_size=3,stride=2,padding=1)
out = layer.forward(x)
out.shape
Out[16]: torch.Size([1, 3, 14, 14])

layer = nn.Conv2d(1,3,kernel_size=3,stride=1,padding=1)
out = layer(x) # __call__魔法函数（建议直接使用这个方法，而不用forward）
out.shape
Out[24]: torch.Size([1, 3, 28, 28])
```

## 查看Inner weight和bias

```Python
layer.weight
Out[25]: 
Parameter containing:
tensor([[[[-0.1604,  0.0262,  0.0646],
          [ 0.0820,  0.2523, -0.3069],
          [-0.0864,  0.3072,  0.0513]]],
        [[[ 0.0036, -0.2282,  0.1167],
          [ 0.0921, -0.2939, -0.3062],
          [ 0.0886, -0.2455, -0.2242]]],
        [[[-0.2676, -0.1547,  0.2412],
          [ 0.0856, -0.3229,  0.2217],
          [ 0.2142,  0.2494, -0.1159]]]], requires_grad=True)

layer.weight.shape
Out[26]: torch.Size([3, 1, 3, 3])
# 3个kernel 1层个输入 大小为3*3
```

- 可以直接使用`F.conv2d()`进行基本操作（略）
