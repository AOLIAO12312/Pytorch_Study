# Advanced OP高阶OP

## CATALOG

1. [Advanced OP高阶OP](#advanced-op高阶op)
   1. [CATALOG](#catalog)
   2. [Where](#where)
   3. [Gather](#gather)

## Where

- `where()`操作可以处理复杂数据来源的问题

数据：

```Python
a = torch.zeros([2,2])
a
Out[12]: 
tensor([[0., 0.],
        [0., 0.]])
b = torch.ones([2,2])
b
Out[14]: 
tensor([[1., 1.],
        [1., 1.]])

# condition函数，决定数据来源
cond = torch.rand(2,2)
Out[5]: 
tensor([[0.0860, 0.0513],
        [0.4251, 0.6114]])
```

- `where()`的使用
- `condition`必须与a和b的tensor大小一样

```Python
torch.where(cond == False,b,a)
Out[15]: 
tensor([[1., 1.],
        [1., 0.]])
# cond对应值为False时取b对应的数据
# cond对应值为True时取a对应的数据
```

## Gather

PyTorch的`gather`函数是一个非常有用的操作，它允许你根据索引从张量（tensor）中收集数据。这个函数在处理需要基于某些索引来提取数据的场景时特别有用，比如在进行批处理操作或者构建复杂的神经网络架构时。
函数的基本形式如下：

```Python
torch.gather(input, dim, index, *, out=None) -> Tensor
```

- `input`（Tensor）：源张量，从中收集数据。
- `dim`（int）：指定在哪个维度上进行收集操作。
- `index`（Tensor）：索引张量，其形状应与`input`张量在除了`dim`维度之外的所有维度上相匹配。`index`中的每个元素都指定了从`input`的`dim`维度上收集哪个元素。
- `out`（Tensor, 可选）：输出张量。

返回值是一个与`index`具有相同形状的新张量，其中包含了从`input`中根据`index`指定的索引收集的元素。

- 数据

```Python
prob = torch.randn(4,10)

idx = prob.topk(dim=1,k=3)

label = torch.arange(10) + 100

idx
Out[21]: 
torch.return_types.topk(
values=tensor([[1.2684, 1.0465, 0.7268],
        [1.8275, 1.7219, 1.1927],
        [1.2883, 1.2874, 1.2153],
        [2.2869, 2.0007, 1.4192]]),
indices=tensor([[6, 1, 4],
        [8, 7, 9],
        [4, 3, 8],
        [5, 1, 4]]))
```

- `gather()`的使用

```Python
torch.gather(label.expand(4,10),dim = 1,index = idx.indices)
Out[25]: 
tensor([[106, 101, 104],
        [108, 107, 109],
        [104, 103, 108],
        [105, 101, 104]])
# 将索引值映射到label数组上
```
