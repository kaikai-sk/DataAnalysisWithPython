import numpy as np
import datetime
import matplotlib.pyplot as plt

'''
运行结果：

<class 'numpy.ndarray'>
<class 'list'>
0
3990
'''
def numpy_time_low():
    # 创建了ndarray
    my_arr = np.arange(10000)
    # 创建了list
    my_list = list(range(10000))

    # 打印出两个变量的类型
    print(type(my_arr))
    print(type(my_list))

    # print(my_arr)
    # print(my_list)
    '''
        numpy的方法比python要快
    '''
    start_time = datetime.datetime.now()
    for _ in range(10):
        my_arr2 = my_arr * 2
    end_time = datetime.datetime.now()
    print((end_time-start_time).microseconds)

    start_time = datetime.datetime.now()
    for _ in range(10):
        ny_list2 = [x * 2 for x in my_list]
    end_time = datetime.datetime.now()
    print((end_time-start_time).microseconds)

'''
    Numpy多维数组对象

    [[-0.70708187 -0.11535279 -0.77706864]
    [ 0.12345839 -0.25059675  0.6478144 ]]
    [[-7.07081866 -1.15352787 -7.77068637]
    [ 1.23458394 -2.50596748  6.47814402]]
    [[-1.41416373 -0.23070557 -1.55413727]
    [ 0.24691679 -0.5011935   1.2956288 ]]
    (2, 3)
    float64
'''
def multi_dimention_arr():
    import numpy as np

    # 采用随机数方法构建数组
    data = np.random.randn(2,3)

    print(data)
    # 每个元素 * 10
    print(data * 10)
    # 数组中对应元素相加
    print(data + data)
    # 打印矩阵的维度信息
    print(data.shape)
    # 打印矩阵的data type
    print(data.dtype)

"""
    如何生成ndarray

    <class 'numpy.ndarray'>
    <class 'numpy.ndarray'>
    [6.  7.5 8.  0.  1. ]
    [[1 2 3 4]
    [5 6 7 8]]
    1
    (5,)
    float64
    2
    (2, 4)
    int32
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [[0. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 0.]]
    [[[0. 0.]
    [0. 0.]
    [0. 0.]]

    [[0. 0.]
    [0. 0.]
    [0. 0.]]]
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]
"""
def product_ndarray():
    data1 = [6, 7.5, 8, 0, 1]
    # list to ndarray
    arr1 = np.array(data1)

    data2 = [[1, 2, 3, 4],
             [5, 6, 7, 8]]
    arr2 = np.array(data2)

    print(type(arr1))
    print(type(arr2))

    print(arr1)
    print(arr2)

    
    print(arr1.ndim)
    print(arr1.shape)
    print(arr1.dtype)

    print(arr2.ndim)
    print(arr2.shape)
    print(arr2.dtype)

    print(np.zeros(10))
    print(np.zeros((3,6)))
    # 没有初始化的。 python3.6 没有初始化就是随机的
    print(np.empty((2,3,2)))

    print(np.arange(15))

'''
    ndarray的数据类型

    float64
    int32
    float64
    [ 1.25 -9.6  42.  ]
'''
def ndarray_data_type():
    # 强制生命ndarray中的元素的类型为float64
    arr1 = np.array([1, 2, 3], dtype=np.float64)
    arr2 = np.array([1, 2, 3], dtype=np.int32)

    print(arr1.dtype)
    print(arr2.dtype)

    # astype函数进行类型转换
    float_arr = arr2.astype(np.float64)
    print(float_arr.dtype)

    numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)
    print(numeric_strings.astype(np.float64))

"""
8位的无符号整形： 
0000 0000   ：  0
1111 1111   ：  255

8位的有符号整形： 
1 000 0000 ：   -0（-128）  
1 111 1111
0 000 0000 ：   +0
0 111 1111：    +127


"""

'''
Numpy的四则运算

运行结果：
[[1. 2. 3.]
 [4. 5. 6.]]
[[ 1.  4.  9.]
 [16. 25. 36.]]
[[0. 0. 0.]
 [0. 0. 0.]]
[[1.         0.5        0.33333333]
 [0.25       0.2        0.16666667]]
[[1.         1.41421356 1.73205081]
 [2.         2.23606798 2.44948974]]
[[ 0.  4.  1.]
 [ 7.  2. 12.]]
[[False  True False]
 [ True False  True]]
'''
def numpy_4_operator():
    # arr = np.array([1., 2., 3.], [4., 5., 6.])
    arr = np.array([[1., 2., 3.], [4., 5., 6.]])
    print(arr)

    print(arr * arr) # 元素相乘
    print(arr - arr)
    print(1/arr)
    print(arr ** 0.5) # ** 幂操作

    arr2 = np.array([[0., 4., 1.], [7., 2., 12.]])
    print(arr2)
    print(arr2 > arr)

'''
Traceback (most recent call last):
  File "NumPy.py", line 203, in <module>
    numpy_4_operator()
  File "NumPy.py", line 182, in numpy_4_operator
    arr = np.array([1., 2., 3.], [4., 5., 6.])
TypeError: data type not understood

182    arr = np.array([1., 2., 3.], [4., 5., 6.])
       arr = np.array([[1., 2., 3.], [4., 5., 6.]])
'''

'''
基本索引和切片

运行结果：

'''
def base_index_slice():
    # arr = np.arange(10)
    # print(arr)

    # print(arr[5])
    # print(arr[5:8])

    # arr[5:8] = 12
    # print(arr)
    '''
        0 1 2 3 4 12 12 12 8 9(内存里面的)
    arr->
        arr_slice->
                12 12 12

    '''
    # # arr_slice是引用，修改arr_slice会影响arr的值
    # arr_slice = arr[5:8]
    # print(arr_slice)

    # arr_slice[1] = 12345
    # print(arr)

    # 二维矩阵
    arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(arr2d)
    # print(arr2d[2])
    # print(arr2d[0][2])
    # print(arr2d[0,2])

    # 三维矩阵  （图像处理）
    # arr3d = np.array([[[1,2,3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    # print(arr3d)
    # print(arr3d[0])

    # old_values = arr3d[0].copy()

    # arr3d[0] = 42
    # print(arr3d)

    # arr3d[0] = old_values
    # print(arr3d)

    # print(arr3d[1,0])

    # print(arr2d[:2])  # [0:2]
    # print(arr2d[:2, 1:])
    # print(arr2d[1, :2])
    # temp = arr2d[:2, 2]
    # print(temp)
    # print(temp.ndim)
    # print(temp.shape)
    # temp = temp.reshape(2,1)
    # print(temp)
    # print(temp.ndim)
    # print(temp.shape)
    # temp = temp.reshape(1,2)
    # print(temp)
    # print(temp.ndim)
    # print(temp.shape)

    '''
    [[1]
    [4]
    [7]]
    [1 4 7]
    '''
    # print(arr2d[:, :1])
    # print(arr2d[:, 0])

    '''
    [[1 0 0]
    [4 0 0]
    [7 8 9]]
    '''
    arr2d[:2, 1:] = 0
    print(arr2d)

'''
布尔索引

结果：
['Bob' 'Joe' 'Will' 'Bob' 'Will' 'Joe' 'Joe']
[[ 0.69243252  1.04347406  1.55725763 -0.48528173]
 [ 0.08793525  0.02784941 -2.05238174  0.18119838]
 [ 0.6287053   0.8985214  -0.41873237  0.40412278]
 [ 0.92062871 -0.11166092  0.25947438  1.93028334]
 [-0.7594316  -0.63923709 -0.78700976  0.85967505]
 [-0.31072595 -0.21578612 -0.71202604  0.69716139]
 [ 1.63542645  1.00164738 -0.26152352 -0.00775471]]
[ True False False  True False False False]
[[ 0.69243252  1.04347406  1.55725763 -0.48528173]
 [ 0.92062871 -0.11166092  0.25947438  1.93028334]]
[[ 1.55725763 -0.48528173]
 [ 0.25947438  1.93028334]]
[-0.48528173  1.93028334]
[False  True  True False  True  True  True]
[[ 0.08793525  0.02784941 -2.05238174  0.18119838]
 [ 0.6287053   0.8985214  -0.41873237  0.40412278]
 [-0.7594316  -0.63923709 -0.78700976  0.85967505]
 [-0.31072595 -0.21578612 -0.71202604  0.69716139]
 [ 1.63542645  1.00164738 -0.26152352 -0.00775471]]
<class 'numpy.ndarray'>
[ True False False  True False False False]
[[ 0.08793525  0.02784941 -2.05238174  0.18119838]
 [ 0.6287053   0.8985214  -0.41873237  0.40412278]
 [-0.7594316  -0.63923709 -0.78700976  0.85967505]
 [-0.31072595 -0.21578612 -0.71202604  0.69716139]
 [ 1.63542645  1.00164738 -0.26152352 -0.00775471]]
[ True False  True  True  True False False]
[[ 0.69243252  1.04347406  1.55725763 -0.48528173]
 [ 0.6287053   0.8985214  -0.41873237  0.40412278]
 [ 0.92062871 -0.11166092  0.25947438  1.93028334]
 [-0.7594316  -0.63923709 -0.78700976  0.85967505]]
[[0.69243252 1.04347406 1.55725763 0.        ]
 [0.08793525 0.02784941 0.         0.18119838]
 [0.6287053  0.8985214  0.         0.40412278]
 [0.92062871 0.         0.25947438 1.93028334]
 [0.         0.         0.         0.85967505]
 [0.         0.         0.         0.69716139]
 [1.63542645 1.00164738 0.         0.        ]]
[[7.         7.         7.         7.        ]
 [0.08793525 0.02784941 0.         0.18119838]
 [7.         7.         7.         7.        ]
 [7.         7.         7.         7.        ]
 [7.         7.         7.         7.        ]
 [0.         0.         0.         0.69716139]
 [1.63542645 1.00164738 0.         0.        ]]
'''
def bool_index():
    names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
    print(names)

    data = np.random.randn(7,4)
    print(data)

    print(names =='Bob')
    print(data[names == 'Bob'])

    print(data[names == 'Bob', 2:])
    print(data[names == 'Bob', 3])

    print(names !='Bob')
    print(data[~(names == 'Bob')])

    cond = names == 'Bob'
    print(type(cond))
    print(cond)

    print(data[~cond])

    mask = (names == 'Bob') | (names == 'Will')
    print(mask)
    print(data[mask])

    data[data<0] = 0
    print(data)

    data[names!='Joe'] =7
    print(data)

'''
神奇索引

结果：
[[0. 0. 0. 0.]
 [1. 1. 1. 1.]
 [2. 2. 2. 2.]
 [3. 3. 3. 3.]
 [4. 4. 4. 4.]
 [5. 5. 5. 5.]
 [6. 6. 6. 6.]
 [7. 7. 7. 7.]]
[[4. 4. 4. 4.]
 [3. 3. 3. 3.]
 [0. 0. 0. 0.]
 [6. 6. 6. 6.]]
[[5. 5. 5. 5.]
 [3. 3. 3. 3.]
 [1. 1. 1. 1.]]
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]
 [16 17 18 19]
 [20 21 22 23]
 [24 25 26 27]
 [28 29 30 31]]
[[ 4  5  6  7]
 [20 21 22 23]
 [28 29 30 31]
 [ 8  9 10 11]]
[ 4 23 29 10]
[[ 4  7  5  6]
 [20 23 21 22]
 [28 31 29 30]
 [ 8 11  9 10]]
[[ 4  5  6  7]
 [ 8  9 10 11]
 [20 21 22 23]
 [28 29 30 31]]
'''
def magical_index():
    arr = np.empty((8,4))

    for i in range(8):
        arr[i] = i

    print(arr)

    print(arr[[4,3,0,6]])
    print(arr[[-3, -5, -7]])

    arr = np.arange(32).reshape((8,4))
    print(arr)
    print(arr[[1,5,7,2]])
    print(arr[ [1,5,7,2], [0,3,1,2] ])
    print(arr[[1,5,7,2]][:,[0,3,1,2]])
    print(arr[[1,5,7,2]][[0,3,1,2]])















'''
数组的切片索引
'''
def arr_slice_index():
    arr = np.arange(10)
    arr_slice = arr[5:8]
    arr_slice[0:3]=64
    print(arr)
    print(arr[1:6])


'''
作业：

练习sql；
安装python；
命令行运行python；
安装visual studio code；

解决MySQL存储中文；
'''

'''
数组的转置和换轴
'''
def arr_reverse_alter_axies():
    # arr = np.arange(15).reshape(3,5)
    # print(arr)

    # print(arr.T) 

    # arr = np.random.randn(6,3)
    # print(arr)

    # print(np.dot(arr.T, arr))

    arr = np.arange(16).reshape(2, 2, 4)
    print(arr)
    # 没看明白
    # print(arr.transpose((1,0,2)))

    print(arr.swapaxes(1,2))

'''
通用函数

结果：
[0 1 2 3 4 5 6 7 8 9]
[0.         1.         1.41421356 1.73205081 2.         2.23606798
 2.44948974 2.64575131 2.82842712 3.        ]
[1.00000000e+00 2.71828183e+00 7.38905610e+00 2.00855369e+01
 5.45981500e+01 1.48413159e+02 4.03428793e+02 1.09663316e+03
 2.98095799e+03 8.10308393e+03]
[-0.38198195 -0.41349334 -0.22681757  1.07143114  0.81742682  0.20248671
 -0.70789151 -1.43965095]
[-0.49113055  0.40064256  1.1547515  -0.92835289  0.61822386 -0.76110323
  0.44788687 -0.00267016]
[-0.38198195  0.40064256  1.1547515   1.07143114  0.81742682  0.20248671
  0.44788687 -0.00267016]
[ 3.93518567  2.70134773  4.04039374 -0.55103264 -1.49523371 -3.24249544
 -3.59989243]
[ 0.93518567  0.70134773  0.04039374 -0.55103264 -0.49523371 -0.24249544
 -0.59989243]
[ 3.  2.  4. -0. -1. -3. -3.]
'''
def common_function():
    arr = np.arange(10)
    print(arr)

    print(np.sqrt(arr))
    print(np.exp(arr))

    x = np.random.randn(8)
    y = np.random.randn(8)
    print(x)
    print(y)

    print(np.maximum(x,y))

    arr = np.random.randn(7) * 5
    print(arr)

    #modf函数返回浮点数的小数部分和证书部分
    remainder, whole_part = np.modf(arr)
    print(remainder)
    print(whole_part)

'''
面向数组编程

结果：
[-2 -1  0  1]
[[-2 -1  0  1]
 [-2 -1  0  1]
 [-2 -1  0  1]
 [-2 -1  0  1]]
[[-2 -2 -2 -2]
 [-1 -1 -1 -1]
 [ 0  0  0  0]
 [ 1  1  1  1]]
[[2.82842712 2.23606798 2.         2.23606798]
 [2.23606798 1.41421356 1.         1.41421356]
 [2.         1.         0.         1.        ]
 [2.23606798 1.41421356 1.         1.41421356]]
'''
def face_to_array():
    points = np.arange(-2, 2 , 1)
    print(points)

    xs,ys = np.meshgrid(points,points)
    print(xs)
    print(ys)
    
    z = np.sqrt(xs ** 2 + ys ** 2)
    print(z)

    plt.imshow(z, cmap = plt.cm.gray)
    plt.colorbar()
    plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
    plt.show()

'''
将条件逻辑作为数组操作

运行结果：
[1.1, 2.2, 1.3, 1.4, 2.5]
[2.1 2.2 2.3 2.4 2.5]
[[ 0.63752465  0.51828278 -0.47482236  0.94612604]
 [ 1.09255775  1.19802064 -0.08868469 -0.56213092]
 [ 0.13160435  0.94530834 -0.0552738   0.5749255 ]
 [ 2.10613667 -0.55368743 -1.1441873   0.41439221]]
[[ True  True False  True]
 [ True  True False False]
 [ True  True False  True]
 [ True False False  True]]
[[ 2  2 -2  2]
 [ 2  2 -2 -2]
 [ 2  2 -2  2]
 [ 2 -2 -2  2]]
[[ 2.          2.         -0.47482236  2.        ]
 [ 2.          2.         -0.08868469 -0.56213092]
 [ 2.          2.         -0.0552738   2.        ]
 [ 2.         -0.55368743 -1.1441873   2.        ]]
'''
def condition_array_op():
    xarray = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
    yarray = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
    cond = np.array([True, False, True, True, False])

    result = [(x if c else y) for x, y, c in zip(xarray, yarray, cond)]
    print(result)

    result = np.where(xarray, yarray, cond)
    print(result)

    arr = np.random.randn(4,4)
    print(arr)

    print(arr > 0)

    print(np.where(arr > 0, 2, -2))
    print(np.where(arr > 0, 2, arr))

'''
数学和统计方法

[[0 1 2]
 [3 4 5]
 [6 7 8]]
[[ 0  1  2]
 [ 3  5  7]
 [ 9 12 15]]
[[ 0  1  2]
 [ 0  4 10]
 [ 0 28 80]]
'''
def math_statistic_method():
    arr = np.random.randn(5,4)
    # print(arr)

    # print(arr.mean())
    # print(np.mean(arr))
    # print(arr.sum())

    # print(arr.mean(axis = 1))
    # print(arr.sum(axis = 0))

    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    # print(arr)
    # print(arr.cumsum())

    arr = np.array([[0,1,2], [3, 4, 5], [6, 7, 8]])
    # print(arr)
    # print(arr.cumsum(axis = 0))
    # print(arr.cumprod(axis = 0))


    ### bool值数组的方法
    arr = np.random.randn(10)
    print(arr)
    # 统计正数的个数
    print((arr > 0).sum())

    bools = np.array([False, False, True, False])
    # 至少有一个为TRUE
    print(bools.any())
    # 所有都为true
    print(bools.all())

'''
排序
'''
def sort():
    arr = np.random.randn(5)
    print(arr)
    arr.sort()
    print(arr)

    # 20% percentile 
    print(arr[int(0.2 * len(arr))])

    arr = np.random.randn(5,3)
    print(arr)
    arr.sort(axis = 1)
    print(arr)

'''
集合操作

['Bob' 'Joe' 'Will']
['Bob', 'Joe', 'Will']
[ True False False  True  True False  True]
'''
def set_operate():
    names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
    # distinct 
    print(np.unique(names))

    print(sorted(set(names)))

    values = np.array([6, 0, 0, 3, 2, 5, 6])
    print(np.in1d(values, [2, 3, 6]))

'''
文件操作
'''
def file_operate():
    arr = np.arange(10)
    np.save('some_arr', arr)

    print(np.load('some_arr.npy'))

    # k - v 结构
    np.savez('arr_archive.npz', a = arr, b = arr)
    arch = np.load('arr_archive.npz')
    print(arch)
    # k - v 结构
    # {'a':[0 1 2 3 4 5 6 7 8 9],
    #  'b':[0 1 2 3 4 5 6 7 8 9] }
    print(arch['b'])
    print(arch['a'])
    print(arch['A'])

'''
线性代数

[[ 28.  64.]
 [ 67. 181.]]
[[ 28.  64.]
 [ 67. 181.]]
[[ 1.16999411 -0.10067219 -1.33726426 -1.20643037 -1.32953707]
 [-0.10067219  0.50586977  0.42394564  0.27183461 -0.10028987]
 [-1.33726426  0.42394564  2.92391913  2.01479364  1.78308665]
 [-1.20643037  0.27183461  2.01479364  1.95243865  1.81276902]
 [-1.32953707 -0.10028987  1.78308665  1.81276902  2.71003925]]

[[ 1.00000000e+00  1.00804182e-17  3.63016207e-16  5.32506845e-16
  -3.21293394e-16]
 [-9.83318096e-17  1.00000000e+00  2.08552662e-18  1.73960705e-16
   3.96910012e-17]
 [-2.59141111e-16 -1.65073195e-17  1.00000000e+00 -2.85411106e-16
   3.59974278e-16]
 [ 1.62130244e-16 -1.61380005e-16 -3.56803605e-16  1.00000000e+00
  -2.30618373e-17]
 [ 2.55489996e-16  1.93773878e-16 -8.54602643e-17  1.86392195e-16
   1.00000000e+00]]
[[-0.9338917  -0.05630678 -0.09346094  0.05038271 -0.33675283]
 [ 0.02899543 -0.90409277  0.12760495 -0.40601384 -0.025402  ]
 [-0.12679454  0.11498625 -0.70762233 -0.51575367  0.45163049]
 [-0.27264825  0.26225619  0.68774028 -0.41602943  0.45914861]
 [-0.19128938 -0.31216287 -0.03558053  0.6273213   0.6864144 ]]
[[-2.91756504  0.29133256 -0.2324026  -1.32121277 -0.45447127]
 [ 0.         -3.32956346  0.20826961  1.93050414 -1.66676745]
 [ 0.          0.         -1.7613716   2.91880761 -0.80664011]
 [ 0.          0.          0.         -1.12946828  0.98699176]
 [ 0.          0.          0.          0.          0.25328578]]
'''
def liner_daishu():
    x = np.array([[1.,2.,3.], [4., 5., 6.]])
    y = np.array([[6., 23.], [-1, 7], [8, 9]])

    # 矩阵乘法
    # * 
    print(x.dot(y))
    print(np.dot(x, y))

    from numpy.linalg import inv,qr

    X = np.random.randn(5, 5)
    mat = X.T.dot(X)

    # inv 求逆矩阵
    print(inv(mat))
    print()
    # E矩阵
    # 数值解：3.33333333333333333333333333333333
    # 分析解：10/3
    print(mat.dot(inv(mat)))

    q, r = qr(mat)
    print(q)
    print(r)

'''
伪随机数

[[-0.6397935   0.47535849  0.43555024  1.18810862]
 [ 0.17208923 -1.31364949 -0.1641457   0.40799942]
 [-0.37597055 -0.92800192  1.1709309   1.85358984]
 [ 1.08287133  1.18566971 -0.69085958  1.70358103]]
-2.198533078572999
'''
def random_num():
    # 正态分布
    samples = np.random.normal(size = (4,4))
    print(samples)

    from random import normalvariate
    print(normalvariate(0,1))

'''
随机漫步
'''
def random_walk():
    import random

    # 
    position = 0
    walk = [position] # walk = [0]
    steps = 1000
    for i in range(steps):
        step = 1 if random.randint(0, 1) else -1
        
        # step = 0
        # if random.randint(0, 1) == 1:
        #     step = 1
        # else:
        #     step = -1

        position += step
        walk.append(position)

    print(len(walk))
    plt.plot(walk[:100])
    plt.show()

    '''
    效果同上
    '''
    nsteps = 1000
    draws = np.random.randint(0, 2, size = nsteps)
    steps = np.where(draws >0 , 1, -1)
    walk = steps.cumsum()
    print(len(walk))

    print(walk.min())
    print(walk.max())

    #朝同一方向，连续走了10步的第一次出现的位置 
    # +8 -2 +4
    print((np.abs(walk) >= 10).argmax())

    #一次性模拟多个随机漫步 
    nwalks = 5000
    nsteps = 1000
    draws = np.random.randint(0,2, size=(nwalks, nsteps))
    steps = np.where(draws >0 , 1, -1)
    walks = np.cumsum(1)



# 声明主函数入口
if __name__ == "__main__":
    # multi_dimention_arr()
    # product_ndarray()
    # ndarray_data_type()
    # numpy_4_operator()
    # base_index_slice()
    # arr_slice_index()
    # bool_index()
    # magical_index()
    # arr_reverse_alter_axies()
    # common_function()
    # face_to_array()
    # condition_array_op()
    # math_statistic_method()
    # sort()
    # set_operate()
    # file_operate()
    # liner_daishu()
    # random_num()
    random_walk()
