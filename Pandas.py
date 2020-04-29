import pandas as pd
import numpy as np
from pandas import Series, DataFrame

'''
认识Series数据结构

0    4
1    7
2   -5
3    3
dtype: int64
[ 4  7 -5  3]
RangeIndex(start=0, stop=4, step=1)
d    4
b    7
a   -5
c    3
dtype: int64
-5
c    3
a   -5
d    6
dtype: int64
d    6
b    7
c    3
dtype: int64
[4 7 3]
d    12
b    14
a   -10
c     6
dtype: int64
d     403.428793
b    1096.633158
a       0.006738
c      20.085537
dtype: float64
True
False
Ohio      35000
Texas     71000
Oregen    16000
Utah       5000
dtype: int64
California        NaN
Ohio          35000.0
Oregen        16000.0
Texas         71000.0
dtype: float64
California     True
Ohio          False
Oregen        False
Texas         False
dtype: bool
California    False
Ohio           True
Oregen         True
Texas          True
dtype: bool
California     True
Ohio          False
Oregen        False
Texas         False
dtype: bool
Ohio      35000
Texas     71000
Oregen    16000
Utah       5000
dtype: int64
California        NaN
Ohio          35000.0
Oregen        16000.0
Texas         71000.0
dtype: float64
California         NaN
Ohio           70000.0
Oregen         32000.0
Texas         142000.0
Utah               NaN
dtype: float64
states
California        NaN
Ohio          35000.0
Oregen        16000.0
Texas         71000.0
Name: population, dtype: float64
Index(['California', 'Ohio', 'Oregen', 'Texas'], dtype='object', name='states')
0    4
1    7
2   -5
3    3
dtype: int64
Bob      4
Steve    7
Jeff    -5
Ryan     3
dtype: int64
'''
def introduce_series():
    obj = pd.Series([4, 7, -5, 3])
    print(obj)
    print(obj.values)
    print(obj.index)

    obj2 = pd.Series([4, 7, -5, 3], index = ['d', 'b', 'a', 'c'])
    print(obj2)
    print(obj2['a'])
    obj2 ['d']=6
    print(obj2[['c','a','d']])

    print(obj2[obj2 > 0])

    arr = np.array([4, 7, -5, 3])
    print(arr[arr > 0])

    print(obj2 * 2)
    print(np.exp(obj2))

    print('b' in obj2)
    print('e' in obj2)
    
    sdata ={'Ohio':35000,'Texas':71000,'Oregen':16000,'Utah':5000}
    obj3 = pd.Series(sdata)
    print(obj3)

    states = ['California', 'Ohio', 'Oregen', 'Texas']
    obj4 = pd.Series(sdata, index=states)
    print(obj4)

    #NaN 是缺失值

    print(pd.isnull(obj4))
    print(pd.notnull(obj4))
    print(obj4.isnull())

    print(obj3)
    print(obj4)
    print(obj3+obj4)

    obj4.name = 'population'
    obj4.index.name = 'states'
    print(obj4)
    print(obj4.index)

    print(obj)
    obj.index=['Bob','Steve','Jeff','Ryan']
    print(obj)

if __name__ == "__main__":
    introduce_series()