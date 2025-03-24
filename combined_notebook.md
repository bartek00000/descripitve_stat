# Exercise 1. - Getting and Knowing your Data

This time we are going to pull data directly from the internet.
Special thanks to: https://github.com/justmarkham for sharing the dataset and materials.

Check out [Occupation Exercises Video Tutorial](https://www.youtube.com/watch?v=W8AB5s-L3Rw&list=PLgJhDSE2ZLxaY_DigHeiIDC1cD09rXgJv&index=4) to watch a data scientist go through the exercises

### Step 1. Import the necessary libraries!


```python
import pandas as pd
import seaborn as sns
from dfply import *
```

### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user). 

### Step 3. Assign it to a variable called users and use the 'user_id' as index


```python
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user'  
users= pd.read_csv(url, sep="|")
users.set_index('user_id', inplace=True)
```

### Step 4. See the first 25 entries


```python
users >> head(25)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>gender</th>
      <th>occupation</th>
      <th>zip_code</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>24</td>
      <td>M</td>
      <td>technician</td>
      <td>85711</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53</td>
      <td>F</td>
      <td>other</td>
      <td>94043</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23</td>
      <td>M</td>
      <td>writer</td>
      <td>32067</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24</td>
      <td>M</td>
      <td>technician</td>
      <td>43537</td>
    </tr>
    <tr>
      <th>5</th>
      <td>33</td>
      <td>F</td>
      <td>other</td>
      <td>15213</td>
    </tr>
    <tr>
      <th>6</th>
      <td>42</td>
      <td>M</td>
      <td>executive</td>
      <td>98101</td>
    </tr>
    <tr>
      <th>7</th>
      <td>57</td>
      <td>M</td>
      <td>administrator</td>
      <td>91344</td>
    </tr>
    <tr>
      <th>8</th>
      <td>36</td>
      <td>M</td>
      <td>administrator</td>
      <td>05201</td>
    </tr>
    <tr>
      <th>9</th>
      <td>29</td>
      <td>M</td>
      <td>student</td>
      <td>01002</td>
    </tr>
    <tr>
      <th>10</th>
      <td>53</td>
      <td>M</td>
      <td>lawyer</td>
      <td>90703</td>
    </tr>
    <tr>
      <th>11</th>
      <td>39</td>
      <td>F</td>
      <td>other</td>
      <td>30329</td>
    </tr>
    <tr>
      <th>12</th>
      <td>28</td>
      <td>F</td>
      <td>other</td>
      <td>06405</td>
    </tr>
    <tr>
      <th>13</th>
      <td>47</td>
      <td>M</td>
      <td>educator</td>
      <td>29206</td>
    </tr>
    <tr>
      <th>14</th>
      <td>45</td>
      <td>M</td>
      <td>scientist</td>
      <td>55106</td>
    </tr>
    <tr>
      <th>15</th>
      <td>49</td>
      <td>F</td>
      <td>educator</td>
      <td>97301</td>
    </tr>
    <tr>
      <th>16</th>
      <td>21</td>
      <td>M</td>
      <td>entertainment</td>
      <td>10309</td>
    </tr>
    <tr>
      <th>17</th>
      <td>30</td>
      <td>M</td>
      <td>programmer</td>
      <td>06355</td>
    </tr>
    <tr>
      <th>18</th>
      <td>35</td>
      <td>F</td>
      <td>other</td>
      <td>37212</td>
    </tr>
    <tr>
      <th>19</th>
      <td>40</td>
      <td>M</td>
      <td>librarian</td>
      <td>02138</td>
    </tr>
    <tr>
      <th>20</th>
      <td>42</td>
      <td>F</td>
      <td>homemaker</td>
      <td>95660</td>
    </tr>
    <tr>
      <th>21</th>
      <td>26</td>
      <td>M</td>
      <td>writer</td>
      <td>30068</td>
    </tr>
    <tr>
      <th>22</th>
      <td>25</td>
      <td>M</td>
      <td>writer</td>
      <td>40206</td>
    </tr>
    <tr>
      <th>23</th>
      <td>30</td>
      <td>F</td>
      <td>artist</td>
      <td>48197</td>
    </tr>
    <tr>
      <th>24</th>
      <td>21</td>
      <td>F</td>
      <td>artist</td>
      <td>94533</td>
    </tr>
    <tr>
      <th>25</th>
      <td>39</td>
      <td>M</td>
      <td>engineer</td>
      <td>55107</td>
    </tr>
  </tbody>
</table>
</div>



### Step 5. See the last 10 entries


```python
users >> tail(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>gender</th>
      <th>occupation</th>
      <th>zip_code</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>934</th>
      <td>61</td>
      <td>M</td>
      <td>engineer</td>
      <td>22902</td>
    </tr>
    <tr>
      <th>935</th>
      <td>42</td>
      <td>M</td>
      <td>doctor</td>
      <td>66221</td>
    </tr>
    <tr>
      <th>936</th>
      <td>24</td>
      <td>M</td>
      <td>other</td>
      <td>32789</td>
    </tr>
    <tr>
      <th>937</th>
      <td>48</td>
      <td>M</td>
      <td>educator</td>
      <td>98072</td>
    </tr>
    <tr>
      <th>938</th>
      <td>38</td>
      <td>F</td>
      <td>technician</td>
      <td>55038</td>
    </tr>
    <tr>
      <th>939</th>
      <td>26</td>
      <td>F</td>
      <td>student</td>
      <td>33319</td>
    </tr>
    <tr>
      <th>940</th>
      <td>32</td>
      <td>M</td>
      <td>administrator</td>
      <td>02215</td>
    </tr>
    <tr>
      <th>941</th>
      <td>20</td>
      <td>M</td>
      <td>student</td>
      <td>97229</td>
    </tr>
    <tr>
      <th>942</th>
      <td>48</td>
      <td>F</td>
      <td>librarian</td>
      <td>78209</td>
    </tr>
    <tr>
      <th>943</th>
      <td>22</td>
      <td>M</td>
      <td>student</td>
      <td>77841</td>
    </tr>
  </tbody>
</table>
</div>



### Step 6. What is the number of observations in the dataset?


```python
users.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 943 entries, 1 to 943
    Data columns (total 4 columns):
     #   Column      Non-Null Count  Dtype 
    ---  ------      --------------  ----- 
     0   age         943 non-null    int64 
     1   gender      943 non-null    object
     2   occupation  943 non-null    object
     3   zip_code    943 non-null    object
    dtypes: int64(1), object(3)
    memory usage: 36.8+ KB
    

### Step 7. What is the number of columns in the dataset?


```python
print(users.shape[1])
```

    4
    

### Step 8. Print the name of all the columns.


```python
users.columns
```




    Index(['age', 'gender', 'occupation', 'zip_code'], dtype='object')



### Step 9. How is the dataset indexed?


```python
users.index
```




    Index([  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,
           ...
           934, 935, 936, 937, 938, 939, 940, 941, 942, 943],
          dtype='int64', name='user_id', length=943)



### Step 10. What is the data type of each column?


```python
users.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 943 entries, 1 to 943
    Data columns (total 4 columns):
     #   Column      Non-Null Count  Dtype 
    ---  ------      --------------  ----- 
     0   age         943 non-null    int64 
     1   gender      943 non-null    object
     2   occupation  943 non-null    object
     3   zip_code    943 non-null    object
    dtypes: int64(1), object(3)
    memory usage: 36.8+ KB
    

### Step 11. Print only the occupation column


```python
 users >> select("occupation") >> head(10)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>occupation</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>technician</td>
    </tr>
    <tr>
      <th>2</th>
      <td>other</td>
    </tr>
    <tr>
      <th>3</th>
      <td>writer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>technician</td>
    </tr>
    <tr>
      <th>5</th>
      <td>other</td>
    </tr>
    <tr>
      <th>6</th>
      <td>executive</td>
    </tr>
    <tr>
      <th>7</th>
      <td>administrator</td>
    </tr>
    <tr>
      <th>8</th>
      <td>administrator</td>
    </tr>
    <tr>
      <th>9</th>
      <td>student</td>
    </tr>
    <tr>
      <th>10</th>
      <td>lawyer</td>
    </tr>
  </tbody>
</table>
</div>




```python
users.occupation.head(10)
```




    user_id
    1        technician
    2             other
    3            writer
    4        technician
    5             other
    6         executive
    7     administrator
    8     administrator
    9           student
    10           lawyer
    Name: occupation, dtype: object



### Step 12. How many different occupations are in this dataset?


```python

x= users.occupation.drop_duplicates()
print(x.count())
```

    21
    


```python
users.occupation.nunique()
```




    21



### Step 13. What is the most frequent occupation?


```python
users.occupation.value_counts()
```




    occupation
    student          196
    other            105
    educator          95
    administrator     79
    engineer          67
    programmer        66
    librarian         51
    writer            45
    executive         32
    scientist         31
    artist            28
    technician        27
    marketing         26
    entertainment     18
    healthcare        16
    retired           14
    lawyer            12
    salesman          12
    none               9
    homemaker          7
    doctor             7
    Name: count, dtype: int64



### Step 14. Summarize the DataFrame.


```python
users.describe(include='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>gender</th>
      <th>occupation</th>
      <th>zip_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>943.000000</td>
      <td>943</td>
      <td>943</td>
      <td>943</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>2</td>
      <td>21</td>
      <td>795</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>M</td>
      <td>student</td>
      <td>55414</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>670</td>
      <td>196</td>
      <td>9</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>34.051962</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12.192740</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>7.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>25.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>31.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>43.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>73.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Step 15. Summarize all the columns


```python
users.describe(include='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>gender</th>
      <th>occupation</th>
      <th>zip_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>943.000000</td>
      <td>943</td>
      <td>943</td>
      <td>943</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>2</td>
      <td>21</td>
      <td>795</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>M</td>
      <td>student</td>
      <td>55414</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>670</td>
      <td>196</td>
      <td>9</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>34.051962</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12.192740</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>7.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>25.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>31.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>43.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>73.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Step 16. Summarize only the occupation column


```python
users.occupation.describe()
```




    count         943
    unique         21
    top       student
    freq          196
    Name: occupation, dtype: object



### Step 17. What is the mean age of users?


```python
users.age.mean()
```




    np.float64(34.05196182396607)



### Step 18. What is the age with least occurrence?


```python
users.age.value_counts().tail()
```




    age
    7     1
    11    1
    66    1
    10    1
    73    1
    Name: count, dtype: int64



---
End of Exercise1.ipynb
---

# Exercise 2. - Filtering and Sorting Data

Check out [Euro 12 Exercises Video Tutorial](https://youtu.be/iqk5d48Qisg) to watch a data scientist go through the exercises

This time we are going to pull data directly from the internet.

### Step 1. Import the necessary libraries


```python
import pandas as pd
import seaborn as sns
from dfply import *

```

### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/kflisikowsky/pandas_exercises/refs/heads/main/Euro_2012_stats_TEAM.csv). 

### Step 3. Assign it to a variable called euro12.


```python
url ='https://raw.githubusercontent.com/kflisikowsky/pandas_exercises/refs/heads/main/Euro_2012_stats_TEAM.csv'
df=pd.read_csv(url)
euro12 = df
print(euro12.head())
```

                 Team  Goals  Shots on target  Shots off target Shooting Accuracy  \
    0         Croatia      4               13                12             51.9%   
    1  Czech Republic      4               13                18             41.9%   
    2         Denmark      4               10                10             50.0%   
    3         England      5               11                18             50.0%   
    4          France      3               22                24             37.9%   
    
      % Goals-to-shots  Total shots (inc. Blocked)  Hit Woodwork  Penalty goals  \
    0            16.0%                          32             0              0   
    1            12.9%                          39             0              0   
    2            20.0%                          27             1              0   
    3            17.2%                          40             0              0   
    4             6.5%                          65             1              0   
    
       Penalties not scored  ...  Saves made  Saves-to-shots ratio  Fouls Won  \
    0                     0  ...          13                 81.3%         41   
    1                     0  ...           9                 60.1%         53   
    2                     0  ...          10                 66.7%         25   
    3                     0  ...          22                 88.1%         43   
    4                     0  ...           6                 54.6%         36   
    
      Fouls Conceded  Offsides  Yellow Cards  Red Cards  Subs on  Subs off  \
    0             62         2             9          0        9         9   
    1             73         8             7          0       11        11   
    2             38         8             4          0        7         7   
    3             45         6             5          0       11        11   
    4             51         5             6          0       11        11   
    
       Players Used  
    0            16  
    1            19  
    2            15  
    3            16  
    4            19  
    
    [5 rows x 35 columns]
    

### Step 4. Select only the Goal column.


```python
euro12 >> select(X.Goals)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Goals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>6</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>5</td>
    </tr>
    <tr>
      <th>13</th>
      <td>12</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



### Step 5. How many team participated in the Euro2012?


```python
print(euro12.Team.count())
```

    16
    

### Step 6. What is the number of columns in the dataset?


```python
euro12.shape[1]
```




    35



### Step 7. View only the columns Team, Yellow Cards and Red Cards and assign them to a dataframe called discipline


```python
discipline = euro12[['Team', 'Yellow Cards', 'Red Cards']]
discipline
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team</th>
      <th>Yellow Cards</th>
      <th>Red Cards</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Croatia</td>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Czech Republic</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Denmark</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>England</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>France</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Germany</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Greece</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Italy</td>
      <td>16</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Netherlands</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Poland</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Portugal</td>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Republic of Ireland</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Russia</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Spain</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Sweden</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Ukraine</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Step 8. Sort the teams by Red Cards, then to Yellow Cards


```python
discipline.sort_values(['Red Cards', 'Yellow Cards'] ,ascending = False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team</th>
      <th>Yellow Cards</th>
      <th>Red Cards</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>Greece</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Poland</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Republic of Ireland</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Italy</td>
      <td>16</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Portugal</td>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Spain</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Croatia</td>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Czech Republic</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Sweden</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>France</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Russia</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>England</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Netherlands</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Ukraine</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Denmark</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Germany</td>
      <td>4</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Step 9. Calculate the mean Yellow Cards given per Team


```python
mean = euro12['Yellow Cards'].mean()
print(mean)
```

    7.4375
    

### Step 10. Filter teams that scored more than 6 goals


```python
more_than_6= euro12.query('Goals > 6')
more_than_6
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team</th>
      <th>Goals</th>
      <th>Shots on target</th>
      <th>Shots off target</th>
      <th>Shooting Accuracy</th>
      <th>% Goals-to-shots</th>
      <th>Total shots (inc. Blocked)</th>
      <th>Hit Woodwork</th>
      <th>Penalty goals</th>
      <th>Penalties not scored</th>
      <th>...</th>
      <th>Saves made</th>
      <th>Saves-to-shots ratio</th>
      <th>Fouls Won</th>
      <th>Fouls Conceded</th>
      <th>Offsides</th>
      <th>Yellow Cards</th>
      <th>Red Cards</th>
      <th>Subs on</th>
      <th>Subs off</th>
      <th>Players Used</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>Germany</td>
      <td>10</td>
      <td>32</td>
      <td>32</td>
      <td>47.8%</td>
      <td>15.6%</td>
      <td>80</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>62.6%</td>
      <td>63</td>
      <td>49</td>
      <td>12</td>
      <td>4</td>
      <td>0</td>
      <td>15</td>
      <td>15</td>
      <td>17</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Spain</td>
      <td>12</td>
      <td>42</td>
      <td>33</td>
      <td>55.9%</td>
      <td>16.0%</td>
      <td>100</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>15</td>
      <td>93.8%</td>
      <td>102</td>
      <td>83</td>
      <td>19</td>
      <td>11</td>
      <td>0</td>
      <td>17</td>
      <td>17</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 35 columns</p>
</div>



### Step 11. Select the teams that start with G


```python
starts_with_G= euro12.query('Team.str.startswith("G")')
starts_with_G
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team</th>
      <th>Goals</th>
      <th>Shots on target</th>
      <th>Shots off target</th>
      <th>Shooting Accuracy</th>
      <th>% Goals-to-shots</th>
      <th>Total shots (inc. Blocked)</th>
      <th>Hit Woodwork</th>
      <th>Penalty goals</th>
      <th>Penalties not scored</th>
      <th>...</th>
      <th>Saves made</th>
      <th>Saves-to-shots ratio</th>
      <th>Fouls Won</th>
      <th>Fouls Conceded</th>
      <th>Offsides</th>
      <th>Yellow Cards</th>
      <th>Red Cards</th>
      <th>Subs on</th>
      <th>Subs off</th>
      <th>Players Used</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>Germany</td>
      <td>10</td>
      <td>32</td>
      <td>32</td>
      <td>47.8%</td>
      <td>15.6%</td>
      <td>80</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>62.6%</td>
      <td>63</td>
      <td>49</td>
      <td>12</td>
      <td>4</td>
      <td>0</td>
      <td>15</td>
      <td>15</td>
      <td>17</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Greece</td>
      <td>5</td>
      <td>8</td>
      <td>18</td>
      <td>30.7%</td>
      <td>19.2%</td>
      <td>32</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>13</td>
      <td>65.1%</td>
      <td>67</td>
      <td>48</td>
      <td>12</td>
      <td>9</td>
      <td>1</td>
      <td>12</td>
      <td>12</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 35 columns</p>
</div>



### Step 12. Select the first 7 columns


```python
euro12.iloc[0:7]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team</th>
      <th>Goals</th>
      <th>Shots on target</th>
      <th>Shots off target</th>
      <th>Shooting Accuracy</th>
      <th>% Goals-to-shots</th>
      <th>Total shots (inc. Blocked)</th>
      <th>Hit Woodwork</th>
      <th>Penalty goals</th>
      <th>Penalties not scored</th>
      <th>...</th>
      <th>Saves made</th>
      <th>Saves-to-shots ratio</th>
      <th>Fouls Won</th>
      <th>Fouls Conceded</th>
      <th>Offsides</th>
      <th>Yellow Cards</th>
      <th>Red Cards</th>
      <th>Subs on</th>
      <th>Subs off</th>
      <th>Players Used</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Croatia</td>
      <td>4</td>
      <td>13</td>
      <td>12</td>
      <td>51.9%</td>
      <td>16.0%</td>
      <td>32</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>13</td>
      <td>81.3%</td>
      <td>41</td>
      <td>62</td>
      <td>2</td>
      <td>9</td>
      <td>0</td>
      <td>9</td>
      <td>9</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Czech Republic</td>
      <td>4</td>
      <td>13</td>
      <td>18</td>
      <td>41.9%</td>
      <td>12.9%</td>
      <td>39</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>9</td>
      <td>60.1%</td>
      <td>53</td>
      <td>73</td>
      <td>8</td>
      <td>7</td>
      <td>0</td>
      <td>11</td>
      <td>11</td>
      <td>19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Denmark</td>
      <td>4</td>
      <td>10</td>
      <td>10</td>
      <td>50.0%</td>
      <td>20.0%</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>66.7%</td>
      <td>25</td>
      <td>38</td>
      <td>8</td>
      <td>4</td>
      <td>0</td>
      <td>7</td>
      <td>7</td>
      <td>15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>England</td>
      <td>5</td>
      <td>11</td>
      <td>18</td>
      <td>50.0%</td>
      <td>17.2%</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>22</td>
      <td>88.1%</td>
      <td>43</td>
      <td>45</td>
      <td>6</td>
      <td>5</td>
      <td>0</td>
      <td>11</td>
      <td>11</td>
      <td>16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>France</td>
      <td>3</td>
      <td>22</td>
      <td>24</td>
      <td>37.9%</td>
      <td>6.5%</td>
      <td>65</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>6</td>
      <td>54.6%</td>
      <td>36</td>
      <td>51</td>
      <td>5</td>
      <td>6</td>
      <td>0</td>
      <td>11</td>
      <td>11</td>
      <td>19</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Germany</td>
      <td>10</td>
      <td>32</td>
      <td>32</td>
      <td>47.8%</td>
      <td>15.6%</td>
      <td>80</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>62.6%</td>
      <td>63</td>
      <td>49</td>
      <td>12</td>
      <td>4</td>
      <td>0</td>
      <td>15</td>
      <td>15</td>
      <td>17</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Greece</td>
      <td>5</td>
      <td>8</td>
      <td>18</td>
      <td>30.7%</td>
      <td>19.2%</td>
      <td>32</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>13</td>
      <td>65.1%</td>
      <td>67</td>
      <td>48</td>
      <td>12</td>
      <td>9</td>
      <td>1</td>
      <td>12</td>
      <td>12</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
<p>7 rows × 35 columns</p>
</div>



### Step 13. Select all columns except the last 3.


```python
euro12.drop(euro12.index[-3:])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team</th>
      <th>Goals</th>
      <th>Shots on target</th>
      <th>Shots off target</th>
      <th>Shooting Accuracy</th>
      <th>% Goals-to-shots</th>
      <th>Total shots (inc. Blocked)</th>
      <th>Hit Woodwork</th>
      <th>Penalty goals</th>
      <th>Penalties not scored</th>
      <th>...</th>
      <th>Saves made</th>
      <th>Saves-to-shots ratio</th>
      <th>Fouls Won</th>
      <th>Fouls Conceded</th>
      <th>Offsides</th>
      <th>Yellow Cards</th>
      <th>Red Cards</th>
      <th>Subs on</th>
      <th>Subs off</th>
      <th>Players Used</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Croatia</td>
      <td>4</td>
      <td>13</td>
      <td>12</td>
      <td>51.9%</td>
      <td>16.0%</td>
      <td>32</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>13</td>
      <td>81.3%</td>
      <td>41</td>
      <td>62</td>
      <td>2</td>
      <td>9</td>
      <td>0</td>
      <td>9</td>
      <td>9</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Czech Republic</td>
      <td>4</td>
      <td>13</td>
      <td>18</td>
      <td>41.9%</td>
      <td>12.9%</td>
      <td>39</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>9</td>
      <td>60.1%</td>
      <td>53</td>
      <td>73</td>
      <td>8</td>
      <td>7</td>
      <td>0</td>
      <td>11</td>
      <td>11</td>
      <td>19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Denmark</td>
      <td>4</td>
      <td>10</td>
      <td>10</td>
      <td>50.0%</td>
      <td>20.0%</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>66.7%</td>
      <td>25</td>
      <td>38</td>
      <td>8</td>
      <td>4</td>
      <td>0</td>
      <td>7</td>
      <td>7</td>
      <td>15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>England</td>
      <td>5</td>
      <td>11</td>
      <td>18</td>
      <td>50.0%</td>
      <td>17.2%</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>22</td>
      <td>88.1%</td>
      <td>43</td>
      <td>45</td>
      <td>6</td>
      <td>5</td>
      <td>0</td>
      <td>11</td>
      <td>11</td>
      <td>16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>France</td>
      <td>3</td>
      <td>22</td>
      <td>24</td>
      <td>37.9%</td>
      <td>6.5%</td>
      <td>65</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>6</td>
      <td>54.6%</td>
      <td>36</td>
      <td>51</td>
      <td>5</td>
      <td>6</td>
      <td>0</td>
      <td>11</td>
      <td>11</td>
      <td>19</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Germany</td>
      <td>10</td>
      <td>32</td>
      <td>32</td>
      <td>47.8%</td>
      <td>15.6%</td>
      <td>80</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>62.6%</td>
      <td>63</td>
      <td>49</td>
      <td>12</td>
      <td>4</td>
      <td>0</td>
      <td>15</td>
      <td>15</td>
      <td>17</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Greece</td>
      <td>5</td>
      <td>8</td>
      <td>18</td>
      <td>30.7%</td>
      <td>19.2%</td>
      <td>32</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>13</td>
      <td>65.1%</td>
      <td>67</td>
      <td>48</td>
      <td>12</td>
      <td>9</td>
      <td>1</td>
      <td>12</td>
      <td>12</td>
      <td>20</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Italy</td>
      <td>6</td>
      <td>34</td>
      <td>45</td>
      <td>43.0%</td>
      <td>7.5%</td>
      <td>110</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>20</td>
      <td>74.1%</td>
      <td>101</td>
      <td>89</td>
      <td>16</td>
      <td>16</td>
      <td>0</td>
      <td>18</td>
      <td>18</td>
      <td>19</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Netherlands</td>
      <td>2</td>
      <td>12</td>
      <td>36</td>
      <td>25.0%</td>
      <td>4.1%</td>
      <td>60</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>12</td>
      <td>70.6%</td>
      <td>35</td>
      <td>30</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>7</td>
      <td>7</td>
      <td>15</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Poland</td>
      <td>2</td>
      <td>15</td>
      <td>23</td>
      <td>39.4%</td>
      <td>5.2%</td>
      <td>48</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>6</td>
      <td>66.7%</td>
      <td>48</td>
      <td>56</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
      <td>7</td>
      <td>17</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Portugal</td>
      <td>6</td>
      <td>22</td>
      <td>42</td>
      <td>34.3%</td>
      <td>9.3%</td>
      <td>82</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>71.5%</td>
      <td>73</td>
      <td>90</td>
      <td>10</td>
      <td>12</td>
      <td>0</td>
      <td>14</td>
      <td>14</td>
      <td>16</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Republic of Ireland</td>
      <td>1</td>
      <td>7</td>
      <td>12</td>
      <td>36.8%</td>
      <td>5.2%</td>
      <td>28</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>17</td>
      <td>65.4%</td>
      <td>43</td>
      <td>51</td>
      <td>11</td>
      <td>6</td>
      <td>1</td>
      <td>10</td>
      <td>10</td>
      <td>17</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Russia</td>
      <td>5</td>
      <td>9</td>
      <td>31</td>
      <td>22.5%</td>
      <td>12.5%</td>
      <td>59</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>77.0%</td>
      <td>34</td>
      <td>43</td>
      <td>4</td>
      <td>6</td>
      <td>0</td>
      <td>7</td>
      <td>7</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
<p>13 rows × 35 columns</p>
</div>



### Step 14. Present only the Shooting Accuracy from England, Italy and Russia


```python
euro12.set_index('Team',inplace=True)
euro12.loc[['England', 'Italy', 'Russia'], ['Shooting Accuracy']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Shooting Accuracy</th>
    </tr>
    <tr>
      <th>Team</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>England</th>
      <td>50.0%</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>43.0%</td>
    </tr>
    <tr>
      <th>Russia</th>
      <td>22.5%</td>
    </tr>
  </tbody>
</table>
</div>



---
End of Exercise2.ipynb
---

# Exercise 3. - GroupBy

### Introduction:

GroupBy can be summarized as Split-Apply-Combine.

Special thanks to: https://github.com/justmarkham for sharing the dataset and materials.

Check out this [Diagram](http://i.imgur.com/yjNkiwL.png)  

Check out [Alcohol Consumption Exercises Video Tutorial](https://youtu.be/az67CMdmS6s) to watch a data scientist go through the exercises


### Step 1. Import the necessary libraries


```python
import pandas as pd
import seaborn as sns
from dfply import *
```

### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/justmarkham/DAT8/master/data/drinks.csv). 

### Step 3. Assign it to a variable called drinks.


```python
url=('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/drinks.csv')
drinks = pd.read_csv(url)
drinks >>head(15)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>beer_servings</th>
      <th>spirit_servings</th>
      <th>wine_servings</th>
      <th>total_litres_of_pure_alcohol</th>
      <th>continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>AS</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>89</td>
      <td>132</td>
      <td>54</td>
      <td>4.9</td>
      <td>EU</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>25</td>
      <td>0</td>
      <td>14</td>
      <td>0.7</td>
      <td>AF</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Andorra</td>
      <td>245</td>
      <td>138</td>
      <td>312</td>
      <td>12.4</td>
      <td>EU</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Angola</td>
      <td>217</td>
      <td>57</td>
      <td>45</td>
      <td>5.9</td>
      <td>AF</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Antigua &amp; Barbuda</td>
      <td>102</td>
      <td>128</td>
      <td>45</td>
      <td>4.9</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Argentina</td>
      <td>193</td>
      <td>25</td>
      <td>221</td>
      <td>8.3</td>
      <td>SA</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Armenia</td>
      <td>21</td>
      <td>179</td>
      <td>11</td>
      <td>3.8</td>
      <td>EU</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Australia</td>
      <td>261</td>
      <td>72</td>
      <td>212</td>
      <td>10.4</td>
      <td>OC</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Austria</td>
      <td>279</td>
      <td>75</td>
      <td>191</td>
      <td>9.7</td>
      <td>EU</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Azerbaijan</td>
      <td>21</td>
      <td>46</td>
      <td>5</td>
      <td>1.3</td>
      <td>EU</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Bahamas</td>
      <td>122</td>
      <td>176</td>
      <td>51</td>
      <td>6.3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Bahrain</td>
      <td>42</td>
      <td>63</td>
      <td>7</td>
      <td>2.0</td>
      <td>AS</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Bangladesh</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>AS</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Barbados</td>
      <td>143</td>
      <td>173</td>
      <td>36</td>
      <td>6.3</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Step 4. Which continent drinks more beer on average?


```python
drinks >>group_by('continent') >>summarize(total_beer_servings=X.beer_servings.mean()) >>arrange(desc(X.total_beer_servings)) 

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>continent</th>
      <th>total_beer_servings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AF</td>
      <td>61.471698</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AS</td>
      <td>37.045455</td>
    </tr>
    <tr>
      <th>2</th>
      <td>EU</td>
      <td>193.777778</td>
    </tr>
    <tr>
      <th>3</th>
      <td>OC</td>
      <td>89.687500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SA</td>
      <td>175.083333</td>
    </tr>
  </tbody>
</table>
</div>



### Step 5. For each continent print the statistics for wine consumption.


```python


d = drinks.groupby('continent').agg({
'wine_servings': 'describe'
})
d
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">wine_servings</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>continent</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AF</th>
      <td>53.0</td>
      <td>16.264151</td>
      <td>38.846419</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>13.00</td>
      <td>233.0</td>
    </tr>
    <tr>
      <th>AS</th>
      <td>44.0</td>
      <td>9.068182</td>
      <td>21.667034</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>8.00</td>
      <td>123.0</td>
    </tr>
    <tr>
      <th>EU</th>
      <td>45.0</td>
      <td>142.222222</td>
      <td>97.421738</td>
      <td>0.0</td>
      <td>59.0</td>
      <td>128.0</td>
      <td>195.00</td>
      <td>370.0</td>
    </tr>
    <tr>
      <th>OC</th>
      <td>16.0</td>
      <td>35.625000</td>
      <td>64.555790</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>8.5</td>
      <td>23.25</td>
      <td>212.0</td>
    </tr>
    <tr>
      <th>SA</th>
      <td>12.0</td>
      <td>62.416667</td>
      <td>88.620189</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>12.0</td>
      <td>98.50</td>
      <td>221.0</td>
    </tr>
  </tbody>
</table>
</div>



### Step 6. Print the mean alcohol consumption per continent for every column


```python
drinks.groupby('continent').mean()
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\groupby\groupby.py:1942, in GroupBy._agg_py_fallback(self, how, values, ndim, alt)
       1941 try:
    -> 1942     res_values = self._grouper.agg_series(ser, alt, preserve_dtype=True)
       1943 except Exception as err:
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\groupby\ops.py:864, in BaseGrouper.agg_series(self, obj, func, preserve_dtype)
        862     preserve_dtype = True
    --> 864 result = self._aggregate_series_pure_python(obj, func)
        866 npvalues = lib.maybe_convert_objects(result, try_float=False)
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\groupby\ops.py:885, in BaseGrouper._aggregate_series_pure_python(self, obj, func)
        884 for i, group in enumerate(splitter):
    --> 885     res = func(group)
        886     res = extract_result(res)
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\groupby\groupby.py:2454, in GroupBy.mean.<locals>.<lambda>(x)
       2451 else:
       2452     result = self._cython_agg_general(
       2453         "mean",
    -> 2454         alt=lambda x: Series(x, copy=False).mean(numeric_only=numeric_only),
       2455         numeric_only=numeric_only,
       2456     )
       2457     return result.__finalize__(self.obj, method="groupby")
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\series.py:6549, in Series.mean(self, axis, skipna, numeric_only, **kwargs)
       6541 @doc(make_doc("mean", ndim=1))
       6542 def mean(
       6543     self,
       (...)   6547     **kwargs,
       6548 ):
    -> 6549     return NDFrame.mean(self, axis, skipna, numeric_only, **kwargs)
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\generic.py:12420, in NDFrame.mean(self, axis, skipna, numeric_only, **kwargs)
      12413 def mean(
      12414     self,
      12415     axis: Axis | None = 0,
       (...)  12418     **kwargs,
      12419 ) -> Series | float:
    > 12420     return self._stat_function(
      12421         "mean", nanops.nanmean, axis, skipna, numeric_only, **kwargs
      12422     )
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\generic.py:12377, in NDFrame._stat_function(self, name, func, axis, skipna, numeric_only, **kwargs)
      12375 validate_bool_kwarg(skipna, "skipna", none_allowed=False)
    > 12377 return self._reduce(
      12378     func, name=name, axis=axis, skipna=skipna, numeric_only=numeric_only
      12379 )
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\series.py:6457, in Series._reduce(self, op, name, axis, skipna, numeric_only, filter_type, **kwds)
       6453     raise TypeError(
       6454         f"Series.{name} does not allow {kwd_name}={numeric_only} "
       6455         "with non-numeric dtypes."
       6456     )
    -> 6457 return op(delegate, skipna=skipna, **kwds)
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\nanops.py:147, in bottleneck_switch.__call__.<locals>.f(values, axis, skipna, **kwds)
        146 else:
    --> 147     result = alt(values, axis=axis, skipna=skipna, **kwds)
        149 return result
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\nanops.py:404, in _datetimelike_compat.<locals>.new_func(values, axis, skipna, mask, **kwargs)
        402     mask = isna(values)
    --> 404 result = func(values, axis=axis, skipna=skipna, mask=mask, **kwargs)
        406 if datetimelike:
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\nanops.py:720, in nanmean(values, axis, skipna, mask)
        719 the_sum = values.sum(axis, dtype=dtype_sum)
    --> 720 the_sum = _ensure_numeric(the_sum)
        722 if axis is not None and getattr(the_sum, "ndim", False):
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\nanops.py:1701, in _ensure_numeric(x)
       1699 if isinstance(x, str):
       1700     # GH#44008, GH#36703 avoid casting e.g. strings to numeric
    -> 1701     raise TypeError(f"Could not convert string '{x}' to numeric")
       1702 try:
    

    TypeError: Could not convert string 'AlgeriaAngolaBeninBotswanaBurkina FasoBurundiCote d'IvoireCabo VerdeCameroonCentral African RepublicChadComorosCongoDR CongoDjiboutiEgyptEquatorial GuineaEritreaEthiopiaGabonGambiaGhanaGuineaGuinea-BissauKenyaLesothoLiberiaLibyaMadagascarMalawiMaliMauritaniaMauritiusMoroccoMozambiqueNamibiaNigerNigeriaRwandaSao Tome & PrincipeSenegalSeychellesSierra LeoneSomaliaSouth AfricaSudanSwazilandTogoTunisiaUgandaTanzaniaZambiaZimbabwe' to numeric

    
    The above exception was the direct cause of the following exception:
    

    TypeError                                 Traceback (most recent call last)

    Cell In[37], line 1
    ----> 1 drinks.groupby('continent').mean()
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\groupby\groupby.py:2452, in GroupBy.mean(self, numeric_only, engine, engine_kwargs)
       2445     return self._numba_agg_general(
       2446         grouped_mean,
       2447         executor.float_dtype_mapping,
       2448         engine_kwargs,
       2449         min_periods=0,
       2450     )
       2451 else:
    -> 2452     result = self._cython_agg_general(
       2453         "mean",
       2454         alt=lambda x: Series(x, copy=False).mean(numeric_only=numeric_only),
       2455         numeric_only=numeric_only,
       2456     )
       2457     return result.__finalize__(self.obj, method="groupby")
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\groupby\groupby.py:1998, in GroupBy._cython_agg_general(self, how, alt, numeric_only, min_count, **kwargs)
       1995     result = self._agg_py_fallback(how, values, ndim=data.ndim, alt=alt)
       1996     return result
    -> 1998 new_mgr = data.grouped_reduce(array_func)
       1999 res = self._wrap_agged_manager(new_mgr)
       2000 if how in ["idxmin", "idxmax"]:
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\internals\managers.py:1469, in BlockManager.grouped_reduce(self, func)
       1465 if blk.is_object:
       1466     # split on object-dtype blocks bc some columns may raise
       1467     #  while others do not.
       1468     for sb in blk._split():
    -> 1469         applied = sb.apply(func)
       1470         result_blocks = extend_blocks(applied, result_blocks)
       1471 else:
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\internals\blocks.py:393, in Block.apply(self, func, **kwargs)
        387 @final
        388 def apply(self, func, **kwargs) -> list[Block]:
        389     """
        390     apply the function to my values; return a block if we are not
        391     one
        392     """
    --> 393     result = func(self.values, **kwargs)
        395     result = maybe_coerce_values(result)
        396     return self._split_op_result(result)
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\groupby\groupby.py:1995, in GroupBy._cython_agg_general.<locals>.array_func(values)
       1992     return result
       1994 assert alt is not None
    -> 1995 result = self._agg_py_fallback(how, values, ndim=data.ndim, alt=alt)
       1996 return result
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\groupby\groupby.py:1946, in GroupBy._agg_py_fallback(self, how, values, ndim, alt)
       1944     msg = f"agg function failed [how->{how},dtype->{ser.dtype}]"
       1945     # preserve the kind of exception that raised
    -> 1946     raise type(err)(msg) from err
       1948 if ser.dtype == object:
       1949     res_values = res_values.astype(object, copy=False)
    

    TypeError: agg function failed [how->mean,dtype->object]


### Step 7. Print the median alcohol consumption per continent for every column


```python
drinks.groupby('continent').median()
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\groupby\groupby.py:1942, in GroupBy._agg_py_fallback(self, how, values, ndim, alt)
       1941 try:
    -> 1942     res_values = self._grouper.agg_series(ser, alt, preserve_dtype=True)
       1943 except Exception as err:
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\groupby\ops.py:864, in BaseGrouper.agg_series(self, obj, func, preserve_dtype)
        862     preserve_dtype = True
    --> 864 result = self._aggregate_series_pure_python(obj, func)
        866 npvalues = lib.maybe_convert_objects(result, try_float=False)
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\groupby\ops.py:885, in BaseGrouper._aggregate_series_pure_python(self, obj, func)
        884 for i, group in enumerate(splitter):
    --> 885     res = func(group)
        886     res = extract_result(res)
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\groupby\groupby.py:2534, in GroupBy.median.<locals>.<lambda>(x)
       2461 """
       2462 Compute median of groups, excluding missing values.
       2463 
       (...)   2530 Freq: MS, dtype: float64
       2531 """
       2532 result = self._cython_agg_general(
       2533     "median",
    -> 2534     alt=lambda x: Series(x, copy=False).median(numeric_only=numeric_only),
       2535     numeric_only=numeric_only,
       2536 )
       2537 return result.__finalize__(self.obj, method="groupby")
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\series.py:6559, in Series.median(self, axis, skipna, numeric_only, **kwargs)
       6551 @doc(make_doc("median", ndim=1))
       6552 def median(
       6553     self,
       (...)   6557     **kwargs,
       6558 ):
    -> 6559     return NDFrame.median(self, axis, skipna, numeric_only, **kwargs)
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\generic.py:12431, in NDFrame.median(self, axis, skipna, numeric_only, **kwargs)
      12424 def median(
      12425     self,
      12426     axis: Axis | None = 0,
       (...)  12429     **kwargs,
      12430 ) -> Series | float:
    > 12431     return self._stat_function(
      12432         "median", nanops.nanmedian, axis, skipna, numeric_only, **kwargs
      12433     )
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\generic.py:12377, in NDFrame._stat_function(self, name, func, axis, skipna, numeric_only, **kwargs)
      12375 validate_bool_kwarg(skipna, "skipna", none_allowed=False)
    > 12377 return self._reduce(
      12378     func, name=name, axis=axis, skipna=skipna, numeric_only=numeric_only
      12379 )
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\series.py:6457, in Series._reduce(self, op, name, axis, skipna, numeric_only, filter_type, **kwds)
       6453     raise TypeError(
       6454         f"Series.{name} does not allow {kwd_name}={numeric_only} "
       6455         "with non-numeric dtypes."
       6456     )
    -> 6457 return op(delegate, skipna=skipna, **kwds)
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\nanops.py:147, in bottleneck_switch.__call__.<locals>.f(values, axis, skipna, **kwds)
        146 else:
    --> 147     result = alt(values, axis=axis, skipna=skipna, **kwds)
        149 return result
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\nanops.py:787, in nanmedian(values, axis, skipna, mask)
        786     if inferred in ["string", "mixed"]:
    --> 787         raise TypeError(f"Cannot convert {values} to numeric")
        788 try:
    

    TypeError: Cannot convert ['Algeria' 'Angola' 'Benin' 'Botswana' 'Burkina Faso' 'Burundi'
     "Cote d'Ivoire" 'Cabo Verde' 'Cameroon' 'Central African Republic' 'Chad'
     'Comoros' 'Congo' 'DR Congo' 'Djibouti' 'Egypt' 'Equatorial Guinea'
     'Eritrea' 'Ethiopia' 'Gabon' 'Gambia' 'Ghana' 'Guinea' 'Guinea-Bissau'
     'Kenya' 'Lesotho' 'Liberia' 'Libya' 'Madagascar' 'Malawi' 'Mali'
     'Mauritania' 'Mauritius' 'Morocco' 'Mozambique' 'Namibia' 'Niger'
     'Nigeria' 'Rwanda' 'Sao Tome & Principe' 'Senegal' 'Seychelles'
     'Sierra Leone' 'Somalia' 'South Africa' 'Sudan' 'Swaziland' 'Togo'
     'Tunisia' 'Uganda' 'Tanzania' 'Zambia' 'Zimbabwe'] to numeric

    
    The above exception was the direct cause of the following exception:
    

    TypeError                                 Traceback (most recent call last)

    Cell In[38], line 1
    ----> 1 drinks.groupby('continent').median()
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\groupby\groupby.py:2532, in GroupBy.median(self, numeric_only)
       2459 @final
       2460 def median(self, numeric_only: bool = False) -> NDFrameT:
       2461     """
       2462     Compute median of groups, excluding missing values.
       2463 
       (...)   2530     Freq: MS, dtype: float64
       2531     """
    -> 2532     result = self._cython_agg_general(
       2533         "median",
       2534         alt=lambda x: Series(x, copy=False).median(numeric_only=numeric_only),
       2535         numeric_only=numeric_only,
       2536     )
       2537     return result.__finalize__(self.obj, method="groupby")
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\groupby\groupby.py:1998, in GroupBy._cython_agg_general(self, how, alt, numeric_only, min_count, **kwargs)
       1995     result = self._agg_py_fallback(how, values, ndim=data.ndim, alt=alt)
       1996     return result
    -> 1998 new_mgr = data.grouped_reduce(array_func)
       1999 res = self._wrap_agged_manager(new_mgr)
       2000 if how in ["idxmin", "idxmax"]:
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\internals\managers.py:1469, in BlockManager.grouped_reduce(self, func)
       1465 if blk.is_object:
       1466     # split on object-dtype blocks bc some columns may raise
       1467     #  while others do not.
       1468     for sb in blk._split():
    -> 1469         applied = sb.apply(func)
       1470         result_blocks = extend_blocks(applied, result_blocks)
       1471 else:
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\internals\blocks.py:393, in Block.apply(self, func, **kwargs)
        387 @final
        388 def apply(self, func, **kwargs) -> list[Block]:
        389     """
        390     apply the function to my values; return a block if we are not
        391     one
        392     """
    --> 393     result = func(self.values, **kwargs)
        395     result = maybe_coerce_values(result)
        396     return self._split_op_result(result)
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\groupby\groupby.py:1995, in GroupBy._cython_agg_general.<locals>.array_func(values)
       1992     return result
       1994 assert alt is not None
    -> 1995 result = self._agg_py_fallback(how, values, ndim=data.ndim, alt=alt)
       1996 return result
    

    File c:\Users\barte\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\groupby\groupby.py:1946, in GroupBy._agg_py_fallback(self, how, values, ndim, alt)
       1944     msg = f"agg function failed [how->{how},dtype->{ser.dtype}]"
       1945     # preserve the kind of exception that raised
    -> 1946     raise type(err)(msg) from err
       1948 if ser.dtype == object:
       1949     res_values = res_values.astype(object, copy=False)
    

    TypeError: agg function failed [how->median,dtype->object]


### Step 8. Print the mean, min and max values for spirit consumption.
#### This time output a DataFrame


```python
drinks.groupby('continent').spirit_servings.agg(['mean', 'min', 'max'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>min</th>
      <th>max</th>
    </tr>
    <tr>
      <th>continent</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AF</th>
      <td>16.339623</td>
      <td>0</td>
      <td>152</td>
    </tr>
    <tr>
      <th>AS</th>
      <td>60.840909</td>
      <td>0</td>
      <td>326</td>
    </tr>
    <tr>
      <th>EU</th>
      <td>132.555556</td>
      <td>0</td>
      <td>373</td>
    </tr>
    <tr>
      <th>OC</th>
      <td>58.437500</td>
      <td>0</td>
      <td>254</td>
    </tr>
    <tr>
      <th>SA</th>
      <td>114.750000</td>
      <td>25</td>
      <td>302</td>
    </tr>
  </tbody>
</table>
</div>



---
End of Exercise3.ipynb
---
