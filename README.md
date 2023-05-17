# python-tip
```
#Sorting list using Lambda function

Example 1)
def solution(strings, n):
    strings.sort() 
    return sorted(strings, key=lambda x:x[n])
    
Example 2)
def solution(strings, n):
    return sorted(strings, key=lambda x:(x[n],x))
```

```
#Using 'OR' in if Statement (s[i]  == 'p' or s[i] == 'p')

Example 1)
def solution(s):
    cnt_p = 0
    cnt_y = 0
    for i in range(len(s)):
        if s[i] == 'p' or s[i] == "P":
            cnt_p += 1
        elif s[i] == 'y' or s[i] == "Y":
            cnt_y += 1
        else:
            cnt_p += 0
            cnt_y += 0
    if cnt_p == cnt_y:
        return True
    else:
        return False
    
```
```
#Instead of using two for loops use "and" in if statement!
Example 1)
def solution(s):
    if (len(s)==4 or len(s)==6) and (any(c.isalpha() for c in the_string)) == False:
        return True
    else:
        return False
```
```
# Counter, elements
    Counter1 = Counter(str1_lst)
    Counter2 = Counter(str2_lst)
    
    inter = list((Counter1 & Counter2).elements())
    union = list((Counter1 | Counter2).elements())
```

```
# Network Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

df_w = pd.read_csv('network_weight.csv')
df_p = pd.read_csv('network_pos.csv')

size = 10
edge_weights = []
for i in range(len(df_w)):
    for j in range(len(df_w.columns)):
        edge_weights.append(df_w.iloc[i][j]*size)

G = nx.Graph()

for i in range(len(df_w.columns)):
    G.add_node(df_w.columns[i])

for i in range(len(df_w.columns)):
    for j in range(len(df_w.columns)):
        G.add_edge(df_w.columns[i],df_w.columns[j])

pos = {}
for i in range(len(df_w.columns)):
    node = df_w.columns[i]
    pos[node] = (df_p[node][0],df_p[node][1])

nx.draw(G, pos, with_labels=True,font_size=16, node_size = 1000, node_color='k', font_color='w', width=edge_weights)

plt.show()
```
```
#simple list index
s = "{{2},{2,1},{2,1,3},{2,1,3,4}}"
s[2:-2] = '2},{2,1},{2,1,3},{2,1,3,4'

#split tip
ls = sorted([s.split(',') for s in s[2:-2].split('},{')], key=len)
```
```
# How to save index and element at the same time using deque
priorities = [2,3,1,2]
d = deque([(i,v) for i,v in enumerate(priorities)])
```
```
# Mixing up the orders of list (getting all possible outcomes)
import itertools
list(itertools.permutations([1, 2, 3]))
```
```
def solution(numbers):
    numbers = list(map(str, numbers)) #converting int to string
    numbers.sort(key=lambda x: x * 3, reverse=True) #sorting using lambda key function
    return str(int(''.join(numbers)))
```
```
# reading dataset (when there are many) and reading them as dictionary
path_gpkg = 'C:/Users/SUNDO/Desktop/workplace/jb_weather/날씨결합명소/*.gpkg'
gpkg_list = glob(path_gpkg)

gdf_dict={}
for i, gpkg in enumerate(gpkg_list):
    fn = gpkg.split('\\')[-1]
    gdf = gpd.read_file(gpkg)
    cols = ['id']
    cols.append(gdf.columns[2])
    gdf_dict[i] = gdf[cols]
    
# combining all the dataset in the dictionary
df = pd.concat(gdf_dict,axis=1)
```
```
# By using zfill, you can fill up the empty digits with 0 (ex: 6 -> 06)
for r in raster_list:
    y,m = r.split('\\')[-1].split('.')[0].split('_')
    m = m.zfill(2)
    y = y[2:]
    input_tif = r
    new_col = f'{y}_{m}_'
    out_path = f'{out_dir}tmp{new_col}.gpkg'
    processing.run("native:zonalstatisticsfb", 
                   {'INPUT': input_shp,
                    'INPUT_RASTER': input_tif,
                    'RASTER_BAND':1,
                    'COLUMN_PREFIX': new_col,
                    'STATISTICS':[2],'OUTPUT':out_path})    
 ```
```
# When you want to add a column in a dataframe that show the length of other column just use
df['count'] = df['column_name'].apply(lambda x: len(x))
```

```
Regex Compilation
\s: equal as either space or tab
\b: boundary of words, beginning or end of a letter, space, tab, comma, apostrophe, dash, etc. 
?: the letter prior to contains either 0 or 1 group of letters 
\w: word, alphabet (including capital), numbers, underscore 
*: more than 0
[bc]: either contains b or c
(b|c): either contains b or c

r'(?P<road_sd>\b[가-힣]+[시도]\s+\b)?
(?P<road_sg>\b[가-힣]+[시군]\s+\b)?
(?P<road_g>\b[가-힣]+[구]\s+\b)?
(?P<road_em>\b\w+[읍면]\s+\b)?
(?P<road_rg>\b\w+[로길]\s*)+(?P<road_jb>([,|\s]*\d+-?\d*[^가-힣]*)+)'
 ```

```
Splitting the element in a row and appending it as a new row
def split_address(df):
    
    for i in range (len(df)):
        aa = df['col_name'][i].split(",")
        
        for j in range(len(aa)):
            df = df.append({'idx': i, 'col_name':aa[j]},ignore_index = True)
        
    return df
 ```

```
#Loading dataset using glob
shp_lst2 = glob('./csv/*.csv') #Assigning path for dataset, use wildcard (*) when the name of datasets have a pattern.

# Designate file name and save it as a new dataframe
for s in shp_list: 
    fn = s.split('\\')[-2].split('_')[-1]
    print(fn)
    gdf = gpd.read_file(s,encoding='cp949')
    df = PNU_2_df(gdf)
    dst = f'./csv/{fn}.csv'
    print(dst)
    df.to_csv(dst,index=False,encoding='utf8')
        
    return df
 ```
```
# Split the full xy coordinates into two different columns using lambda 
df_geo_pnu['x'] = df_geo_pnu['xy'].apply(lambda x : x.split(",")[0])
df_geo_pnu['y'] = df_geo_pnu['xy'].apply(lambda x : x.split(",")[-1])
 ```
 ```
#Remove parenthesis from column in xy points 
df['x'] = df['x'].str.replace("(","")
df['y'] = df['y'].str.replace(")","")
```
```
# Merge dataframe different occasions
df1 = pd.DataFrame({"key": list("bbacaab"), "data1":range(7)})
df2 = pd.DataFrame({"data2":range(3), "key": list("abd")})

#When there is a common column and you want to merge it on either left or right
pd.merge(df1,df2, on = "key", how = "left")
pd.merge(df1,df2, on = "key", how = "right")

#When there is no common column but you want to merge it using two different key columns (use left_on and right_on)
df5 = pd.DataFrame({"key2": list("bbacaab"), "data1":range(7)})
df6 = pd.DataFrame({"data2":range(3), "key": list("abd")})

pd.merge(df5,df6, left_on = "key2", right_on = "key")
 ```
```
#How to suppress scientific notation
gdf_pnu['PNU_CD'].apply(lambda x: '%.0f' % x)
```
```
#Using API
import requests
import pprint
import json

#인증키 입력
encoding = "d3rSSDLIYE05PVPSUtRYTVa%2BuRy%2FtXEDhS%2BBOlSn6Nc%2Bi%2Fsy9WXlgaM5%2BJOmVZyoTv3TNk5j%2BJkYu8wNHusX0Q%3D%3D"
decoding = "d3rSSDLIYE05PVPSUtRYTVa+uRy/tXEDhS+BOlSn6Nc+i/sy9WXlgaM5+JOmVZyoTv3TNk5j+JkYu8wNHusX0Q=="

#url 입력
url =  "http://apis.data.go.kr/B552584/EvCharger/getChargerInfo?serviceKey=d3rSSDLIYE05PVPSUtRYTVa%2BuRy%2FtXEDhS%2BBOlSn6Nc%2Bi%2Fsy9WXlgaM5%2BJOmVZyoTv3TNk5j%2BJkYu8wNHusX0Q%3D%3D&numOfRows=100000&pageNo=1&dataType=json&zscode=36110"

response = requests.get(url)
contents = response.text

pp = pprint.PrettyPrinter(indent=4)
print(pp.pprint(contents))

from os import name
import xml.etree.ElementTree as et
import pandas as pd
import bs4
from lxml import html
from urllib.parse import urlencode, quote_plus, unquote


#bs4 사용하여 item 태그 분리

xml_obj = bs4.BeautifulSoup(contents,'lxml-xml')
rows = xml_obj.findAll('item')
print(rows)

row_list = [] # 행값
name_list = [] # 열이름값
value_list = [] #데이터값

# xml 안의 데이터 수집
for i in range(0, len(rows)):
    columns = rows[i].find_all()
    #첫째 행 데이터 수집
    for j in range(0,len(columns)):
        if i ==0:
            # 컬럼 이름 값 저장
            name_list.append(columns[j].name)
        # 컬럼의 각 데이터 값 저장
        value_list.append(columns[j].text)
    # 각 행의 value값 전체 저장
    row_list.append(value_list)
    # 데이터 리스트 값 초기화
    value_list=[]
    
df_sejong = pd.DataFrame(row_list, columns=name_list)
len(df_sejong)
```
```
#How to groupby and sum based on one column while keeping index and other columns
new_df = jb_senior_pharm_intersection.groupby(['index'], as_index = False).sum()
```
```
#How to get rid of "Unnamed :0" in pandas
df_pharm = pd.read_csv("C:/Users/SUNDO/Desktop/workplace/2023 전라북도 빅데이터 분석사업/공공야간약국 데이터/전북약국현황.csv", encoding = "cp949", index_col = 0)
```
```
#How to read multiple csvs individually 
import pandas as pd
import os  
# assign dataset names
list_of_names = os.listdir("D:/workplace/연령별 인구현황/고령층")
 
# create empty list
df_lst = []
 
# append datasets into the list
for i in range(len(list_of_names)):
    temp_df = pd.read_csv("D:/workplace/연령별 인구현황/고령층/"+list_of_names[i], encoding = "cp949")
    df_lst.append(temp_df) 
```
```
#Filter rows based on contents of each row 
df[df['ids'].str.contains("ball")]
```
```
#Calculating Z-score using dataframe columns 
df_senior_infant['zscore(생활인구)'] = (df_senior_infant['생활인구수(유아,노년층)'] - df_senior_infant['생활인구수(유아,노년층)'].mean())/df_senior_infant['생활인구수(유아,노년층)'].std(ddof=0)
```
```
#Calculating decile rank using dataframe columns
df['시군구 별 Decile_rank'] = pd.qcut(df['Percentile Mean'],10,labels = False)
```
