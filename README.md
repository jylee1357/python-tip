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
path_gpkg = 'C:/Users/SUNDO/Desktop/workplace/jb_weather/??????????????????/*.gpkg'
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
\s: ?????? ?????? ??? ??? ????????? ????????????
\b: ????????? ??????, ????????? ????????? ???, ??????, ??????, ???, ??????, ?????????, ???????????? ?????? ??? ??? ??????.
?: ?????? ?????? ?????? ?????? ????????? 1??? ?????? 0?????????.
\w: ????????? ?????? ??? ?????? ??????. ????????? ????????????, ??????, ?????????????????? ????????????.
*: 0??? ????????????
[bc]: b?????? c??? ????????????
(b|c): b ?????? c??? ????????????

r'(?P<road_sd>\b[???-???]+[??????]\s+\b)?
(?P<road_sg>\b[???-???]+[??????]\s+\b)?
(?P<road_g>\b[???-???]+[???]\s+\b)?
(?P<road_em>\b\w+[??????]\s+\b)?
(?P<road_rg>\b\w+[??????]\s*)+(?P<road_jb>([,|\s]*\d+-?\d*[^???-???]*)+)'
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
