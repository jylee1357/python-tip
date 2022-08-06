# python-tip
```
#Sorting list using lambda

Example 1)
def solution(strings, n):
    strings.sort() 
    return sorted(strings, key=lambda x:x[n])
    
Example 2)
def solution(strings, n):
    return sorted(strings, key=lambda x:(x[n],x))
```

```
#Using 'OR' in if statement (s[i]  == 'p' or s[i] == 'p')

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

```



