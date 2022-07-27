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
