import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
a = [[1,2,3],[0,1,2],[3,4,5]]
b = [[1,2,3],[0,1,2],[3,4,5]]

df = pd.DataFrame(a)

plt.figure()
plt.scatter(a,a,label="Chunk 2", c="w")
plt.scatter(b,b,label="b")
l = plt.legend()
for text in l.get_texts():
    if text.get_text() == "Chunk 1": 
        text.set_color('b') 
    elif text.get_text() == "Chunk 2": 
        text.set_color('r') 
    else: 
        text.set_color('g')
plt.show()
sys.exit()
print(df)


df = pd.melt(frame = df,
             var_name = 'column',
             value_name = 'value')

fig, ax = plt.subplots()

sns.lineplot(ax = ax,
             data = df,
             x = 'column',
             y = 'value')

plt.show()