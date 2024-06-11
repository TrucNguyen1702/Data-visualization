#!/usr/bin/env python
# coding: utf-8

# ## PHẦN BỔ SUNG CODE

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("Clean_Dataset.csv", index_col = 0)


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df_grouped = df.groupby('airline').agg(
    {'source_city': pd.Series.mode, 'destination_city': pd.Series.mode})
df_grouped


# ## Vẽ bar plot

# In[6]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[7]:


result = df.groupby(['source_city', 'destination_city','airline']).size().reset_index(name='trip')


# In[8]:


top_5_trips = result.nlargest(5, 'trip')
print(top_5_trips)


# In[9]:


# In ra thông tin về các trip lớn nhất
for i in range(len(top_5_trips)):
    print(f"Trip {i+1}: {top_5_trips.iloc[i]['trip']} trips from {top_5_trips.iloc[i]['source_city']} to {top_5_trips.iloc[i]['destination_city']}, operated by {top_5_trips.iloc[i]['airline']}")


# In[21]:


travel = top_5_trips['source_city'] +'_'+ top_5_trips['destination_city']
travel


# In[42]:


# Tạo biểu đồ
fig = plt.figure(figsize=(15,6))
ax = fig.add_subplot(111)
bars= ax.bar(travel, top_5_trips['trip'],color=['blue', 'red', 'green', 'orange', 'purple'])
ax.set_xlabel('Airlines')
ax.set_ylabel('Number of trips')
ax.set_title('Top 5 popular flight routes')
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{int(height)}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

autolabel(bars)
ax.set_xticklabels(travel, rotation=45)
plt.show()


# In[ ]:




