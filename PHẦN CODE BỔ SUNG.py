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


# In[7]:


df['trip'] = df['source_city'] + '_' + df['destination_city']  # Tạo cột 'trip'
trip_counts = df['trip'].value_counts()  # Đếm số lượng giá trị trong mỗi nhóm
print(trip_counts)


# In[12]:


# In ra 5 giá trị lớn nhất theo mỗi hãng bay
for a in df['airline'].unique():
    temp = df[df['airline'] == a]['trip'].value_counts().nlargest(5)
    print('\nTop 5 trips for', a)
    print(temp)


# In[13]:


results = pd.DataFrame(columns=['airline', 'trip', 'count'])
for a in df['airline'].unique():
    temp = df[df['airline'] == a]['trip'].value_counts().nlargest(5)
    temp_df = pd.DataFrame({
        'airline': a,
        'trip': temp.index,
        'count': temp.values
    })
    results = pd.concat([results, temp_df])


# In[15]:


import seaborn as sns


# In[34]:


# Vẽ biểu đồ cột cho kết quả
fig, ax = plt.subplots(figsize=(20, 15))
ax = sns.barplot(x='trip', y='count', hue='airline',palette=['#191970','#ffdead', '#ffa500', '#4169e1', '#dc143c','#ADA2FF'],data=results)

# Định dạng biểu đồ và hiển thị
plt.xticks(rotation=45, ha='right')
plt.xlabel('Trip')
plt.ylabel('Count')
plt.title('Top 5 trips per airline')
plt.legend(loc='upper right')
plt.show()


# In[ ]:




