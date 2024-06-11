#!/usr/bin/env python
# coding: utf-8

# ### Bài tập 1.7

# ### Yêu cầu
# ## <span style='color:blue'> Bài tập 1.7. Scatter plots với marginal histogram
#     
# - Sử dụng bộ dữ liệu như trong Bài tập 1.6. Hãy lọc bộ dữ liệu này sao cho bạn có samples chứa thông tin về body mass và maximum longevity. Chọn tất cả các samples của class Aves với a body mass nhỏ hơn 20,000.
# - Hãy vẽ một scatter plot cùng với marginal histograms. Thêm vào labels và figure title.

# In[3]:


### khai báo hàm thư viện cần dùng
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
import cv2


# In[4]:


# Load dataset
data = pd.read_csv("anage_data.csv",index_col = 0)
data


# In[5]:


data["Maximum longevity (yrs)"].fillna(1, inplace = True)
data["Body mass (g)"].fillna(2, inplace = True)
bm_mxmlongevity = data[["Body mass (g)","Maximum longevity (yrs)"]]
bm_mxmlongevity


# In[6]:


filtered_data1 = data.loc[data['Class'] == 'Aves' ]
filtered_data1


# In[22]:


filtered_data2 = data.loc[(data['Class'] == 'Aves') & (data['Body mass (g)']<=20000)] 
filtered_data2


# In[26]:


# create figure
plt.figure(figsize=(10, 7))
sns.jointplot(data = data, x= "Body mass (g)", y ="Maximum longevity (yrs)")
plt.xlabel('Body mass')
plt.ylabel('Maximum longevity(yrs)')
plt.title('the complex graph describes two scores', fontsize = 15, loc = 'left')
# Show plot
plt.show()


# In[ ]:




