#!/usr/bin/env python
# coding: utf-8

# # <center> <span style='color:red'> Homework 1: Vẽ một số dạng biểu đồ cơ bản trong Matplotlib và Seaborn </span></center> 
# 

# ## <span style='color:blue'> Bài tập 1.1. Line plots:
# 
# Trong bài tập này, chúng ta tạo line plot để thể hiện stock trends. 
# 
# Bài toán: Bạn có ý định đầu tư vào stocks và bạn tải dữ liệu về the stock prices đối với “big five”: Amazon, Google, Apple, Facebook, and Microsoft. Hãy vẽ line plot để thể hiện stock trends. 

# In[1]:


### khai báo hàm thư viện cần dùng
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
import cv2


# Sử dụng pandas để đọc dữ liệu từ máy tính (hoặc từ đường dẫn)

# In[2]:


# load datasets
# đọc dữ liệu
data1 = pd.read_csv("AAPL_data.csv")
data1


# In[3]:


data2 = pd.read_csv("AMZN_data.csv")
data2


# In[4]:


data3 = pd.read_csv("FB_data.csv")
data3


# In[5]:


data4 = pd.read_csv("GOOGL_data.csv")
data4


# Use Matplotlib to create a line chart visualizing the closing prices (using feature 'close' in the dataset) for the past five years (whole data sequence) for all five companies. Add labels, titles, and a legend to make the visualization self-explanatory. Use plt.grid() to add a grid to your plot.

# In[6]:


date = data1.set_index('date')
col_to_plot = 'close'


# In[7]:


list(data1.columns)


# In[8]:


data1.info()


# In[9]:


# Create figure (figure size of 16"x8", dpi of 300) & Plot data
plt.figure(figsize=(16,8), dpi=300)
x1 = data1['date']
y1 = data1['close']
plt.plot(x1,y1,color = 'green')
x2 = data2['date']
y2 = data2['close']
plt.plot(x2,y2, color = 'blue')
x3 = data3['date']
y3 = data3['close']
plt.plot(x3,y3,color = 'red')
x4 = data4['date']
y4 = data4['close']
plt.plot(x4,y4, color ='orange')
# Specify ticks for x- and y-axis
plt.xticks(np.arange(0, 1260, 40), rotation=70)
plt.yticks(np.arange(0, 1450, 100))
plt.title('Visualizing the closing prices') # Add title
#add labels
plt.ylabel('close')
plt.xlabel('date')
plt.grid(color = 'white') # Add grid
labels = ['AAPL_data','AMZN_data','FB_data','GOOGL_data'] # add label for legend
plt.legend(labels,loc= 8) # Add legend
plt.show() # Show plot


# 

# ## <span style='color:blue'> Bài tập 1.2. Pie charts:
#     
# Với dataset về mức sử dụng nước, hãy vẽ pie chart để thể hiện mức sử dụng nước (water usage). Highlight một 'usage' (tuỳ chọn) bằng việc sử dụng the explode parameter. Đồng thời, thể hiện tỷ lệ cho mỗi slice và thêm title.
#     
# Sử dụng pandas để load dataset từ máy tính hoặc theo đường dẫn. 

# In[10]:


# load datasets
# đọc dữ liệu
data5 = pd.read_csv("water_usage.csv")
data5


# In[11]:


y = np.array(data5)


# In[12]:


# Create figure
plt.figure(figsize=(8, 8), dpi=300)
# create plot.pie
mylabels = ["Leak", "Clothes Washer", "Faucet", "Shower	","Toilet","Other"]
explode = (0.5,0,0,0,0,0)
data5.groupby(mylabels).sum().plot(kind='pie', y='Percentage', autopct='%1.0f%%',explode=explode)
plt.title('Pie chart the use of water')
# Show plot
plt.show()


# 

# ## <span style='color:blue'> Bài tập 1.3. Bar plots:
# 
# Trong bài tập này, chúng ta sẽ sử dụng bar plot để so sánh điểm đánh giá một số bộ phim. Giả sử bạn có dữ liệu là điểm đánh giá cho 5 bộ phim, từ Rotten Tomatoes. The <span style='color:forestgreen'> **Tomatometer**</span> là tỷ lệ những nhà đánh giá Tomatometer mà đã có đánh giá tích cực (positive review) dành cho bộ phim. The <span style='color:forestgreen'> **Audience Score**</span> là tỷ lệ người xem mà đã cho điểm 3.5 hoặc cao hơn cả 5. Hãy vẽ biểu đồ cột để so sánh hai mức điểm đánh giá đó giữa 5 bộ phim. 

# In[13]:


# Import statements
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# Sử dụng pandas để load dataset từ máy tính hoặc đường dẫn. 

# In[14]:


# Load dataset
# đọc dữ liệu
data6 = pd.read_csv("movie_scores.csv",index_col = 0)
data6


# In[15]:


data6.head()


# read.csv,

# In[16]:


df = pd.DataFrame(data6)


# In[17]:


value1 = df["Tomatometer"]
value1


# In[18]:


value2 = df["AudienceScore"]
value2


# In[19]:


name = df["MovieTitle"]
name


# Use Matplotlib to create a visually-appealing bar plot comparing the two scores for all five movies. Use the movie titles as labels for the x-axis. Use percentages in an interval of 20 for the y-axis and minor ticks in interval of 5. Add a legend and a suitable title to the plot.

# In[20]:


list[value1,value2]


# In[21]:


data6.drop(['MovieTitle'],axis = 1)


# In[22]:


# Create figure
fig = plt.subplots(figsize=(12, 8))
# Create bar plot Df[“name”]
x_axis = np.arange(5)
team = data6['MovieTitle'].unique()
plt.bar(x = x_axis-0.2, height = data6['Tomatometer'],width = 0.4, label = "Tomatometer")
plt.bar(x = x_axis+0.2, height = data6['AudienceScore'],width = 0.4,label = "AudienceScore")
# # Specify ticks----
plt.xticks(x_axis,team)
#plt.yticks(y_)
# # Get current Axes for setting tick labels and horizontal grid
ax = plt.gca()
ax.set_xlabel('MovieTitle')
ax.set_ylabel('Percentage')
ax.yaxis.grid(True)
 # Add minor ticks for y-axis in the interval of 5
ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
# Add major horizontal grid with solid lines
plt.grid(which='major',axis='y',linestyle='solid')
# # Add minor horizontal grid with dashed lines
plt.grid(which='minor',axis='y',linestyle='dashed')
# # Add title
plt.xlabel("the move titles")
plt.ylabel("Percentage")
plt.title("Comparision the two scores for all five movies")
# # Add legend
plt.legend()
# Show plot
plt.show()


# ## <span style='color:blue'> Bài tập 1.4. Stacked Bar plots:
#     
# Trong bài tập này, ta sẽ vẽ <span style='color:red'> ***stacked bar plot***</span> (biểu đồ cột chồng) để thể hiện tình hình hoạt động của một nhà hàng. Bài toán: giả sử bạn là chủ một nhà hàng và bởi vì một điều luật mới bắt buộc bạn phải lập ra một ngày free-smoking. Nhằm giảm thiểu tối đa việc sụt giảm doanh thu, hãy vẽ biểu đồ thể hiện mức doanh thu mỗi ngày theo khách hàng có hút thuốc và không hút thuốc.

# In[23]:


# Import statements
import seaborn as sns


# In[24]:


#Load dataset
bills = sns.load_dataset('tips')
bills


# Use the given dataset and create a matrix where the elements contain the sum of the total bills for each day and smoking/non-smoking person.

# In[25]:


bills.info()


# In[26]:


a = bills.loc[(bills['day']=='Thur') & (bills['smoker']=='No'),"total_bill"]
a
sum(a)
round(sum(a),1)


# In[27]:


bills['day'].unique()


# In[28]:


zeros_array = np.zeros((2,4))
print(zeros_array)


# In[29]:


b = list()
for d in bills['day'].unique():
    c = list()
    for f in bills['smoker'].unique():
        a = bills.loc[(bills['day']==d) & (bills['smoker']==f),"total_bill"]
        c.append(sum(a))
    b.append(c)
b


# In[55]:


d = bills['day'].unique()
d


# In[62]:


y1 = b[1]
y1


# Create a stacked bar plot, stacking the summed smoking and non-smoking total bills separated for each day. Add a legend, labels, and a title.

# In[67]:


#create figure
fig = plt.figure(figsize =(10, 7))
bills_grouped = bills.groupby(['day', 'smoker'])['total_bill'].sum().unstack()
bills_grouped.plot(kind='bar', stacked=True)
plt.xlabel('Day of the Week')
plt.ylabel('Total Bill')
plt.title('Total Bills by Day and Smoking Status')
plt.show()


# 

# ## <span style='color:blue'> Bài tập 1.5. Histograms và Box plots:
#     
# Trong bài tập này, chúng ta sẽ trực quan dữ liệu về chỉ số the intelligent quotient (IQ) bằng histogram và box plots.

# In[68]:


# Import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[69]:


# IQ samples
iq_scores = [126,  89,  90, 101, 102,  74,  93, 101,  66, 120, 108,  97,  98,
            105, 119,  92, 113,  81, 104, 108,  83, 102, 105, 111, 102, 107,
            103,  89,  89, 110,  71, 110, 120,  85, 111,  83, 122, 120, 102,
            84, 118, 100, 100, 114,  81, 109,  69,  97,  95, 106, 116, 109,
            114,  98,  90,  92,  98,  91,  81,  85,  86, 102,  93, 112,  76,
            89, 110,  75, 100,  90,  96,  94, 107, 108,  95,  96,  96, 114,
            93,  95, 117, 141, 115,  95,  86, 100, 121, 103,  66,  99,  96,
            111, 110, 105, 110,  91, 112, 102, 112,  75]


# Hãy vẽ histogram với bins = 10, cho dữ liệu iq_scores đã cho. IQ scores thường được phân bố theo phân phối chuẩn (normal distribution) với trung bình bằng 100 và độ lệch chuẩn bằng 15. Hãy vẽ giá trị trung bình này bằng một solid red line, và vẽ độ lệch chuẩn bằng dashed vertical lines. Thêm vào labels và title.

# In[70]:


# Create figure
plt.figure(figsize=(6, 4), dpi=150)
# Create histogram
fig, ax = plt.subplots()
ax.hist(iq_scores,bins=10)
mean = np.mean(iq_scores)
stddev = np.std(iq_scores)
plt.axvline(mean, color='red', linewidth=2)
plt.axvline(mean - stddev, linestyle='--', color='red')
plt.axvline(mean + stddev, linestyle='--', color='red')
# Add labels and title
plt.xlabel('IQ scores')
plt.ylabel('Frequency')
plt.title('Distribution of IQ scores with mean and standard deviation')
# Show plot
plt.show()


# In[71]:


# Create figure
plt.figure(figsize=(6, 4), dpi=150)
# Create boxplot
sns.boxplot(data = iq_scores)
# Add labels and title
plt.xlabel('IQ scores')
plt.ylabel('Frequency')
plt.title('IQ scores for a test group of a hundred adults')
# Show plot
plt.show()


# ### Bây giờ, ta có dữ liệu theo 4 groups như sau:

# In[72]:


group_a = [118, 103, 125, 107, 111,  96, 104,  97,  96, 114,  96,  75, 114,
       107,  87, 117, 117, 114, 117, 112, 107, 133,  94,  91, 118, 110,
       117,  86, 143,  83, 106,  86,  98, 126, 109,  91, 112, 120, 108,
       111, 107,  98,  89, 113, 117,  81, 113, 112,  84, 115,  96,  93,
       128, 115, 138, 121,  87, 112, 110,  79, 100,  84, 115,  93, 108,
       130, 107, 106, 106, 101, 117,  93,  94, 103, 112,  98, 103,  70,
       139,  94, 110, 105, 122,  94,  94, 105, 129, 110, 112,  97, 109,
       121, 106, 118, 131,  88, 122, 125,  93,  78]
group_b = [126,  89,  90, 101, 102,  74,  93, 101,  66, 120, 108,  97,  98,
            105, 119,  92, 113,  81, 104, 108,  83, 102, 105, 111, 102, 107,
            103,  89,  89, 110,  71, 110, 120,  85, 111,  83, 122, 120, 102,
            84, 118, 100, 100, 114,  81, 109,  69,  97,  95, 106, 116, 109,
            114,  98,  90,  92,  98,  91,  81,  85,  86, 102,  93, 112,  76,
            89, 110,  75, 100,  90,  96,  94, 107, 108,  95,  96,  96, 114,
            93,  95, 117, 141, 115,  95,  86, 100, 121, 103,  66,  99,  96,
            111, 110, 105, 110,  91, 112, 102, 112,  75]
group_c = [108,  89, 114, 116, 126, 104, 113,  96,  69, 121, 109, 102, 107,
       122, 104, 107, 108, 137, 107, 116,  98, 132, 108, 114,  82,  93,
        89,  90,  86,  91,  99,  98,  83,  93, 114,  96,  95, 113, 103,
        81, 107,  85, 116,  85, 107, 125, 126, 123, 122, 124, 115, 114,
        93,  93, 114, 107, 107,  84, 131,  91, 108, 127, 112, 106, 115,
        82,  90, 117, 108, 115, 113, 108, 104, 103,  90, 110, 114,  92,
       101,  72, 109,  94, 122,  90, 102,  86, 119, 103, 110,  96,  90,
       110,  96,  69,  85, 102,  69,  96, 101,  90]
group_d = [ 93,  99,  91, 110,  80, 113, 111, 115,  98,  74,  96,  80,  83,
       102,  60,  91,  82,  90,  97, 101,  89,  89, 117,  91, 104, 104,
       102, 128, 106, 111,  79,  92,  97, 101, 106, 110,  93,  93, 106,
       108,  85,  83, 108,  94,  79,  87, 113, 112, 111, 111,  79, 116,
       104,  84, 116, 111, 103, 103, 112,  68,  54,  80,  86, 119,  81,
        84,  91,  96, 116, 125,  99,  58, 102,  77,  98, 100,  90, 106,
       109, 114, 102, 102, 112, 103,  98,  96,  85,  97, 110, 131,  92,
        79, 115, 122,  95, 105,  74,  85,  85,  95]


# Vẽ một box plot cho the IQ scores của mỗi test groups. Thêm labels và title.

# fig, axs = plt.subplots(2, 2, figsize=(8, 8))
# axs[0, 0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
# axs[0, 1].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
# axs[1, 0].imshow(cv2.cvtColor(image3, cv2.COLOR_BGR2RGB))
# axs[1, 1].imshow(cv2.cvtColor(image4, cv2.COLOR_BGR2RGB))
# 

# In[73]:


### group A --- chỉnh lại box plot
# Create figure
plt.figure(figsize=(10, 6), dpi=150)
# Create boxplots
image1 = sns.boxplot(data = group_a)
# Add labels and title
plt.xlabel('IQ scores')
plt.ylabel('Frequency')
plt.title('IQ scores for different test groups')
# Show plot
plt.show()


# In[74]:


### group B
# Create figure
plt.figure(figsize=(6, 4), dpi=150)
# Create boxplots
image2 = sns.boxplot(data = group_b)
# Add labels and title
plt.xlabel('IQ scores')
plt.ylabel('Frequency')
plt.title('IQ scores for different test groups')
# Show plot
plt.show()


# In[75]:


### group C
# Create figure
plt.figure(figsize=(6, 4), dpi=150)
# Create boxplots
image3 = sns.boxplot(data = group_c)
# Add labels and title
plt.xlabel('IQ scores')
plt.ylabel('Frequency')
plt.title('IQ scores for different test groups')
# Show plot
plt.show()


# In[95]:


# Tạo figure và các subplot
fig, axs = plt.subplots(1, 3, figsize=(20, 6))

# Vẽ biểu đồ đường lên subplot đầu tiên
#image1 = sns.boxplot(data = group_a)
sns.boxplot(data = group_a,ax=axs[0])
axs[0].set_title('Group_a')

# Vẽ biểu đồ thanh lên subplot thứ hai
sns.boxplot(data = group_b,ax=axs[1])
axs[1].set_title('Group_b')

# Vẽ biểu đồ scatter lên subplot thứ ba
sns.boxplot(data = group_c,ax=axs[2])
axs[2].set_title('Group_c')

fig.supxlabel('IQ scores')
fig.supylabel('Frequency')
fig.suptitle('IQ scores for different test groups')
plt.show()


# ## <span style='color:blue'> Bài tập 1.6. Scatter plots:
#     
# Trong bài tập này, ta sẽ sử dụng a scatter plot để thể hiện mối tương quan (correlation) giữa các feature trong một dataset. Giả sử xét một dataset chứa một số thông tin về các loài động vật khác nhau. Hãy vẽ biểu đồ thể hiện mối tương quan giữa các loài vật này.

# In[67]:


# Import statements
import pandas as pd


# In[68]:


# Load dataset
data7 = pd.read_csv("anage_data.csv",index_col = 0)
data7


# Dataset được cho vẫn chưa hoàn thiện. Hãy lọc dữ liệu sao cho bạn sẽ có mẫu chứa thông tin về body mass và maximum longevity. 
# 
# Sắp dữ liệu theo animal class.

# In[69]:


data7.info()


# In[70]:


missing_data = pd.DataFrame({'total_missing': data7.isnull().sum(), 'perc_missing': (data7.isnull().sum()/82790)*100})
missing_data


# In[71]:


data7.describe()


# In[72]:


#Dataset được cho vẫn chưa hoàn thiện. Hãy lọc dữ liệu sao cho bạn sẽ có mẫu chứa thông tin về body mass và maximum longevity. 
#Sắp dữ liệu theo animal class.
# Preprocessing
data7["Maximum longevity (yrs)"].fillna(4, inplace = True)
data7["Body mass (g)"].fillna(5, inplace = True)
bm_mxmlongevity = data7[["Body mass (g)","Maximum longevity (yrs)"]]
bm_mxmlongevity


# In[73]:


# Sort according to class
data7_sorted = data7.sort_values(by='Class', ascending=True)
data7_sorted 


# Vẽ một scatter plot thể hiện mối tương quan giữa body mass và the maximum longevity. Sử dụng màu khác nhau cho grouping data samples tương ứng với class. Thêm vào a legend, labels và title. Sử dụng log scale cho cả x-axis và y-axis.

# In[74]:


# Create figure
plt.figure(figsize=(10, 6), dpi=300)
# Create scatter plot
#sns.scatterplot(x = data7['Body mass (g)'], y = data7['Maximum longevity (yrs)']);
ax1 = data7.plot.scatter(x='Body mass (g)',y='Maximum longevity (yrs)',c='Blue',s= 80)
# Add legend
label1 = ['bm&ml']
plt.legend(label1)
# Log scale
plt.xlim(6,30)
plt.ylim(0,30)
# Add labels
plt.xlabel('Body mass')
plt.ylabel('Maximum longevity(yrs)')
plt.title('Correlation between two scores')
# Show plot
plt.show()


# In[76]:


import matplotlib.pyplot as plt


# In[78]:


df3 = pd.DataFrame(data7)


# In[88]:


# Create figure
plt.figure(figsize=(10, 6), dpi=300)
# Create scatter plot
#ax2 = data7.plot.scatter(x='Body mass (g)',y='Maximum longevity (yrs)',hue ='class',s= 80)
sns.scatterplot(x='Body mass (g)',y='Maximum longevity (yrs)',hue ='Class', data = df3)
# Add legend
plt.legend()
plt.xlim(0,100)
plt.ylim(0,150)
# Add labels
plt.xlabel('Body mass')
plt.ylabel('Maximum longevity(yrs)')
plt.title('Correlation between two scores')
# Show plot
plt.show()

