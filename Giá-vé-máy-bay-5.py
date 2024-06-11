#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("archive/Clean_Dataset.csv")


# In[3]:


df.head()


# In[4]:


df = df.drop('Unnamed: 0', axis=1) # Bỏ cột dư thừa


# # Context
# - Mục tiêu của nghiên cứu là phân tích tập dữ liệu đặt vé máy bay thu được từ trang web “Ease My Trip” và tiến hành các thử nghiệm giả thuyết thống kê khác nhau để thu được thông tin có ý nghĩa từ đó. 'Easemytrip' là một nền tảng internet để đặt vé máy bay và do đó là một nền tảng mà các hành khách tiềm năng sử dụng để mua vé. Một nghiên cứu kỹ lưỡng về dữ liệu sẽ hỗ trợ khám phá những hiểu biết có giá trị sẽ có giá trị to lớn đối với hành khách.

# ## FEATURES:
# - Hãng hàng không: Tên hãng hàng không được lưu trong cột hãng hàng không. Đây là một tính năng phân loại có 6 hãng hàng không khác nhau.
# - Chuyến bay: Chuyến bay lưu trữ thông tin liên quan đến mã chuyến bay của máy bay. Nó là một tính năng phân loại.
# - Thành phố nguồn: Thành phố nơi chuyến bay cất cánh. Đây là một tính năng phân loại có 6 thành phố độc đáo.
# - Thời gian khởi hành: Đây là một tính năng phân loại có nguồn gốc được tạo bằng cách nhóm các khoảng thời gian thành các thùng. Nó lưu trữ thông tin về thời gian khởi hành và có 6 nhãn thời gian duy nhất.
# - Điểm dừng: Một tính năng phân loại với 3 giá trị riêng biệt lưu trữ số điểm dừng giữa các thành phố nguồn và đích.
# - Thời gian đến: Đây là một tính năng phân loại có nguồn gốc được tạo bằng cách nhóm các khoảng thời gian thành các ngăn. Nó có sáu nhãn thời gian riêng biệt và lưu giữ thông tin về thời gian đến.
# - Destination City: Thành phố nơi chuyến bay sẽ hạ cánh. Đây là một tính năng phân loại có 6 thành phố độc đáo.
# - Hạng ghế: Một tính năng phân loại chứa thông tin về hạng ghế; nó có hai giá trị riêng biệt: Kinh doanh và Kinh tế.
# - Thời lượng: Một tính năng liên tục hiển thị tổng thời gian cần thiết để di chuyển giữa các thành phố tính bằng giờ.
# - Số ngày còn lại: Đây là một đặc tính bắt nguồn được tính bằng cách lấy ngày đặt trước trừ đi ngày của chuyến đi.
# - Giá: Biến mục tiêu lưu thông tin về giá vé.

# # Dataset
# - Bộ dữ liệu chứa thông tin về các tùy chọn đặt vé máy bay từ trang web Easemytrip cho hành trình bay giữa 6 thành phố lớn hàng đầu của Ấn Độ. Có 300261 điểm dữ liệu và 11 thuộc năng trong tập dữ liệu

# In[5]:


df.head()


# In[6]:


# Print the type of each columns
df.info()


# * Sau khi kiểm tra tổng quan bộ data, ta quan sát thấy có 300153 data và không có giá trị khuyết 

# <h2>Phân Loại các đặc trưng có trong data</h2>

# In[7]:


# count the values of each airline
df1=df.groupby(['flight','airline'],as_index=False).count()
df1.airline.value_counts()


# - Nhìn chung ta có thể thấy được hảng hàng không Indigo là hãng hàng không phổ biến nhất trong 6 hãng và hãng hàng không ít phổ biến nhất là AirAsia.

# In[8]:


for column in df.columns:
    if column != 'duration' and column != 'price' and column != 'days_left' and column != 'flight':
        print(f"{column}: \n{df[column].unique()}\n\n")
# In ra các loại giá trị trong mỗi cột


# # Visualization

# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="white", rc=custom_params)


# <h6>Tìm hiểu các đặc trưng giữa giá vé và các yếu tố khác trong data</h6>

# **1.  Có sự chênh lệch giá giữa các hãng hàng không?**

# In[10]:


# vẽ đồ thị so sánh giá vé giữa các hãng hàng không
plt.figure(figsize = (10,8))
ax = sns.barplot(data=df, x='airline', y='price', hue='class',palette='hls')
for i in ax.containers : 
    ax.bar_label(i,padding= 6,
                 label_type = 'edge',
                 fontsize = 11)
plt.ylabel('Giá vé',fontsize=12)
plt.xticks(rotation=30, ha = 'right')
plt.xlabel('Hãng bay',fontsize=12)
plt.ylim(0,70000)
plt.title('SỰ CHÊNH LỆCH GIÁ VÉ GIỮA CÁC HÃNG HÀNG KHÔNG',fontsize=12)


# + Giá có thay đổi tùy theo từng Hãng hàng không.
# + Rõ ràng, ở Hạng phổ thông, Vistara được sử dụng nhiều nhất và Air Asia là được sử dụng ít nhất trong khi đó, Hạng thương gia
# + Chỉ có 2 hãng : Vistara và Air_India có hạng thương gia (Business) 

# **2. Số lượng chuyến bay của mỗi hãng**

# In[11]:


plt.figure(figsize=(10,8))
ax = sns.countplot(x = df['airline'],palette='hls', hue = df["class"])
plt.title('Thống kê số lượng chuyển bay của các hãng hàng không',fontsize=17)
for i in ax.containers : 
    ax.bar_label(i,padding= 6,
                 label_type = 'edge',
                 fontsize = 11)
plt.ylim(0,80000)
plt.xlabel('Hãng hàng không',fontsize=12, labelpad= 20)
plt.ylabel('Số lượng',fontsize=12, rotation = 0, labelpad = 50, loc = 'center')
plt.xticks(rotation=25, 
            horizontalalignment='right',
            fontweight='light',
            fontsize=10)  
plt.show()


# **3.  Sự phụ thuộc giá vé vào số ngày mua trước khi bay**

# In[12]:


df_temp = df.groupby(['days_left'])['price'].mean().reset_index()
plt.figure(figsize=(15,6)).suptitle('SỰ PHỤ THUỘC GIÁ VÉ VÀO NGÀY MUA TRƯỚC KHI BAY', fontsize=17)
ax = plt.axes()
sns.regplot(x=df_temp.loc[df_temp["days_left"]==1].days_left, y=df_temp.loc[df_temp["days_left"]==1].price, fit_reg=False, ax=ax)
sns.regplot(x=df_temp.loc[(df_temp["days_left"]>1)&(df_temp["days_left"]<20)].days_left, y=df_temp.loc[(df_temp["days_left"]>1)&(df_temp["days_left"]<20)].price, fit_reg=True, ax=ax)
sns.regplot(x=df_temp.loc[df_temp["days_left"]>=20].days_left, y=df_temp.loc[df_temp["days_left"]>=20].price, fit_reg=True, ax=ax)


# + Dựa trên phân tích dữ liệu, rõ ràng là giá vé máy bay tăng đáng kể khi được mua trong tầm 2 tuần trước ngày khởi hành theo lịch trình, trong đó vé hạng Thương gia có thể tăng giá cao hơn so với vé hạng Phổ thông.
# 
# + Để đảm bảo có được vé tiết kiệm chi phí nhất, nên mua vé hạng Phổ thông ít nhất khoảng 20 ngày (2 tuần ) trước ngày khởi hành, trong khi đối với vé hạng Thương gia, nên mua vé ít nhất mười ngày trước ngày khởi hành theo lịch trình.

# **4. Giá vé có thay đổi hay không khi dựa vào thời gian khởi hành và thời gian đến**

# In[13]:


df_gh = df.loc[df['airline'] == 'Vistara']
df_gh


# In[14]:


depature_time_price = df.groupby('departure_time')['price'].mean().round(0).sort_values(ascending = True)
arrival_time_price = df.groupby('arrival_time')['price'].mean().round(0).sort_values(ascending = True)

plt.figure(figsize =(16,13))
plt.subplot(1,2,1)
plt.bar(depature_time_price.index, depature_time_price.values,color=['paleturquoise', 'paleturquoise','paleturquoise', 'paleturquoise', 'dodgerblue'])
plt.title("Prices/Depature Times Histogram")
plt.xlabel("Depature Times", labelpad=10)
plt.ylabel("Average Prices")
plt.xticks(rotation = 35)

plt.subplot(1,2,2)
plt.bar(arrival_time_price.index, arrival_time_price.values,color=['paleturquoise', 'paleturquoise','paleturquoise', 'paleturquoise', 'paleturquoise', 'dodgerblue'])
plt.title("Prices/Arrival Times Histogram")
plt.xlabel("Arrival Times", labelpad=3)
plt.ylabel("Average Prices")
plt.xticks(rotation = 35)

plt.show()


# **5. Giá thay đổi như thế nào với sự thay đổi giữa các điểm đi và đến khác nhau?**

# In[13]:


#removing outliers
df1=df.drop(["flight","departure_time","arrival_time","stops"],axis=1)
df2=df1.copy()
#feature engineering destination and source in on column
df2["source_and_destination"] = df2[["source_city", "destination_city"]].apply(lambda x: "-".join(x), axis=1)
df2.drop(["source_city","destination_city"],axis=1,inplace=True)
#source and destination is made it into a single column


# In[14]:


#seperating dataframes as business and economy for better analysis
df_economy=df2[df2["class"]=="Economy"]
df_business=df2[df2["class"]=="Business"]
dftemp=df1.reset_index() #temporary dataframe used for counts
df_economy.head()
#df_business.head()


# In[15]:


fig = plt.figure(figsize=(8,10))
sns.displot(data = df_economy,x='price',y='source_and_destination', height = 8)


# In[16]:


fig = plt.figure(figsize=(8,10))
sns.displot(data=df_business,x='price',y='source_and_destination', height = 8)


# ### Nhận xét
# - Đối với hạng phổ thông, ta thấy các chuyến đi bắt đầu tại Delhi và Mumbai nhiều hơn so với các địa điểm khác
# - Đối với hạng thương gia, các chuyến đi thưa thớt hơn và rải rác khắp các địa điểm
# - Giá vé không phụ thuộc vào điểm đi và điểm đén. Tuy nhiên, quan sát đồ thị, ta thấy có 1 vài outlier. Dù vậy, ta vẫn có thể thấy được xu hướng chung của giá vé khi xét đến yếu tố Địa điểm khởi hành 

# **6.So sánh giá vé của hạng Thương gia và Phổ Thông**

# In[17]:


# Chốt
# draw the boxplot to compare between the different class with ticket price
plt.figure(figsize=(10,5))
sns.boxplot(x='class',y='price',data=df,palette='hls', width = 0.5, whis = 3, linewidth=0.8)
plt.title('Class Vs Ticket Price',fontsize=15)
plt.xlabel('Class',fontsize=10)
plt.ylabel('Price',fontsize=10)
plt.yticks()
plt.show()


# ## Nhận xét:
# - Trung bình giá vé của hạng economy vẫn rẻ hơn nhiều lần so với hạng thương gia. Tuy nhiên, vẫn có vẽ ở hạng phổ thông có giá đắt bằng hoặc hơn thương gia. 
# - Tương tự, ở hạng Thương gia, có những vé bằng hay rẻ hơn hạng Phổ Thông

# **7.Ảnh hưởng của số điểm dừng tác động lên giá vé**

# In[18]:


# check the values of "stops"
df["stops"].unique()


# In[19]:


fig = plt.figure(figsize  = (8,10))
explode = (0,0,0.1)
df.stops.value_counts().plot(kind='pie',autopct="%1.1f%%",explode = explode, fontsize = 14)
plt.ylabel('COUNT', fontsize = 15)
plt.title("ẢNH HƯỞNG CỦA SỐ ĐIỂM DỪNG LÊN GIÁ VÉ", fontsize = 18)
plt.legend(['one','zero','two and more '], loc = 'upper right', fontsize = 13)
plt.tight_layout()


# ### Nhận xét:
# - Từ biểu đồ này có thể thấy được với một điểm dừng trong suốt chặng bay thường có giá mắc nhất trong tất cả các loại hình. Tuy nhiên với các loại hình bay không có điểm dừng nào lại có giá thấp nhất
# - Hơn thế nữa từ biểu đồ này ta có thể thấy hàng bay Vistara là hãng bay có giá thành vé máy bay cao nhất so với các hãng máy bay khác. Thêm vào đó là hãng bay Air_India là hãng bay có giá thành tương dối bình ổn so với mặt bằng trung.
# - Hầu hết đa phần các hành khách chọn các hãng bay nhu Vistara và Air India thay vì các hãng bay còn lại mặc dù nhìn trong biểu đồ ta có thể thấy hãng bay Air Asia cũng là một hãng bay có mức giá tương đối thấp nhưng lại không được ưa chuộng bằng.

# **8. Giá thay đổi như thế nào với sự thay đổi giữa các điểm đi và đến khác nhau?**

# In[20]:


# Import Dataset
X = df.loc[:,"duration" ].values
y = df.loc[:,"price"].values


# In[21]:


X = X.reshape(-1,1)


# In[22]:


# Separating the dataset as Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8, random_state = 22)


# In[23]:


# Linear Regression


# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Result prediction on Test set
y_pred = regressor.predict(X_test)



# In[24]:


# Đánh giá mô hình
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[25]:


# Result visualization on Training Set
plt.scatter(X, y, color = 'blue', s = 0.7, alpha= 0.3)
plt.plot(X_train, regressor.predict(X_train), color = 'green')
plt.title('Days Left vs Price (Training set)')
plt.xlabel('Duration') 
plt.ylabel('Price')
plt.show()


# - Qua mô hình ta thấy giá vé có xu hướng tăng khi thời gian bay tăng và tác động của yếu tố "thời gian bay" khá rõ ràng

# ## Model Machine Learning

# In[26]:


# Use the Label Encoder to transform all columns which is object to int
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in df.columns:
    if df[col].dtype=='object':
        df[col]=le.fit_transform(df[col])


# In[27]:


def prepare_X_y(df):
    """
    Feature engineering and create X and y 
    :param df: pandas dataframe
    :return: (X, y) output feature matrix (dataframe), target (series)
    """
    # Todo: Split data into X and y (using sklearn train_test_split). Return two dataframes
    feature_names = df.columns.tolist()
    feature_names.remove("price")
    X = df[feature_names].values
    y = df.price.values
    return X, y


# In[28]:


# print the function above
X,y = prepare_X_y(df)


# In[29]:


# spli data 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler 
RANDOM_STATE = 42
TRAIN_SIZE = 0.7

X_train,X_test ,Y_train, Y_test = train_test_split(X, y, train_size=TRAIN_SIZE, random_state=RANDOM_STATE)


# In[30]:


from sklearn.preprocessing import MinMaxScaler


# In[31]:


# norm data
mmscaler=MinMaxScaler()
X_train=mmscaler.fit_transform(X_train)
X_test=mmscaler.fit_transform(X_test)


# ## Using Desicion Tree model

# In[32]:


# apply decision tree
modeldcr = DecisionTreeRegressor()
pipe = Pipeline(steps=[("mmscaler", mmscaler), ("modeldcr", modeldcr)])
modeldcr.fit(X_train,Y_train)


# In[33]:


# calculate y_pred
y_pred = modeldcr.predict(X_test)


# In[34]:


from sklearn.metrics import mean_squared_error, r2_score

# Evaluate the mean squared error of the model
mse = mean_squared_error(Y_test, y_pred)
print(f"Mean squared error: {mse}")

# Evaluate the R2 score of the model
r2 = r2_score(Y_test, y_pred)
print(f"R2 score: {r2}")


# In[35]:


# checking the difference of ticket price from one source_city to destination_city 
df.groupby(['airline','source_city','destination_city'],as_index=False)['price'].mean().head(10)


# In[36]:


# copy new data into "df_bk"
df_bk=df.copy()


# In[37]:


# calculate the output
out=pd.DataFrame({'Price_actual':Y_test,'Price_pred':y_pred})
result=df_bk.merge(out,left_index=True,right_index=True)


# In[38]:


# plot the regplot
plt.figure(figsize=(10,5))
sns.regplot(x='Price_actual',y='Price_pred',data=result,color='pink')
plt.title('Actual Price  Vs  Predicted Price ',fontsize=20)
plt.xlabel('Actual Price',fontsize=15)
plt.ylabel('Predicted Price',fontsize=15)
plt.show()


# ## Nhận xét:
# - Nhìn chung từ mô hình này vẫn còn khá nhiều điểm outliers trên biểu đồ regplot cho nên mô hình decision tree vẫn chưa phải là mô hình chính xác để dự đoán mô hình này

# ## Using Random - forest model

# In[39]:


# apply random forest
from sklearn.ensemble import RandomForestRegressor
modelrfr = RandomForestRegressor()
pipe1 = Pipeline(steps=[("mmscaler", mmscaler), ("modelrfr", modelrfr)]) #Build a pipeline with a scaler and a model
modelrfr.fit(X_train,Y_train)


# In[40]:


#calculate y_pred
y_pred1 = modelrfr.predict(X_test)


# In[41]:


from sklearn.metrics import mean_squared_error, r2_score

# Evaluate the mean squared error of the model
mse = mean_squared_error(Y_test, y_pred1)
print(f"Mean squared error: {mse}")

# Evaluate the R2 score of the model
r2 = r2_score(Y_test, y_pred1)
print(f"R2 score: {r2}")


# In[42]:


# calculate "out"
out=pd.DataFrame({'Price_actual':Y_test,'Price_pred':y_pred1})
result=df_bk.merge(out,left_index=True,right_index=True)


# In[48]:


# plot the regplot
plt.figure(figsize=(10,5))
sns.regplot(x='Price_actual',y='Price_pred',data=result,color='blue')
plt.title('Actual Price  Vs  Predicted Price ',fontsize=20)
plt.xlabel('Actual Price',fontsize=15)
plt.ylabel('Predicted Price',fontsize=15)
plt.show()


# ## Nhận xét:
# - Mô hình này thì có ít điểm outliers hơn cho thấy random forest là mô hình tốt để đưa ra dự đoán cho bài toán này.
# - Ngoài ra từ biểu đồ này ta còn thấy được mối quan hệ tuyến tính dương giữa hai biến số tức là giá tăng theo một cách tuyến khi biến số thay đổi, túc giá tăng khi biến số thay đổi

# ## Using linear regression model

# In[43]:


# apply the linear regression
from sklearn.linear_model import LinearRegression
modelmlg = LinearRegression()
pipe1 = Pipeline(steps=[("mmscaler", mmscaler), ("modelmlg", modelmlg)]) #Build a pipeline with a scaler and a model
modelmlg.fit(X_train,Y_train)


# In[44]:


# calculate y_pred2
y_pred2 = modelmlg.predict(X_test)


# In[45]:


from sklearn.metrics import mean_squared_error, r2_score

# Evaluate the mean squared error of the model
mse = mean_squared_error(Y_test, y_pred2)
print(f"Mean squared error: {mse}")

# Evaluate the R2 score of the model
r2 = r2_score(Y_test, y_pred2)
print(f"R2 score: {r2}")


# In[46]:


# calculate "output" and display it by dataframe
out=pd.DataFrame({'Price_actual':Y_test,'Price_pred':y_pred2})
result=df_bk.merge(out,left_index=True,right_index=True)


# In[47]:


# plot regplot
plt.figure(figsize=(10,5))
sns.regplot(x='Price_actual',y='Price_pred',data=result,color='blue')
plt.title('Actual Price  Vs  Predicted Price ',fontsize=20)
plt.xlabel('Actual Price',fontsize=15)
plt.ylabel('Predicted Price',fontsize=15)
plt.show()


# ## Nhận xét:
# - Từ biểu đồ ta có thể thấy ở dạng bài này sử dụng mô hình linear regression là sẽ không phù hợp do là các điểm outliers nhiều và bị lệch ra khổi đường hồi quy do đó dẫn tới kết quả của mô hình dự đoán giá bị sai lệch.

# <h1>Tổng Kết</h1>

# Nếu bạn là người đang muốn tiết kiệm tiền mua vé máy bay, có một số yếu tố cần xem xét trước khi mua. Một trong những yếu tố quan trọng nhất là thời gian của các chuyến bay của bạn. Điều đáng chú ý là thời gian trong ngày và ngày trong tuần có thể có tác động đáng kể đến giá vé. Nhìn chung, các chuyến bay khởi hành và đến vào ban đêm thường rẻ hơn so với các chuyến bay khởi hành và đến vào giờ cao điểm.
# 
# Đối với các chuyến bay có thời gian hành trình với 0 điểm dừng trong 2-3 giờ, thời gian khởi hành và hạ cánh vào đêm muộn thường là một lựa chọn khả thi hơn nếu muốn săn vé giá rẻ. Các chuyến bay này thường khởi hành sau 9 hoặc 10 giờ tối và đến điểm đến vào đầu giờ sáng. Mặc dù có thể không thuận tiện nhất về mặt lịch trình, nhưng chúng có thể là cách tốt nhất để tiết kiệm tiền mua vé.
# 
# Một yếu tố khác cần xem xét khi đặt chuyến bay của bạn là số điểm dừng. Trong một số trường hợp, chuyến bay có nhiều điểm dừng hơn có thể rẻ hơn chuyến bay có ít điểm dừng hơn, tùy thuộc vào đường bay và hãng hàng không. Bạn nên thực hiện một số nghiên cứu để tìm ra hãng hàng không nào cung cấp các giao dịch tốt nhất cho tuyến đường bạn đã chọn và họ thường thực hiện bao nhiêu điểm dừng.
# 
# Khi nói đến việc tìm kiếm ưu đãi tốt nhất cho vé máy bay, điều quan trọng là phải linh hoạt và mở rộng quan sát của bản thân đối với các lựa chọn khác nhau. Bằng cách xem xét các yếu tố như thời gian, số điểm dừng và các tùy chọn hãng hàng không, bạn có thể tăng cơ hội tìm được ưu đãi lớn trên chuyến bay tiếp theo.
