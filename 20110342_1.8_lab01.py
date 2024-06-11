#!/usr/bin/env python
# coding: utf-8

# In[5]:


### khai báo hàm thư viện cần dùng
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import matplotlib.image as mpimg


# In[8]:


# Load images
#img_filenames = os.listdir('data')
#imgs = [mpimg.imread(os.path.join('data', img_filename)) for img_filename in img_filenames]
image1 = cv2.imread('photo-1.jpg')
image2 = cv2.imread("photo-2.jpg")
image3 = cv2.imread("photo-3.jpg")
image4 = cv2.imread("photo-4.jpg")
# Tạo figure và thêm các ảnh vào figure
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
axs[0, 0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
axs[0, 1].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
axs[1, 0].imshow(cv2.cvtColor(image3, cv2.COLOR_BGR2RGB))
axs[1, 1].imshow(cv2.cvtColor(image4, cv2.COLOR_BGR2RGB))

# Đặt tiêu đề cho mỗi ảnh
axs[0, 0].set_title('coast')
axs[0, 1].set_title('beach')
axs[1, 0].set_title('building')
axs[1, 1].set_title('city at night')

# Tắt các trục để giảm độ phức tạp của figure
for ax in axs.flat:
    ax.axis('off')

# Hiển thị figure
plt.show()

