#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests

url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
r = requests.get(url)

import os
os.mkdir('data')
    
import tarfile
import urllib

open('data/housing.tgz','wb+').write(r.content)

with tarfile.open('data/housing.tgz','r:gz') as tar:
    tar.extractall('data')

os.remove('data/housing.tgz')

    



# In[ ]:





# In[2]:


# import os

# parent_dir = "."

# path2 = os.path('.')
# print(path2)

# # directory = "data"
# # path = os.path.join(parent_dir, directory) 
# # os.mkdir(path)


import pandas as pd


df = pd.read_csv('data/housing.csv')
df.head()



# In[3]:


df.info()


# In[4]:


import matplotlib.pyplot as plt
import pandas as pd
df.hist(bins=50, figsize=(20,15))
plt.savefig('obraz1.png')



# In[5]:


import matplotlib.pyplot as plt
df.plot(kind="scatter", x = "longitude", y = "latitude",
        alpha=0.1, figsize=(7,4))

plt.savefig('obraz2.png')


# In[6]:


import matplotlib.pyplot as plt


df.plot(kind="scatter", x = "longitude", y ="latitude",
        alpha = 0.4, figsize = (7,3), colorbar=True,
        s=df["population"]/100, label = "population",
        c = "median_house_value", cmap = plt.get_cmap("jet"))

plt.savefig('obraz3.png')


# In[7]:


s = df.corr(numeric_only=True)["median_house_value"].sort_values(ascending=False).reset_index()
s = s.rename(columns = {'index': 'atrybut', 'median_house_value' : 'wspolczynnik_korelacji'} )
s.to_csv('korelacja.csv', index=False)

# df.rename(columns = {'index': 'atrybut' , 'median_house_value' : 'wspolczynnik korelacji'} )


# In[8]:


import seaborn as sns
sns.pairplot(df)


# In[9]:


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df,
                                       test_size = 0.2,
                                       random_state=42)

len(train_set),len(test_set)


# In[10]:


train_set.head()




# In[11]:


test_set.head()


# In[12]:


train_set.corr(numeric_only=True)


# In[13]:


test_set.corr(numeric_only=True)


# In[14]:


train_set.to_pickle('train_set.pkl')


# In[15]:


test_set.to_pickle('test_set.pkl')


# In[ ]:




