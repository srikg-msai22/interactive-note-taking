#!/usr/bin/env python
# coding: utf-8

# In[185]:


import numpy as np
import statistics
from functools import reduce
import matplotlib.pyplot as plt

# In[1]:


with open('test.txt') as f:
    lines = f.readlines()

# In[65]:


len(lines)

# In[75]:


total_count_sections = []
for paper in range(len(lines)):
    words = lines[paper].split()
    for i in range(len(words)):
        if '\"section_names\"' in words[i]:
            #             print(words[i])
            #             print(i)
            section_names_begin = i
        elif '\"sections\"' in words[i]:
            #             print(words[i])
            #             print(i)
            section_names_end = i
    count = 0
    for i in words[section_names_begin + 1:section_names_end]:

        if len(i) > 1:
            if '\"' in i[0] or '\"' in i[1]:
                count += 1
                # print(i)
    total_count_sections.append(count)

# In[150]:


total_sec_lengths = []
for paper in range(len(lines)):
    w = lines[paper].split()

    for i in range(len(w)):
        if '\"sections\"' in w[i]:
            #             print(words[i])
            #             print(i)
            section_names_end = i

    sec = w[section_names_end + 1:]

    indices = []
    for i in range(len(sec)):
        if '[' in sec[i][0]:
            if len(sec[i]) > 2:
                if 'fig' not in sec[i + 1]:
                    #         print('word ',sec[i])
                    #         print('index ',i)
                    #         print('nextword ',sec[i+1])
                    indices.append(i)
    indices.append(len(sec))

    section_lengths = []
    for i in range(len(indices) - 1):
        size = indices[i + 1] - indices[i]
        section_lengths.append(size)
    total_sec_lengths.append(section_lengths)

# In[159]:


all_lengths = reduce(lambda x, y: x + y, total_sec_lengths)

# In[161]:


all_lengths = np.array(all_lengths)

# In[165]:


mean_length = np.mean(all_lengths)
sd_length = np.std(all_lengths)
print('mean ', mean_length, 'standard_deviation ', sd_length)

# In[176]:


median_length = np.median(all_lengths)
print(median_length)

# In[173]:


maximum_length = max(all_lengths)
minimum_length = min(all_lengths)
print('max ', maximum_length, 'min ', minimum_length)

# In[179]:


print(np.quantile(all_lengths, 0.25))
np.quantile(all_lengths, 0.75)
