#!/usr/bin/env python
# coding: utf-8

# # Load CSV

# In[2]:


import pandas as pd
df = pd.read_csv("../n1_data_and_annotations.csv")
df.head(4)


# # MIN-MAX SCALER

# In[4]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
new_df = df.iloc[:,:-2]

new_df.head(4)
print(scaler.fit(new_df))

print(scaler.data_max_)
df_norm = scaler.transform(new_df)
df_norm = pd.DataFrame(df_norm)
df_norm['times'] = df['times']
df_norm['sleepstage'] = df['sleepstage']
df_norm.columns = df.columns
df_norm.head(4)


# In[5]:


df.to_numpy


# In[9]:


import matplotlib.pyplot as plt
plt.figure(figsize=(30,5))
#plt.plot(norm["times"],norm["ROC-LOC"]*2)
#plt.plot(norm["times"],norm["LOC-ROC"]*2)
#plt.plot(norm["times"],norm["F2-F4"]*2)
#plt.plot(norm["times"],norm["F4-C4"]*2)
#plt.plot(norm["times"],norm["C4-P4"]*2)
#plt.plot(norm["times"],norm["P4-O2"]*3)
#plt.plot(norm["times"],norm["F1-F3"]*3)
#plt.plot(norm["times"],norm["F3-C3"]*3)
#plt.plot(norm["times"],norm["C3-P3"]*3)
#plt.plot(norm["times"],norm["P3-O1"]*3)
#plt.plot(norm["times"],norm["C4-A1"])
#plt.plot(norm["times"],norm["EMG1-EMG2"]*3)
#plt.plot(norm["times"],norm["ECG1-ECG2"])
#plt.plot(norm["times"],norm["TERMISTORE"])
#plt.plot(norm["times"],norm["TORACE"])
#plt.plot(norm["times"],norm["ADDOME"])
#plt.plot(norm["times"],norm["Dx1-DX2"]*2)
#plt.plot(norm["times"],norm["SX1-SX2"])
#plt.plot(norm["times"],norm["Posizione"])
#plt.plot(norm["times"],norm["HR"]*2)
#plt.plot(norm["times"],norm["SpO2"]*3)
plt.plot(df_norm["times"],(df_norm["sleepstage"]))


# In[11]:


selected_data=df_norm[["times","F2-F4","F4-C4","C4-P4","P4-O2","F1-F3","F3-C3","C3-P3","P3-O1"]]


# # ICA

# In[14]:


from sklearn.decomposition import FastICA

#X, A = w.result
ica_model = FastICA(n_components=9, random_state=0)
X_transformed = ica_model.fit_transform(selected_data)


# In[15]:


f, ax = plt.subplots(9, 2, figsize=(25, 5))
_ = ax[0,0].plot(selected_data[["times"]], selected_data[["F2-F4"]], 'o')
_ = ax[0,1].plot(X_transformed[:, 0], X_transformed[:, 1], 'o')
_ = ax[0,0].set_title("Captured Distribution")
_ = ax[0,1].set_title("ICA components")

_ = ax[1,0].plot(selected_data[["times"]], selected_data[["F4-C4"]], 'o')
_ = ax[1,1].plot(X_transformed[:, 0], X_transformed[:, 2], 'o')
_ = ax[1,0].set_title("Captured Distribution")
_ = ax[1,1].set_title("ICA components")

_ = ax[2,0].plot(selected_data[["times"]], selected_data[["C4-P4"]], 'o')
_ = ax[2,1].plot(X_transformed[:, 0], X_transformed[:, 3], 'o')
_ = ax[2,0].set_title("Captured Distribution")
_ = ax[2,1].set_title("ICA components")

