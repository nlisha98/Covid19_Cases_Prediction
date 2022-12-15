#%%
#1. Import Packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, datetime,pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential,Input
from tensorflow.keras.layers import SimpleRNN, Dense, LSTM, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error

#%%
#2. Data Loading
TRAIN_CSV_PATH = os.path.join(os.getcwd(),'dataset','cases_malaysia_train.csv')

df_train = pd.read_csv(TRAIN_CSV_PATH)

#%%
#3. Data Inspection
df_train.info() #check datatype 
df_train.head()
df_train.describe()

#%%
#Check NaNs value in new_cases
print(df_train.isna().sum()) #0 NaNs

#%%
#check duplicates data
print(df_train.duplicated().sum()) #0 duplicates

#%%
#4. Data Cleaning
#Change the datatype of 'new_cases' column
df_train['cases_new'] = pd.to_numeric(df_train['cases_new'],errors='coerce')

#%%
#Check NaNs value
print(df_train.isna().sum()) #12 missing values
#%%
#Plot the data
plt.figure()
plt.plot(df_train['cases_new'])
plt.show()

#%%
#Interpolation - to fill missing values
df_train['cases_new'] = df_train['cases_new'].interpolate(method='polynomial',order=2)

#%%
#Change datatype into int
df_train['cases_new'] = df_train['cases_new'].astype('int64')

#%%
df_train.info()
#%%
#double check missing values
print(df_train.isna().sum()) #0

#%%
#5. Features Selection
#No Features to select

#%%
#6. Data Pre-Processing
#Normalization

#Expand Dimension
data = df_train['cases_new'].values #get the numpy array
data = data[::,None]

#%%
#Min-Max Scaling
mms = MinMaxScaler()
data = mms.fit_transform(data)

#%%
#Define x train and y train
win_size = 30
x_train = []
y_train = []

for i in range(win_size,len(data)):
    x_train.append(data[i-win_size:i])
    y_train.append(data[i])

#%%
#Convert x train and y train into numpy array
x_train = np.array(x_train)
y_train = np.array (y_train)

#%%
#Train Test Split
x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,random_state=123)

#%%
#7. Model Development
model = Sequential()
model.add(LSTM(64,input_shape=x_train.shape[1:],return_sequences=True))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dense(1))

model.summary()

#%% 
#Model Architecture
plot_model(model, show_shapes=True)

#%% Model Compile
model.compile(optimizer='adam',loss='mse',metrics=['mse','mape'])

#%%
#callbacks - early stopping and tensorboard
LOGS_PATH=os.path.join(os.getcwd(),'logs',datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback=TensorBoard(log_dir=LOGS_PATH)
early_stop_callback=EarlyStopping(monitor='val_loss',patience=5)

#%%
#Model Training
hist=model.fit(x_train,y_train,epochs=100,callbacks=[tensorboard_callback,early_stop_callback],validation_data=(x_test,y_test))
#%%
#8. Model Evaluation
#Plot training and validation loss

training_loss = hist.history['loss']
validation_loss = hist.history['val_loss']
epoch_no = hist.epoch

plt.figure(figsize=[10,10])
plt.plot(epoch_no,training_loss,label='Training Loss')
plt.plot(epoch_no,validation_loss,label='Validation Loss')
plt.legend()
plt.show()

#%%
#Load Testing Dataset
TEST_CSV_PATH=os.path.join(os.getcwd(),'dataset','cases_malaysia_test.csv')

df_test=pd.read_csv(TEST_CSV_PATH)

#%%
#Inspection testing dataset
df_test.info()
df_test.describe()
df_test.head(31)
#%%
#Check NaNs in testing dataset
print(df_test.isna().sum()) # 1 missing value

#%%
#Change datatype of cases_new
df_test['cases_new'] = pd.to_numeric(df_test['cases_new'],errors='coerce') #change to float

#%%
#Interpolation - to fill missing values
df_test['cases_new'] = df_test['cases_new'].interpolate(method='polynomial',order=2)

#%%
#Change datatype into int
df_test['cases_new'] = df_test['cases_new'].astype('int64')

#%%
#double check the datatype
df_test.info()

#%%
#Double check the NaNs value
print(df_test.isna().sum()) #No Nans

#%%
#Concatenation train and test data
concat=pd.concat((df_train['cases_new'],df_test['cases_new']))
concat=concat[len(concat)-win_size-len(df_test):]

concat=mms.transform(concat[::, None])

X_test=[]
Y_test=[]

for i in range(win_size,len(concat)):
    X_test.append(concat[i-win_size:i])
    Y_test.append(concat[i])

X_test=np.array(X_test)
Y_test=np.array(Y_test)

#%%
#Model Prediction - new covid cases based on testing dataset
predicted_cases=model.predict(X_test)

# %%
#to visualize the predicted new cases and actual new cases

plt.figure()
plt.plot(predicted_cases,color='purple')
plt.plot(Y_test,color='green')
plt.legend(['Predicted','Actual'])
plt.xlabel("Time")
plt.ylabel("Number of Cases")
plt.show()

#Inverse Transform
Y_test=mms.inverse_transform(Y_test)
predicted_cases=mms.inverse_transform(predicted_cases)

#%%
#metrics to evaluate the performance (to know how bad the graph is)
print("MAPE: " , mean_absolute_percentage_error(Y_test,predicted_cases))
print("MAE: " , mean_absolute_error(Y_test,predicted_cases))

#%%
#9. Model Saving
#save mms
with open('mms.pkl','wb') as f:
    pickle.dump(mms ,f) #to save the pickle model

#%%
#save model
model.save('model.h5')
# %%
