#!/usr/bin/env python
# coding: utf-8

# Loading Libraries:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import statsmodels.api as sm
import os
from scipy import stats
from scipy.stats import normaltest
import warnings 

warnings.filterwarnings('ignore')
color = sns.color_palette()
sns.set_style('darkgrid')


# Changing Directory:


os.chdir("....\Project\Sales Forcasting")
os.getcwd()



# Loading the dataset


train_df = pd.read_csv('train.csv', parse_dates = ['date'], index_col = ['date'])
test_df = pd.read_csv('test.csv')
print(train_df.shape, test_df.shape)
train_df.head()
#train_df.info()


# Exploring the dataset:


num_stores = len(train_df['store'].unique())
fig, axes = plt.subplots(num_stores, figsize=(8, 16))

for s in train_df['store'].unique():
    t = train_df.loc[train_df['store'] == s, 'sales'].resample('W').sum()
    ax = t.plot(ax=axes[s-1])
    ax.grid()
    ax.set_xlabel('Store' + str(s))
    ax.set_ylabel('sales')
fig.tight_layout();


# In[7]:


train_df = train_df.reset_index()
train_df.head()


# In[8]:


train_1_1 = train_df[(train_df['store'] == 1) & (train_df['item'] == 1)]
train_1_1.shape


# In[9]:


train_1_1.tail()


# Extarcting individual date aspects and storing them in new column:


train_1_1['year'] = train_1_1['date'].dt.year
train_1_1['month'] = train_1_1['date'].dt.month
train_1_1['day'] = train_1_1['date'].dt.day
train_1_1['weekday'] = train_1_1['date'].dt.weekday

train_1_1.head()


# Decomposing the timeseries:


from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(train_1_1['sales'], freq=365)

fig=plt.figure()
fig = result.plot()
fig.set_size_inches(20, 14)


# Buliding Stationarity function:


from statsmodels.tsa.stattools import adfuller

def test_stationarity(ts, window = 12, cutoff = 0.01):
    
    #detemining rolling statistics
    rolmean = ts.rolling(window).mean()
    rolestd = ts.rolling(window).std()
    
    #plot
    fig = plt.figure(figsize=(12,8))
    orig = plt.plot(ts, color = 'blue', label = 'Orignal')
    mean = plt.plot(rolmean, color = 'red', label = 'rolling mean')
    std = plt.plot(rolestd, color = 'black', label = 'rolling std')
    plt.legend(loc = 'best')
    plt.title('Rolling mean and std.dev')
    plt.show()
    
    #dickey fuller test
    print('The results of the test are : ')
    dftest = adfuller(ts, autolag = 'AIC', maxlag = 20)
    dfoutput = pd.Series(dftest[0:4], index=['Test statistic','p-value', 'No. of lags', 'No. of observations'])
    for key,values in dftest[4].items():
        dfoutput['Crtical value (%s)'%key] = values
    pvalue = dfoutput[1]
    if pvalue < cutoff:
        print('p-value = %.4f. The series is likely stationary.' % pvalue)
    else:
        print('p-value = %.4f. The series is likely not stationary.' % pvalue)
    
    
    print(dfoutput)
        


# In[14]:


test_stationarity(train_1_1['sales'])


# Plotting Acf and Pacf graph:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(train_1_1.sales, lags = 49, ax = ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(train_1_1.sales, lags = 49, ax = ax2)


# Differencing the timeseries by lag 7:


seven_diff = train_1_1.sales - train_1_1.sales.shift(7)
seven_diff.head()


# Testing stationarity of new timeseries:


seven_diff = seven_diff.dropna(inplace = False)
test_stationarity(seven_diff)


# In[18]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(seven_diff, lags = 49, ax = ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(seven_diff, lags = 49, ax = ax2)


# In[21]:


train_1_1.iloc[1734]


# In[22]:


p_values = range(0,2)
q_values = range(0,2)
d = 0
s = 7
P = 0
D = 1
Q = 1
y = train_1_1.sales[:1734]
#x = train_1_1.sales[:"09-30-2017"]


# Determining the best Arima model:


cnt = 0
for p in p_values:
    for q in q_values:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,order=(p,d,q),
                                                seasonal_order=(P,D,Q,s),
                                                enforce_stationarity=False,
                                                enforce_invertibility=False
                                               )
                results = mod.fit()
                cnt += 1
                if cnt % 50 :
                    print('Current Iter - {}, ARIMA{}x{}X{} & {}x{}x{}x7 - AIC:{}'.format(cnt, p, d, q, P, D, Q, results.aic))
            except:
                continue


# Running Arima model:


arima_model_1 = sm.tsa.SARIMAX(train_1_1.sales, order=(1,0,1), seasonal_order= (0,1,1,7), 
                              enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
print(arima_model_1.summary())


# Checking Residuals:


resid_1 = arima_model_1.resid
print(normaltest(resid_1))


# In[26]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid_1, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid_1, lags=40, ax=ax2)


# In[27]:


fig = plt.figure(figsize=(12,8))
ax0 = fig.add_subplot(211)

sns.distplot(resid_1 ,fit = stats.norm, ax = ax0) 

(mu, sigma) = stats.norm.fit(resid_1)
 
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('Residual distribution')


# In[28]:


arima_model_1.plot_diagnostics(figsize=(15, 12))


# In[29]:


train_1_1.loc[train_1_1['date'] == '2017-09-30']


# In[30]:


begin = 1734
stop = 1825
train_1_1['forecast_1'] = arima_model_1.predict(start = begin, end = stop, dynamic=True)
train_1_1.head()
#train_1_1.iloc[1734]


# In[31]:


train_1_1[1734:1826][['sales','forecast_1']].plot(figsize=(12,8))


# Defining Function to calculate Mape:


def mape_smape(a,f):
    mape = np.mean(abs((a-f)/a))*100
    smape = np.mean(np.abs(f-a) * 200/(np.abs(a)+abs(f)))
    print('MAPE : %.2f %% \nSMAPE: %.2f'% (mape,smape), "%")


# In[33]:


mape_smape(train_1_1[1734:1826]['sales'],train_1_1[1734:1826]['forecast_1'])


# Buliding Naive Forecast:


train_1_1['forecast_naive'] = " "


# In[35]:


len(train_1_1['forecast_naive'])


# In[36]:


train_1_1.loc[1824]


# In[37]:


a = list()
for i in range(1, len(train_1_1['forecast_naive'])):
       a.append(train_1_1.loc[i-1,'sales'])


# In[38]:


a.insert(0,0)


# In[39]:


train_1_1['forecast_naive'] = a
train_1_1.tail()


# In[40]:


train_1_1[1734:1826][['sales','forecast_naive']].plot(figsize=(12,8))


# In[41]:


mape_smape(train_1_1[1733:1826]['sales'],train_1_1[1733:1826]['forecast_naive'])


# Transforming dataset to include special events like holidays:


storeid = 1
itemid = 1
train_1_1_d = train_df[train_df['store']==storeid]
train_1_1_d = train_1_1_d[train_1_1_d['item']==itemid]
train_1_1_d['year'] = train_1_1_d['date'].dt.year - 2012
train_1_1_d['month'] = train_1_1_d['date'].dt.month
train_1_1_d['day'] = train_1_1_d['date'].dt.dayofyear
train_1_1_d['weekday'] = train_1_1_d['date'].dt.weekday

train_1_1_d.head()
train_1_1_d.tail()


# In[43]:


holiday = pd.read_csv('US_bank_holidays_2012_2020.csv',header=None, names = ['date', 'holiday'])
holiday['date'] = pd.to_datetime(holiday['date'], yearfirst = True)
holiday.head()


# In[44]:


train_1_1_d = train_1_1_d.merge(holiday, how='left', on='date')
train_1_1_d['holiday_bool'] = pd.notnull(train_1_1_d['holiday']).astype(int)
train_1_1_d = pd.get_dummies(train_1_1_d, columns = ['month','holiday','weekday'] , prefix = ['month','holiday','weekday'])

train_1_1_d.head()


# In[45]:


train_1_1_d.columns
#train_1_1_d.shape


# In[46]:


column_list =['date','year', 'day', 'holiday_bool',
       'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6',
       'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12',
       'holiday_Christmas Day', 'holiday_Columbus Day',
       'holiday_Independence Day', 'holiday_Labor Day',
       'holiday_Martin Luther King Jr. Day', 'holiday_Memorial Day',
       'holiday_New Year Day', 'holiday_Presidents Day (Washingtons Birthday)',
       'holiday_Thanksgiving Day', 'holiday_Veterans Day', 'weekday_0',
       'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5',
       'weekday_6']
column_list


# In[47]:


exog_data = train_1_1_d[column_list]
exog_data = exog_data.set_index('date')
exog_data.head()


# In[48]:


train_1_1_d = train_1_1_d.set_index('date')
train_1_1_d.head()


# In[49]:


begin = '2017-10-01'
stop = '2017-12-30'
stop1 = '2017-12-31'


# In[50]:


p_values = range(0,2)
q_values = range(0,2)
d = 0
s = 7
P = 0
D = 1
Q = 1
y = train_1_1_d.sales[:begin]
x = exog_data[:begin]


# Checking for best model:

cnt = 0
for p in p_values:
    for q in q_values:
            try:
                mod = sm.tsa.statespace.SARIMAX(exog=x,endog=y,order=(p,d,q),
                                                seasonal_order=(P,D,Q,s),
                                                enforce_stationarity=False,
                                                enforce_invertibility=False
                                               )
                results = mod.fit()
                cnt += 1
                if cnt % 50 :
                    print('Current Iter - {}, ARIMA{}x{}X{} & {}x{}x{}x7 - AIC:{}'.format(cnt, p, d, q, P, D, Q, results.aic))
            except:
                continue


# Running best chosen arima model:


arima_model_2 = sm.tsa.statespace.SARIMAX(endog=train_1_1_d.sales[:begin], exog=exog_data[:begin], order=(0,0,1),
                                          seasonal_order=(0,1,1,7),enforce_stationarity=False,enforce_invertibility=False).fit()
print(arima_model_2.summary())


# Checking Residuals:


resid_2 = arima_model_2.resid
print(normaltest(resid_2))


# In[54]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid_2, lags=49, ax=ax1)
ax1 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid_2, lags=49, ax=ax1)


# In[55]:


fig = plt.figure(figsize=(12,8))
ax3 = fig.add_subplot()

sns.distplot(resid_2, fit= stats.norm, ax=ax3)

(mu,sigma) = stats.norm.fit(resid_2)

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('frequency')
plt.xlabel('Residual Distribution')


# In[57]:


arima_model_2.plot_diagnostics(figsize=(15, 12))


# In[58]:


train_1_1_d['forecast_2'] = arima_model_2.predict(start=pd.to_datetime(begin), end = pd.to_datetime(stop1), 
                                              exog=exog_data[begin:stop], dynamic=True)
train_1_1_d[begin:stop]['forecast_2'].head()


# In[59]:


train_1_1_d[begin:stop][['sales', 'forecast_2']].plot(figsize=(12, 8))


# Checking Accuracy:


mape_smape(train_1_1_d[begin:stop1]['sales'],train_1_1_d[begin:stop1]['forecast_2'])


# In[61]:


arima_model_3 = sm.tsa.statespace.SARIMAX(endog=train_1_1_d.sales[:begin], exog=exog_data[:begin], order=(0,0,7),
                                          seasonal_order=(0,1,1,7),enforce_stationarity=False,enforce_invertibility=False).fit()
print(arima_model_3.summary())


# In[250]:


train_1_1_d['forecast_3'] = arima_model_3.predict(start=pd.to_datetime(begin), end = pd.to_datetime(stop1), 
                                              exog=exog_data[begin:stop], dynamic=True)
train_1_1_d[begin:stop]['forecast_3'].head()


# In[251]:


mape_smape(train_1_1_d[begin:stop1]['sales'],train_1_1_d[begin:stop1]['forecast_3'])


# Running the chosen model to forecast all the items of each store:


forecast = []
for s in range(1,11):
    for i in range(1,51):
        storeid = s
        itemid = i
        train = train_df[train_df['store']==storeid]
        train = train[train['item']==itemid]
        test = test_df[test_df['store']==storeid]
        test = test[test['item']==itemid]
        test.set_index('id')
        test.drop('id',axis = 1, inplace=True)
        test['sales'] = np.NAN

        df = pd.concat([train,test], axis = 0, join='outer')
        df['date'] = pd.to_datetime(df['date'])

        df['year'] = df['date'].dt.year - 2012
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.dayofyear
        df['weekday'] = df['date'].dt.weekday
        df = df.merge(holiday, how='left', on='date')
        df['holiday_bool'] = pd.notnull(df['holiday']).astype(int)
        df = pd.get_dummies(df, columns = ['month','holiday','weekday'] , prefix = ['month','holiday','weekday'])
        exog_data_2 = df[column_list]
        exog_data_2 = exog_data_2.set_index('date')
        df = df.set_index('date')
        begin = '2018-01-01'
        stop = '2018-03-30'
        stop1 = '2018-03-31'
        arima_model_4 = sm.tsa.statespace.SARIMAX(endog=df.sales[:begin], exog=exog_data_2[:begin], order=(0,0,1),
                                          seasonal_order=(0,1,1,7),enforce_stationarity=False,enforce_invertibility=False).fit()
        df['forecast'] = arima_model_4.predict(start=pd.to_datetime(begin), end = pd.to_datetime(stop1), 
                                              exog=exog_data_2[begin:stop], dynamic=True)
        
        forecast.append(df[begin:stop][['store','item','forecast']])
        print('item:',i,'store:',s,'Finished.')


# In[64]:


forecast[1]


# In[85]:


a = forecast[1]
a.tail()


# In[81]:


df2 = pd.DataFrame()
df2.info()


# In[89]:


df2 = df2.append(a)
df2.tail()


# In[94]:


len(forecast) + 1


# In[107]:


df_forecast = pd.DataFrame()
for i in range(0,len(forecast)):
    df_forecast = df_forecast.append(forecast[i])


# In[110]:


df_forecast = df_forecast.reset_index()


# In[116]:


df_forecast.tail()


# In[119]:


df_forecast.to_csv(r'F:\CSUF\Spring 2019\ISDS 577\Project\forecast.csv')

# End of Code



