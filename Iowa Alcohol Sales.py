################################################################################
# Matthew Spitulnik ############################################################
# Big Data Analytics ###########################################################
# Iowa Alchohol Sale ###########################################################
# Project Summary: For this project, I used publicly available Iowa tequila liquor 
# sales from 2012 through 2023 to try to build an ARIMA model that could forecast out 
# future sales. To do this, I experimented with various smoothing techniques to account 
# for seasonality that was being observed in the timeseries data.
################################################################################

################################################################################
### Install and load required packages #########################################
################################################################################

###First install the required pacakages
#%pip install pandas
#%pip install numpy
#%pip install scipy
#%pip install statsmodels
#%pip install matplotlib
#%pip install scikit-learn

# import packages for analysis and modeling
import pandas as pd #data frame operations
import numpy as np #arrays and math functions
from scipy.stats import uniform #for training and test splits
import statsmodels.api as smf #R-like model specification
import matplotlib.pyplot as plt #2D plotting

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMAResults
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings("ignore")
from scipy.signal import savgol_filter

################################################################################
### Load the initial data, explore it, and clean it ############################
################################################################################

###This creates a "default_directory" variable, where the directory path to the
# data files folder containing all of the required data sets is saved so that it
# does not need to be constantly re-entered. Remember to use forward slashes
# instead of back slashes in the directory path. For example, if the datafiles
# folder is saved in "C:\home\project\datafiles", then "C:/home/project"
# would be inserted between the quotes below.
default_directory = "<DEFAULT DIRECTORY PATH HERE>"

#Import the 2012-2016 tequila data.
tequila=pd.read_csv(f'{default_directory}/datafiles/2012_2016_Tequila.csv')

#Convert the data types per column as needed.
tequila['Store Number'] = tequila['Store Number'].astype(object)
tequila['County Number'] = tequila['County Number'].astype(object)
tequila['Category'] = tequila['Category'].astype(object)
tequila['Vendor Number'] = tequila['Vendor Number'].astype(object)
tequila['Item Number'] = tequila['Item Number'].astype(object)
tequila['Pack'] = tequila['Pack'].astype(float)
tequila['Bottle Volume (ml)'] = tequila['Bottle Volume (ml)'].astype(float)
tequila['Bottles Sold'] = tequila['Bottles Sold'].astype(float)

#Confirm the changes
print(tequila.info())
print(tequila.dtypes)

#group all of the dates together and sum together how many total bottles were sold that day
teq_by_date=tequila.groupby(["Date"])[["Bottles Sold","Sale (Dollars)"]].sum()
teq_by_date

#Check to make sure I have the expected number of dates
len(teq_by_date)
len(set(tequila['Date']))

#now import the remaining 2016 data and all of the 2017 data
tequila_2016=pd.read_csv(f'{default_directory}/datafiles/2016_Iowa_Liquor_Sales.csv')
tequila_2017=pd.read_csv(f'{default_directory}/datafiles/2017_Iowa_Liquor_Sales.csv')

#Now look at and clean up the 2016 data
print(tequila_2016.dtypes)
tequila_2016['Date']=pd.to_datetime(tequila_2016['Date'])
tequila_2016['Date'][(tequila_2016['Date']>'2016-07-31')&(tequila_2016['Date']<'2016-09-01')]
#final date I currently have in our data is 2016-08-26, starting the new data after that
tequila_2016['Date'][(tequila_2016['Date']>'2016-08-26')&(tequila_2016['Date']<'09-01-2016')]
tequila_2016_end=tequila_2016[(tequila_2016['Date']>'2016-08-26')]
print(tequila_2016_end)
#remove rows in the category name that have NA values
tequila_2016_end=tequila_2016_end.dropna(subset=['Category Name'])
#remove everything other than tequila
tequila_2016_end=tequila_2016_end[tequila_2016_end['Category Name'].str.contains('TEQUILA')]

#Now look at and clean up the 2017 data
tequila_2017['Date']=pd.to_datetime(tequila_2017['Date'])
#remove all NA values
tequila_2017=tequila_2017.dropna(subset=['Category Name'])
#remove everything other than tequila
tequila_2017=tequila_2017[tequila_2017['Category Name'].str.contains('TEQUILA')]

#combine the 2017 and 2016 data
temp_2016_2017=[tequila_2016_end,tequila_2017]
full_2016_2017_DF=pd.concat(temp_2016_2017)

#now groupby date and isolate Bottles Sold and Sale (Dollars)
teq_upto_2017=full_2016_2017_DF.groupby(["Date"])[["Bottles Sold","Sale (Dollars)"]].sum()
teq_upto_2017

################################################################################
#### Perform Initial Analysis ##################################################
################################################################################

#See if the bottle and revenue data is stationary. Create a fucntion that will take in the dataframe
#and go through and test the two stats.
def station_test(alc_df):
  tmp_df=pd.DataFrame()
  count=0
  for stat in alc_df.columns:
    result = adfuller(alc_df[stat], autolag='AIC')
    tmp_df.loc[count,'stat']=str(stat)
    tmp_df.loc[count,'ADF Stat'] = round(result[0], 6)
    tmp_df.loc[count,'p-value'] = round(result[1], 6)
    for key, value in result[4].items():
      tmp_df.loc[count,key]=value
    count=count+1
  return tmp_df

station_test(teq_by_date)
#These results make it look as if the data could be stationary, but general knowledge of alchohol sales would make me think there will
# be some seaonality to it. Going to look at some additional plots.

#perform additional plotting to look at the data
fig, ax = plt.subplots(figsize=(20,12))
teq_by_date['Bottles Sold'].plot()
plt.show()
fig, ax = plt.subplots(figsize=(20,12))
teq_by_date['Bottles Sold'][(teq_by_date.index > '2014-12-31') & (teq_by_date.index <'2016-01-01')].plot()
plt.show()
fig, ax = plt.subplots(figsize=(20,12))
teq_by_date['Bottles Sold'][(teq_by_date.index > '2013-12-31') & (teq_by_date.index <'2015-01-01')].plot()
plt.show()

plot_acf(teq_by_date['Bottles Sold'])
plot_pacf(teq_by_date['Bottles Sold'])

fig, ax = plt.subplots(figsize=(20,12))
teq_by_date['Sale (Dollars)'].plot()
plt.show()
fig, ax = plt.subplots(figsize=(20,12))
teq_by_date['Sale (Dollars)'][(teq_by_date.index >= '2014-12-31') & (teq_by_date.index <'2016-01-01')].plot()
plt.show()
fig, ax = plt.subplots(figsize=(20,12))
teq_by_date['Sale (Dollars)'][(teq_by_date.index >= '2013-12-31') & (teq_by_date.index <'2015-01-01')].plot()
plt.show()

plot_acf(teq_by_date['Sale (Dollars)'])
plot_pacf(teq_by_date['Sale (Dollars)'])

#Based on these plots, it does appear there is some seasonality, will try to smooth that out later,
# but first will model the raw data as is.

################################################################################
#### Begin Modelling Process: Raw data, no smoothing ###########################
################################################################################

#First going to look at modeling using the raw data with no smoothing.

#Define a function that will model the data using ARIMA based on inputted p,d,q values.
def evaluate_arima_raw(X, arima_order):
    # prepare training dataset
    X = X.astype('float32')
    train_size = int(len(X) * 0.70)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(pd.Series(history), order=arima_order,trend='n')
        model_fit = model.fit()
        yhat = model_fit.forecast()
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    mse = mean_squared_error(test, predictions)
    rmse = sqrt(mse)
    return rmse

evaluate_arima_raw(teq_by_date['Bottles Sold'],(0,0,0))

#now create the function that runs the models based on the inputted DF, p, d, and q values,
#then creates a DF storing all of the data.
def evaluate_raw_models(dataframe, p_values, d_values, q_values):
  #file_num=0
  #file_num=135
  file_num=207
  full_DF=pd.DataFrame()
  for p in p_values:
    for d in d_values:
      for q in q_values:
        order = (p,d,q)
        tmp_DF=pd.DataFrame()
        for stat in dataframe.columns:
            dataset=dataframe[stat]
            try:
              mse = evaluate_arima_raw(dataset, order)
              tmp_DF.loc[0,'p']=p
              tmp_DF.loc[0,'d']=d
              tmp_DF.loc[0,'q']=q
              tmp_DF.loc[0,stat]=mse
            except:
              tmp_DF.loc[0,'p']=p
              tmp_DF.loc[0,'d']=d
              tmp_DF.loc[0,'q']=q
              tmp_DF.loc[0,stat]=0
              continue
        tmp_DF.to_csv(f'{default_directory}/datafiles/backups/backups_raw/tmp_DF_{file_num}_raw.csv',header=True,index=False)
        file_num=file_num+1
        temp_comb_DF=[full_DF,tmp_DF]
        full_DF=pd.concat(temp_comb_DF)
  return full_DF

'''This code went through and tested the 243 different combinations of p,d,q values, but it took 10+ hours to run.
#The function that did the testing saved the output of each into into 243 different files, which I then imported
# all at once so that the best p,d,q values could be determined. The resulting DF that contains all of that data will
# be manually imported to avoid having to deal with the timeframe again.
p_values = range(0, 9)
d_values = range(0, 3)
q_values = range(0, 9)
arima_test_check_raw=evaluate_raw_models(teq_by_date, p_values, d_values, q_values)
arima_test_check_raw
'''

#now import and combine all of the tests to determine which combination of p,d,q values produces the best RMSE value
'''The overall file will be imported below, no need to run this code again.
RMSE_tests_DF_raw=pd.read_csv(f'{default_directory}/datafiles/backups/backups_raw/tmp_DF_0_raw.csv')
for i in range(1,243):
  tmp_RMSE_DF_raw=pd.read_csv(f'{default_directory}/datafiles/backups/backups_raw/tmp_DF_{i}_raw.csv')
  temp_comb_DF_raw=[RMSE_tests_DF_raw,tmp_RMSE_DF_raw]
  RMSE_tests_DF_raw=pd.concat(temp_comb_DF_raw)
RMSE_tests_DF_raw=RMSE_tests_DF_raw.reset_index(drop=True)
RMSE_tests_DF_raw
'''

#Export and import the file to avoid having to go through the above steps every time.
#RMSE_tests_DF_raw.to_csv('RMSE_tests_DF_raw.csv',index=False, header=True)
RMSE_tests_DF_raw=pd.read_csv(f'{default_directory}/datafiles/backups/backups_raw/RMSE_tests_DF_raw.csv')
RMSE_tests_DF_raw

#now create the function that will lock down the best RMSE values and their respective pdq order,
#then store it in a dataframe.
def best_arima(RMSE_output):
  best_RMSE_DF=pd.DataFrame()
  for col in RMSE_output.columns[3:len(RMSE_output.columns)]:
    tmp_col=RMSE_output[col][RMSE_output[col]!=0]
    min_val=tmp_col.min()
    min_index=(RMSE_output[col]==min_val).argmax()
    best_p=RMSE_output.loc[min_index,'p']
    best_d=RMSE_output.loc[min_index,'d']
    best_q=RMSE_output.loc[min_index,'q']
    best_RMSE_DF.loc[0,col]=''
    best_RMSE_DF=best_RMSE_DF.astype({col: 'object'})
    best_RMSE_DF.at[0,col]=(best_p,best_d,best_q)
  return best_RMSE_DF

#now run the function to create the DF containing the best p,d,q combinations for each stat
best_RMSE_DF_raw=best_arima(RMSE_tests_DF_raw)
best_RMSE_DF_raw

#now create the function that builds and saves the model using best p,d,q combinations
def best_model_raw(timeseries_DF,best_arima_out,use_bias='no'):
  for col in timeseries_DF.columns:
    # fit model
    model = ARIMA(timeseries_DF[col], order=best_arima_out.loc[0,str(col)],trend='n')
    model_fit = model.fit()
    # bias
    if use_bias == 'yes':
        residuals=model_fit.resid
        bias = residuals.mean()
        np.save(f'{default_directory}/models/model_bias_{col}_raw.npy', [bias])
    # save model
    model_fit.save(f'{default_directory}/models/model_{col}_raw.pkl')

#build and export the best models
best_model_raw(teq_by_date,best_RMSE_DF_raw,'yes')
best_model_raw(teq_by_date,best_RMSE_DF_raw)

#now create the function for making the predictions
def arima_forecast_validate_raw(timeseries_DF,test_DF,best_arima_out,use_bias='no'):
  pred_val_DF=pd.DataFrame()
  pred_val_DF.index=test_DF.index
  y_pred_dict={}
  for col in timeseries_DF.columns:
      dataset=timeseries_DF[col]
      X = dataset.values.astype('float32')
      history = [x for x in X]
      validation = test_DF[col]
      y = validation.values.astype('float32')
      # load model
      model_fit = ARIMAResults.load(f'{default_directory}/models/model_{col}_raw.pkl')
      if use_bias=='yes':
          bias = np.load(f'{default_directory}/models/model_bias_{col}_raw.npy')
      else:
          bias=np.array([0])
      # make first prediction
      predictions = list()
      yhat = float(model_fit.forecast())
      yhat = bias + yhat
      predictions.append(yhat)
      history.append(y[0])
      pred_val_DF.loc[pred_val_DF.index[0],f'{str(col)}_Pred']="{:.2f}".format(yhat[0])
      pred_val_DF.loc[pred_val_DF.index[0],f'{str(col)}_Exp']="{:.2f}".format(y[0])
      print('Stat: ', col)
      print('>Predicted=%.3f, Expected=%3.f' % (yhat, y[0]))
      #make additional predictions
      for i in range(1, len(y)):
        model = ARIMA(pd.Series(history), order=best_arima_out.loc[0,str(col)],trend='n')
        model.initialize_approximate_diffuse()
        model_fit = model.fit()
        yhat = float(model_fit.forecast())
        yhat = bias + yhat
        predictions.append(yhat)
        obs = y[i]
        history.append(obs)
        pred_val_DF.loc[pred_val_DF.index[i],f'{str(col)}_Pred']="{:.2f}".format(yhat[0])
        pred_val_DF.loc[pred_val_DF.index[i],f'{str(col)}_Exp']="{:.2f}".format(obs)
        print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
      # report performance
      mse = mean_squared_error(test_DF[col], predictions)
      rmse = sqrt(mse)
      print('RMSE: %.3f' % rmse)
      pred_val_DF.loc['RMSE',f'{str(col)}_Pred']=rmse
      pred_val_DF.loc['RMSE',f'{str(col)}_Exp']=rmse
      plt.plot(np.array(test_DF[col]),color='blue')
      plt.plot(predictions, color='red')
      plt.title(f'Stat: {col}')
      plt.show()
      y_pred_dict[f'y_{col}']=test_DF[col]
      y_pred_dict[f'predictions_{col}']=predictions
  return pred_val_DF, y_pred_dict['y_Bottles Sold'], y_pred_dict['predictions_Bottles Sold'],\
  y_pred_dict['y_Sale (Dollars)'], y_pred_dict['predictions_Sale (Dollars)']

#now make the predictions
Preds_2017_raw_bias,exp_2017_btl_raw_bias,pred_2017_btl_raw_bias,exp_2017_sale_raw_bias,pred_2017_sale_raw_bias=\
arima_forecast_validate_raw(teq_by_date,teq_upto_2017,best_RMSE_DF_raw,'yes')

#want to get a cleaner plot, going to try just plotting the 2016 prediction data
#find how many values are from 2016
Preds_2017_raw_bias[:-1][Preds_2017_raw_bias[:-1].index<pd.to_datetime('2017-01-01')] #first 88 rows
#now plot just the 2016 bottle data
plt.plot(np.array(exp_2017_btl_raw_bias[:88]),color='blue')
plt.plot(pred_2017_btl_raw_bias[:88], color='red')
plt.title('Stat: Btl, Just 2016')
plt.show()

#now the 2016 revenue data
plt.plot(np.array(exp_2017_sale_raw_bias[:88]),color='blue')
plt.plot(pred_2017_sale_raw_bias[:88], color='red')
plt.title('Stat: Sale, Just 2016')
plt.show()

#its still a little messy, try looking at just one months worth of 2016 data (December)
Preds_2017_raw_bias[:-1][(Preds_2017_raw_bias[:-1].index>pd.to_datetime('2016-11-30')) & \
 (Preds_2017_raw_bias[:-1].index<pd.to_datetime('2017-01-01'))]
list(Preds_2017_raw_bias.index).index(pd.to_datetime('2016-12-01 00:00:00'))#index of first december date is 66
list(Preds_2017_raw_bias.index).index(pd.to_datetime('2016-12-30 00:00:00'))#index of first december date is 87
#now plot just the December 2016 data
plt.plot(np.array(exp_2017_btl_raw_bias[66:88]),color='blue')
plt.plot(pred_2017_btl_raw_bias[66:88], color='red')
plt.title('Stat: Btl, Just December 2016')
plt.show()

#now the revenue data
plt.plot(np.array(exp_2017_sale_raw_bias[66:88]),color='blue')
plt.plot(pred_2017_sale_raw_bias[66:88], color='red')
plt.title('Stat: Sale, Just December 2016')
plt.show()

#going to see how much of a difference it makes with a bias of zero
Preds_2017_raw,exp_2017_btl_raw,pred_2017_btl_raw,exp_2017_sale_raw,pred_2017_sale_raw=\
arima_forecast_validate_raw(teq_by_date,teq_upto_2017,best_RMSE_DF_raw,'no')

#now plot just the 2016 data
#find how many values are from 2016
Preds_2017_raw[:-1][Preds_2017_raw[:-1].index<pd.to_datetime('2017-01-01')] #first 88 rows
#now plot just the 2016 data
plt.plot(np.array(exp_2017_btl_raw[:88]),color='blue')
plt.plot(pred_2017_btl_raw[:88], color='red')
plt.title('Stat: Btl, Just 2016')
plt.show()

#now the revenue data
plt.plot(np.array(exp_2017_sale_raw[:88]),color='blue')
plt.plot(pred_2017_sale_raw[:88], color='red')
plt.title('Stat: Sale, Just 2016')
plt.show()

#its still a little messy, try looking at just one months worth of 2016 data (December)
Preds_2017_raw[:-1][(Preds_2017_raw[:-1].index>pd.to_datetime('2016-11-30')) & \
 (Preds_2017_raw[:-1].index<pd.to_datetime('2017-01-01'))]
list(Preds_2017_raw.index).index(pd.to_datetime('2016-12-01 00:00:00'))#index of first december date is 66
list(Preds_2017_raw.index).index(pd.to_datetime('2016-12-30 00:00:00'))#index of first december date is 87
#now plot just the December 2016 data
plt.plot(np.array(exp_2017_btl_raw[66:88]),color='blue')
plt.plot(pred_2017_btl_raw[66:88], color='red')
plt.title('Stat: Btl, Just December 2016')
plt.show()

#now the revenue data
plt.plot(np.array(exp_2017_sale_raw[66:88]),color='blue')
plt.plot(pred_2017_sale_raw[66:88], color='red')
plt.title('Stat: Sale, Just December 2016')
plt.show()

####Bottle RMSE, raw data, no bias: 1376.39
####Bottle RMSE, raw data, with bias: 1376.17
####Revenue RMSE, raw data, no bias: 24454.2
####Revenue RMSE, raw data, with bias: 24434.8

################################################################################
#### Modeling With Smoothed Data: Difference Function ##########################
################################################################################

##Now look at how smoothing the data with the difference function from class affects the models.
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return pd.Series(diff)

#Create a function for smoothing the data
def smoothing_and_check(df, stat, interval):
    X = df[stat]
    X = X.astype('float32')
    # difference data
    tmp_int = interval
    stationary = difference(X, tmp_int)
    stationary.index = df.index[tmp_int:]
    # check if stationary
    result = adfuller(stationary)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
    	print('\t%s: %.3f' % (key, value))
    for i in [2012,2013,2014,2015]:
        fig, ax = plt.subplots(figsize=(20,12))
        stationary[(stationary.index >= f'{i}-01-31') & (stationary.index <f'{i+1}-01-01')].plot()
        plt.title(f'Year: {i}, Interval: {tmp_int}')
        plt.show()
    plot_acf(stationary)
    plt.title(f'ACF Interval: {tmp_int}')
    plt.show()
    plot_pacf(stationary)
    plt.title(f'PACF Interval: {tmp_int}')
    plt.show()
    return stationary

#Now try to smooth the revenue data with an interval of 4, since the seasonality may literally be related to the 4 seasons.
smooth_4=smoothing_and_check(teq_by_date,'Sale (Dollars)',4)
print(smooth_4)
#Now try to smooth the bottle data with an interval of 4, since the seasonality may literally be related to the 4 seasons.
smooth_4_btl=smoothing_and_check(teq_by_date,'Bottles Sold',4)
print(smooth_4_btl)

#The plots still hint at some seasonality, going to try 365 for days of the year instead.
smooth_365=smoothing_and_check(teq_by_date,'Sale (Dollars)',365)
print(smooth_365)
smooth_365_btl=smoothing_and_check(teq_by_date,'Bottles Sold',365)
print(smooth_365_btl)

#365 brings us in the wrong direction, trying for 12 months in a year now.
smooth_12=smoothing_and_check(teq_by_date,'Sale (Dollars)',12)
print(smooth_12)
smooth_12_btl=smoothing_and_check(teq_by_date,'Bottles Sold',12)
print(smooth_12_btl)

#See what happens if I just use 1
smooth_1=smoothing_and_check(teq_by_date,'Sale (Dollars)',1)
#Now try to smooth the revinue data with an interval of, since the seasonality may literally be related to the 4 seasons.
smooth_1_btl=smoothing_and_check(teq_by_date,'Bottles Sold',1)
print(smooth_1)
print(smooth_1_btl)

#Try an interval of 3, for about the number of months per season
smooth_3=smoothing_and_check(teq_by_date,'Sale (Dollars)',3)
#Now try to smooth the revinue data with an interval of, since the seasonality may literally be related to the 4 seasons.
smooth_3_btl=smoothing_and_check(teq_by_date,'Bottles Sold',3)
print(smooth_3)
print(smooth_3_btl)

#Try an interval of 30
smooth_30=smoothing_and_check(teq_by_date,'Sale (Dollars)',30)
#Now try to smooth the revinue data with an interval of, since the seasonality may literally be related to the 4 seasons.
smooth_30_btl=smoothing_and_check(teq_by_date,'Bottles Sold',30)
print(smooth_30)
print(smooth_30_btl)

#Going to try something different...going to try using values 1 to 1000 as an interval and see
#what the best values end up being.
#Creating a function to do this:
def smoothing_iteration(df, stat,min_int,max_int):
    X = df[stat]
    X = X.astype('float32')
    tmp_df=pd.DataFrame()
    count=0
    for i in range(min_int,max_int):
        # difference data
        tmp_int = i
        stationary = difference(X, tmp_int)
        stationary.index = df.index[tmp_int:]
        # check if stationary
        result = adfuller(stationary)
        tmp_df.loc[count,'interval']=i
        tmp_df.loc[count,'ADF_stat']=result[0]
        tmp_df.loc[count,'p-val']=result[1]
        for key, value in result[4].items():
            tmp_df.loc[count,key]=value
        count=count+1
    return tmp_df

#now try to find the best interval to use for the revenue smoothing function
smoothing_sale_DF=smoothing_iteration(teq_by_date,'Sale (Dollars)',1,1001)
print(smoothing_sale_DF)
#find the lowest ADF value
print(min(smoothing_sale_DF['ADF_stat'])) #-12.09037434861305
#get the index for lowest ADV value
print((smoothing_sale_DF['ADF_stat']==min(smoothing_sale_DF['ADF_stat']))).argmax() #653
print(smoothing_sale_DF.loc[653,:])
#based on this output, I will use 654 as the interval (653 is the index, but 654 is the interval)

#now look at the bottles data
smoothing_btl_DF=smoothing_iteration(teq_by_date,'Bottles Sold',1,1001)
print(smoothing_btl_DF)
#find the lowest ADF value
print(min(smoothing_btl_DF['ADF_stat'])) #-13.514850590425477
#get the index for lowest ADV value
(smoothing_btl_DF['ADF_stat']==min(smoothing_btl_DF['ADF_stat'])).argmax() #657
smoothing_btl_DF.loc[657,:]
#based on this output, I will use 658 as the interval (657 is the index, but 658 is the interval)

#put 658 through the original smoothing function now to look at the plots:
smooth_btl_658=smoothing_and_check(teq_by_date,'Bottles Sold',658)

#Now create the functions that will allow us to build and test the best possible values for an ARIMA model

#Define the function to invert the difference:
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

#Define the function for performing the ARIMA modeling.
def evaluate_arima_model(X, stat, arima_order):
    # prepare training dataset
    X = X.astype('float32')
    train_size = int(len(X) * 0.70)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        if stat == 'Sale (Dollars)':
            interval_num=654
        elif stat == 'Bottles Sold':
            interval_num=658
        diff = difference(history, interval_num)
        model   = ARIMA(diff, order=arima_order,trend='n')
        model_fit = model.fit()
        yhat = model_fit.forecast()
        yhat = inverse_difference(history, yhat, interval_num)
        predictions.append(yhat)
        history.append(test[t])
	# calculate out of sample error
    mse = mean_squared_error(test, predictions)
    rmse = sqrt(mse)
    return rmse

#now create the function that runs the models based on the inputted DF, p, d, and q values, then creates a DF storing all of the data.
def evaluate_models(dataframe, p_values, d_values, q_values):
  file_num=0
  #file_num=135
  full_DF=pd.DataFrame()
  for p in p_values:
    for d in d_values:
      for q in q_values:
        order = (p,d,q)
        tmp_DF=pd.DataFrame()
        for stat in dataframe.columns:
          dataset = dataframe[stat].astype('float32')
          try:
            mse = evaluate_arima_model(dataset, stat, order)
            tmp_DF.loc[0,'p']=p
            tmp_DF.loc[0,'d']=d
            tmp_DF.loc[0,'q']=q
            tmp_DF.loc[0,stat]=mse
          except:
            tmp_DF.loc[0,'p']=p
            tmp_DF.loc[0,'d']=d
            tmp_DF.loc[0,'q']=q
            tmp_DF.loc[0,stat]=0
            continue
        tmp_DF.to_csv(f'{default_directory}/datafiles/backups/backups_diff_smoothing/tmp_DF_{file_num}.csv',header=True,index=False)
        file_num=file_num+1
        temp_comb_DF=[full_DF,tmp_DF]
        full_DF=pd.concat(temp_comb_DF)
  return full_DF

#Test the function to make sure it is working
'''
p_values = range(0, 2)
d_values = range(0, 1)
q_values = range(0, 2)
arima_test_check=evaluate_models(teq_by_date, p_values, d_values, q_values)
arima_test_check
'''

#Now run the function to explore the best p,d,q values:
'''This once again took 10+ hours, I will import the end result down below.
p_values = range(0, 9)
d_values = range(0, 3)
q_values = range(0, 9)
teq_bestfit_check=evaluate_models(teq_by_date, p_values, d_values, q_values)
'''

'''#The end result will be imported below.
RMSE_tests_DF=pd.read_csv(f'{default_directory}/datafiles/backups/backups_diff_smoothing/tmp_DF_0.csv')
for i in range(1,243):
  tmp_RMSE_DF=pd.read_csv(f'{default_directory}/datafiles/backups/backups_diff_smoothing/tmp_DF_{i}.csv')
  temp_comb_DF=[RMSE_tests_DF,tmp_RMSE_DF]
  RMSE_tests_DF=pd.concat(temp_comb_DF)
RMSE_tests_DF=RMSE_tests_DF.reset_index(drop=True)
RMSE_tests_DF
'''

#RMSE_tests_DF.to_csv(f'{default_directory}/datafiles/backups/backups_diff_smoothing/RMSE_tests_DF.csv',index=False, header=True)
RMSE_tests_DF=pd.read_csv(f'{default_directory}/datafiles/backups/backups_diff_smoothing/RMSE_tests_DF.csv')
RMSE_tests_DF

print(min(RMSE_tests_DF['Bottles Sold'][RMSE_tests_DF['Bottles Sold']!=0])) #1678.1289458820818
min(RMSE_tests_DF['Sale (Dollars)'][RMSE_tests_DF['Sale (Dollars)']!=0]) #28965.500587625585

#for testing purposes, redo the arima testing function to only use the interval
# that I want for Sale (Dollars) to make sure it was working properly.
# Define the function for performing the ARIMA modeling.
def evaluate_arima_test(X,arima_order,int_num):
    # prepare training dataset
    X = X.astype('float32')
    train_size = int(len(X) * 0.70)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        interval_num=int_num
        diff = difference(history, interval_num)
        model   = ARIMA(diff, order=arima_order,trend='n')
        model_fit = model.fit()
        yhat = model_fit.forecast()
        yhat = inverse_difference(history, yhat, interval_num)
        predictions.append(yhat)
        history.append(test[t])
	# calculate out of sample error
    mse = mean_squared_error(test, predictions)
    rmse = sqrt(mse)
    return rmse

#now test
print(evaluate_arima_test(teq_by_date['Sale (Dollars)'],(0,0,0),654)) #31870.23455920852
evaluate_arima_test(teq_by_date['Bottles Sold'],(0,0,0),658) #1782.8805298486366

#low intervals from our testing earlier still showed good ADF stats,
#checking what those RMSE results would be like
print(evaluate_arima_test(teq_by_date['Sale (Dollars)'],(0,0,0),3)) #33316.06949292517
evaluate_arima_test(teq_by_date['Sale (Dollars)'],(0,0,0),1) #34343.31605196802

#find the top 10 ADF values, see if any of them will lower the RMSE
best_sale_smooth=smoothing_sale_DF.sort_values('ADF_stat').head(15)
best_sale_smooth
for i in best_sale_smooth['interval']:
    print(evaluate_arima_test(teq_by_date['Sale (Dollars)'],(0,0,0),int(i)))

#see if larger intervals are needed/make a difference
smoothing_sale_lrg_DF=smoothing_iteration(teq_by_date,'Sale (Dollars)',1000,1020)
#1020 is the highest I can go without recieving an error
#Now find the lowest ADF value
print(min(smoothing_sale_lrg_DF['ADF_stat'])) #-70.727821
#get the index for lowest ADV value
print((smoothing_sale_lrg_DF['ADF_stat']==min(smoothing_sale_lrg_DF['ADF_stat'])).argmax()) #2
smoothing_sale_lrg_DF.loc[2,:] #based on this output, I will use 1002 as the interval (2 is the index, but 1002 is the interval)

#check RMSE with 1002 as the interval
evaluate_arima_test(teq_by_date['Sale (Dollars)'],(0,0,0),1002)
#returns "ValueError: Pandas data cast to numpy dtype of object. Check input data with np.asarray(data)."
#It appears if the interval gets too big it will not work.

#A little more testing with the revenue data.
#isolate the revenue data
teq_sale_date=pd.DataFrame()
teq_sale_date['Sale (Dollars)']=teq_by_date['Sale (Dollars)']
teq_sale_date

#create a new function for testing p,d,q that doesn't specify intervals to test the revenue data by itself
def evaluate_models_sale(dataframe, p_values, d_values, q_values,int_num):
  file_num=0
  #file_num=135
  full_DF=pd.DataFrame()
  for p in p_values:
    for d in d_values:
      for q in q_values:
        order = (p,d,q)
        tmp_DF=pd.DataFrame()
        for stat in dataframe.columns:
          dataset = dataframe[stat].astype('float32')
          try:
            mse = evaluate_arima_test(dataset, order, int_num)
            tmp_DF.loc[0,'p']=p
            tmp_DF.loc[0,'d']=d
            tmp_DF.loc[0,'q']=q
            tmp_DF.loc[0,stat]=mse
          except:
            tmp_DF.loc[0,'p']=p
            tmp_DF.loc[0,'d']=d
            tmp_DF.loc[0,'q']=q
            tmp_DF.loc[0,stat]=0
            continue
        tmp_DF.to_csv(f'{default_directory}/datafiles/backups/backups_diff_smoothing/tmp_DF_{file_num}.csv',header=True,index=False)
        file_num=file_num+1
        temp_comb_DF=[full_DF,tmp_DF]
        full_DF=pd.concat(temp_comb_DF)
  return full_DF

'''The below may take a little while to run.
#now test
p_values = range(0, 1)
d_values = range(0, 1)
q_values = range(0, 9)
teq_bestfit_check=evaluate_models_sale(teq_sale_date, p_values, d_values, q_values,1002)
teq_bestfit_check
max(teq_bestfit_check['Sale (Dollars)'])
#this is not producing any data.
'''

#making sure the new function is working at all
p_values = range(0, 1)
d_values = range(0, 1)
q_values = range(0, 1)
teq_bestfit_btl_test=evaluate_models_sale(teq_by_date, p_values, d_values, q_values,4)
teq_bestfit_btl_test

#going to try dividing all of the data by 10000 to see if that makes it possible to produce better results
teq_sale_date_red=teq_sale_date/10000
teq_sale_date_red

#try the smoothing interval operation again to find the best smoothing interval
smth_sale_red=smoothing_iteration(teq_sale_date_red,'Sale (Dollars)',1,1001)
smth_sale_red

#find the lowest ADF value
print(min(smth_sale_red['ADF_stat'])) #-12.09037434861305
#get the index for lowest ADV value
print((smth_sale_red['ADF_stat']==min(smth_sale_red['ADF_stat'])).argmax()) #653
print(smth_sale_red.loc[653,:])
#based on this output, I will use 654 as the interval (653 is the index, but 654 is the interval)
#comes out with 654 as the interval still, which makes sense since all numbers were divided
# by the same thing, still going to check if it makese the RMSE any better

'''The below may take a little while to run.
#testing the reduced sales data
p_values = range(0, 1)
d_values = range(0, 1)
q_values = range(0, 9)
teq_bestfit_red=evaluate_models_sale(teq_sale_date_red, p_values, d_values, q_values,653)
teq_bestfit_red
'''#The below may take a little while to run.

#the RMSE is relative and not a definitive value, so just going to try modeling the data now to see how it looks
#lock down the best RMSE values and their respective pdq order
best_RMSE_DF=best_arima(RMSE_tests_DF)
best_RMSE_DF

#function for building the best model and saving/exporting it
def best_model(timeseries_DF,best_arima_out):
  for col in timeseries_DF.columns:
    X = timeseries_DF[col]
    X = X.astype('float32')
    # difference data
    if col == 'Sale (Dollars)':
        num_to_use=654
    elif col == 'Bottles Sold':
        num_to_use=658
    diff = difference(X, num_to_use)
    # fit model
    model = ARIMA(diff, order=best_arima_out.loc[0,str(col)],trend='n')
    model_fit = model.fit()
    # bias
    bias = 0
    # save model
    model_fit.save(f'{default_directory}/models/model_{col}.pkl')
    np.save(f'{default_directory}/models/model_bias_{col}.npy', [bias])

#now get the best model
best_model(teq_by_date,best_RMSE_DF)

#now create the function that will be used to predict and test the remaining 2016/2017 data
def arima_forecast_validate(timeseries_DF,test_DF,best_arima_out):
  pred_val_DF=pd.DataFrame()
  pred_val_DF.index=test_DF.index
  y_pred_dict={}
  for col in timeseries_DF.columns:
    dataset=timeseries_DF[col]
    X = dataset.values.astype('float32')
    history = [x for x in X]
    if col == 'Sale (Dollars)':
        num_to_use=654
    elif col == 'Bottles Sold':
        num_to_use=658
    validation = test_DF[col]
    y = validation.values.astype('float32')
    # load model
    model_fit = ARIMAResults.load(f'{default_directory}/models/model_{col}.pkl')
    bias = np.load(f'{default_directory}/models/model_bias_{col}.npy')
    # make first prediction
    predictions = list()
    yhat = float(model_fit.forecast())
    yhat = bias + inverse_difference(history, yhat, num_to_use)
    predictions.append(yhat)
    history.append(y[0])
    pred_val_DF.loc[pred_val_DF.index[0],f'{str(col)}_Pred']="{:.2f}".format(yhat[0])
    pred_val_DF.loc[pred_val_DF.index[0],f'{str(col)}_Exp']="{:.2f}".format(y[0])
    print('Stat: ', col)
    print('>Predicted=%.3f, Expected=%3.f' % (yhat, y[0]))
    for i in range(1, len(y)):
      diff = difference(history, num_to_use)
      model = ARIMA(diff, order=best_arima_out.loc[0,str(col)],trend='n')
      model.initialize_approximate_diffuse()
      model_fit = model.fit()
      yhat = float(model_fit.forecast())
      yhat = bias + inverse_difference(history, yhat, num_to_use)
      predictions.append(yhat)
      obs = y[i]
      history.append(obs)
      pred_val_DF.loc[pred_val_DF.index[i],f'{str(col)}_Pred']="{:.2f}".format(yhat[0])
      pred_val_DF.loc[pred_val_DF.index[i],f'{str(col)}_Exp']="{:.2f}".format(obs)
      print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
    # report performance
    mse = mean_squared_error(y, predictions)
    rmse = sqrt(mse)
    print('RMSE: %.3f' % rmse)
    pred_val_DF.loc['RMSE',f'{str(col)}_Pred']=rmse
    pred_val_DF.loc['RMSE',f'{str(col)}_Exp']=rmse
    plt.plot(y,color='blue')
    plt.plot(predictions, color='red')
    plt.title(f'Stat: {col}')
    plt.show()
    y_pred_dict[f'y_{col}']=y
    y_pred_dict[f'predictions_{col}']=predictions
  return pred_val_DF, y_pred_dict['y_Bottles Sold'], y_pred_dict['predictions_Bottles Sold'],\
  y_pred_dict['y_Sale (Dollars)'], y_pred_dict['predictions_Sale (Dollars)']

#now make the predictions
Preds_2017,y_2017_btl,pred_2017_btl,y_2017_sale,pred_2017_sale=\
arima_forecast_validate(teq_by_date,teq_upto_2017,best_RMSE_DF)

#want to get a cleaner plot, going to try just plotting the 2016 prediction data
#find how many values are from 2016
Preds_2017[:-1][Preds_2017[:-1].index<pd.to_datetime('2017-01-01')] #first 88 rows
#now plot just the 2016 data
plt.plot(y_2017_btl[:88],color='blue')
plt.plot(pred_2017_btl[:88], color='red')
plt.title('Stat: Btl, Just 2016')
plt.show()
plt.plot(y_2017_sale[:88],color='blue')
plt.plot(pred_2017_sale[:88], color='red')
plt.title('Stat: Sale, Just 2016')
plt.show()

#its still a little messy, try looking at just one months worth of 2016 data (December)
Preds_2017[:-1][(Preds_2017[:-1].index>pd.to_datetime('2016-11-30')) & (Preds_2017[:-1].index<pd.to_datetime('2017-01-01'))]
list(Preds_2017.index).index(pd.to_datetime('2016-12-01 00:00:00'))#index of first december date is 66
list(Preds_2017.index).index(pd.to_datetime('2016-12-30 00:00:00'))#index of first december date is 87
#now plot just the December 2016 data
plt.plot(y_2017_btl[66:88],color='blue')
plt.plot(pred_2017_btl[66:88], color='red')
plt.title('Stat: Btl, Just December 2016')
plt.show()
plt.plot(y_2017_sale[66:88],color='blue')
plt.plot(pred_2017_sale[66:88], color='red')
plt.title('Stat: Sale, Just December 2016')
plt.show()

#The above was performed with bias set to zero, going to run the tests again and see if calculating the bias and adding it to the predictions makes a big difference.
def best_model_bias(timeseries_DF,best_arima_out):
  for col in timeseries_DF.columns:
    X = timeseries_DF[col]
    X = X.astype('float32')
    # difference data
    if col == 'Sale (Dollars)':
        num_to_use=654
    elif col == 'Bottles Sold':
        num_to_use=658
    diff = difference(X, num_to_use)
    # fit model
    model = ARIMA(diff, order=best_arima_out.loc[0,str(col)],trend='n')
    model_fit = model.fit()
    # bias
    residuals=model_fit.resid
    bias = residuals.mean()
    # save model
    model_fit.save(f'{default_directory}/models/model_{col}.pkl')
    np.save(f'{default_directory}/models/model_bias_{col}.npy', [bias])

#get the best models
best_model_bias(teq_by_date,best_RMSE_DF)

#now make the predicitions with the calculated biases
Preds_2017_bias,y_2017_btl_bias,pred_2017_btl_bias,y_2017_sale_bias,pred_2017_sale_bias=\
arima_forecast_validate(teq_by_date,teq_upto_2017,best_RMSE_DF)

#want to get a cleaner plot, going to try just plotting the 2016 prediction data
#find how many values are from 2016
Preds_2017_bias[:-1][Preds_2017_bias[:-1].index<pd.to_datetime('2017-01-01')] #first 88 rows
#now plot just the 2016 data
plt.plot(y_2017_btl_bias[:88],color='blue')
plt.plot(pred_2017_btl_bias[:88], color='red')
plt.title('Stat: Btl, Just 2016')
plt.show()
plt.plot(y_2017_sale_bias[:88],color='blue')
plt.plot(pred_2017_sale_bias[:88], color='red')
plt.title('Stat: Sale, Just 2016')
plt.show()

#its still a little messy, try looking at just one months worth of 2016 data (December)
Preds_2017_bias[:-1][(Preds_2017_bias[:-1].index>pd.to_datetime('2016-11-30')) & (Preds_2017_bias[:-1].index<pd.to_datetime('2017-01-01'))]
list(Preds_2017_bias.index).index(pd.to_datetime('2016-12-01 00:00:00'))#index of first december date is 66
list(Preds_2017_bias.index).index(pd.to_datetime('2016-12-30 00:00:00'))#index of first december date is 87
#now plot just the December 2016 data
plt.plot(y_2017_btl_bias[66:88],color='blue')
plt.plot(pred_2017_btl_bias[66:88], color='red')
plt.title('Stat: Btl, Just December 2016')
plt.show()
plt.plot(y_2017_sale_bias[66:88],color='blue')
plt.plot(pred_2017_sale_bias[66:88], color='red')
plt.title('Stat: Sale, Just December 2016')
plt.show()

####Bottle RMSE, difference smooth data, no bias: 1996.91
####Bottle RMSE, difference smooth data, with bias: 1997.12
####Revenue RMSE, difference smooth data, no bias: 35034.3
####Revenue RMSE, difference smooth data, with bias: 35036.8

################################################################################
#### Modeling With Smoothed Data: Pandas.Rolling Technique #####################
################################################################################

#First look at how the rolling technique works
teq_by_date.index=pd.to_datetime(teq_by_date.index) #make sure the dates are in date format
print(teq_by_date['Bottles Sold'].rolling(window=7).mean().head(10)) #takes the 7 previous values and averages them all together, will be NA if there aren't 7 values before it
print(teq_by_date['Bottles Sold'].rolling(window=7,min_periods=1).mean().head(10)) #since there aren't 7 before some of the values, just uses the average of whatever values come before it instead

#Create a function for smoothing the data and checking the results
def rolling_and_check(df, stat, interval,min_period,center):
    X = df[stat]
    X = X.astype('float32')
    # difference data
    tmp_int = interval
    #stationary
    stationary = df[stat].rolling(window=tmp_int,min_periods=min_period,center=center).mean()
    stationary.index = df.index
    # check if stationary
    result = adfuller(stationary)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
    	print('\t%s: %.3f' % (key, value))
    for i in [2012,2013,2014,2015]:
        fig, ax = plt.subplots(figsize=(20,12))
        stationary[(stationary.index >= f'{i}-01-31') & (stationary.index <f'{i+1}-01-01')].plot()
        plt.title(f'Year: {i}, Interval: {tmp_int}')
        plt.show()
    plot_acf(stationary)
    plt.title(f'ACF Interval: {tmp_int}')
    plt.show()
    plot_pacf(stationary)
    plt.title(f'PACF Interval: {tmp_int}')
    plt.show()
    return stationary

#take a look at the data
print('No rolling: ',adfuller(teq_by_date['Sale (Dollars)']))
print('7 no center: ',rolling_and_check(teq_by_date,'Sale (Dollars)',7,1,False))
print('7 center: ',rolling_and_check(teq_by_date,'Sale (Dollars)',7,1,True))
print('3 no center: ',rolling_and_check(teq_by_date,'Sale (Dollars)',3,1,False))
print('3 center: ',rolling_and_check(teq_by_date,'Sale (Dollars)',3,1,True))
print('30 no center: ',rolling_and_check(teq_by_date,'Sale (Dollars)',30,1,False))
print('30 center: ',rolling_and_check(teq_by_date,'Sale (Dollars)',30,1,True))
print('365 no center: ',rolling_and_check(teq_by_date,'Sale (Dollars)',365,1,False))
print('365 center: ',rolling_and_check(teq_by_date,'Sale (Dollars)',365,1,True))

#create a function for testing all possible window values from 1 to 365 days
def rolling_iteration(df, stat,min_period,center,min_int,max_int):
    X = df[stat]
    X = X.astype('float32')
    tmp_df=pd.DataFrame()
    count=0
    for i in range(min_int,max_int):
        # difference data
        tmp_int = i
        stationary = df[stat].rolling(window=tmp_int,min_periods=min_period,center=center).mean()
        stationary.index = df.index
        # check if stationary
        result = adfuller(stationary)
        tmp_df.loc[count,'interval']=i
        tmp_df.loc[count,'ADF_stat']=result[0]
        tmp_df.loc[count,'p-val']=result[1]
        for key, value in result[4].items():
            tmp_df.loc[count,key]=value
        count=count+1
    return tmp_df

rolling_sale_DF=rolling_iteration(teq_by_date,'Sale (Dollars)',1,False, 1, 366)
print(rolling_sale_DF)
#find the lowest ADF value
print(min(rolling_sale_DF['ADF_stat'])) #-4.275864623150549 <-----the minimum value occurs at interval 1,
# meaning the best value for rolling is when I don't use rolling (if center=false)
#see if there is a next lowest ADF that would still indicate a good interval level
print(min(rolling_sale_DF['ADF_stat'][rolling_sale_DF['ADF_stat']!=-4.275864623150549])) #-4.155118511749357
print((rolling_sale_DF['ADF_stat']==min(rolling_sale_DF['ADF_stat'][rolling_sale_DF['ADF_stat']!=-4.275864623150549])).argmax()) #1
rolling_sale_DF.loc[1,:] #based on this output, 2 is still a good interval to use with this rolling (1 is the index, but 2 is the interval)

rolling_sale_DF_cnt=rolling_iteration(teq_by_date,'Sale (Dollars)',1,True, 1, 366)
print(rolling_sale_DF_cnt)
#find the lowest ADF value
print(min(rolling_sale_DF_cnt['ADF_stat'])) #-4.275864623150549 <-----the minimum value occurs at interbal 1,
# meaning the best value for rolling is when I don't use rolling (if center=True)
print(min(rolling_sale_DF_cnt['ADF_stat'][rolling_sale_DF_cnt['ADF_stat']!=-4.275864623150549])) #-4.155118511749357
print((rolling_sale_DF_cnt['ADF_stat']==min(rolling_sale_DF_cnt['ADF_stat'][rolling_sale_DF_cnt['ADF_stat']!=-4.275864623150549])).argmax()) #1
rolling_sale_DF_cnt.loc[1,:] #based on this output, 2 is still a good interval to use with this rolling (1 is the index, but 2 is the interval)

rolling_btl_DF=rolling_iteration(teq_by_date,'Bottles Sold',1,False, 1, 366)
print(rolling_btl_DF)
#find the lowest ADF value
print(min(rolling_btl_DF['ADF_stat'])) #-4.876213540202406
#get the index for lowest ADV value
print((rolling_btl_DF['ADF_stat']==min(rolling_btl_DF['ADF_stat'])).argmax()) #65
rolling_btl_DF.loc[65,:] #based on this output, 66 is the best interval (65 is the index, but 66 is the interval)

rolling_btl_DF_cnt=rolling_iteration(teq_by_date,'Bottles Sold',1,True, 1, 366)
print(rolling_btl_DF_cnt)
#find the lowest ADF value
print(min(rolling_btl_DF_cnt['ADF_stat'])) #-4.8027328225636285
print((rolling_btl_DF_cnt['ADF_stat']==min(rolling_btl_DF_cnt['ADF_stat'])).argmax()) #63
rolling_btl_DF_cnt.loc[63,:] #based on this output, 64 is the best interval (63 is the index, but 64 is the interval)

#now test applying the rolling function and then trying to reverse the rolling
rolling_test_DF=teq_by_date.copy(deep=True)
#creating rolling versions of the data
rolling_test_DF['Btl_Roll']=rolling_test_DF['Bottles Sold'].rolling(window=66,min_periods=1,center=False).mean()
rolling_test_DF

#now see if I can reverse rolling the data in order to make predictions later
rolling_test_DF['ref_int']=range(1,len(rolling_test_DF)+1)
rolling_test_DF['prev_rows']=rolling_test_DF['Bottles Sold'].rolling(window=66,min_periods=1,center=False).count()
for i in rolling_test_DF.index:
    rolling_test_DF.loc[i,'rolling_sums']=rolling_test_DF['Bottles Sold'][int(rolling_test_DF.loc[i,'ref_int']-rolling_test_DF.loc[i,'prev_rows']):int(rolling_test_DF.loc[i,'ref_int'])].sum()
    rolling_test_DF.loc[i,'rolling_parts']=rolling_test_DF['Bottles Sold'][int(rolling_test_DF.loc[i,'ref_int']-rolling_test_DF.loc[i,'prev_rows']):int(rolling_test_DF.loc[i,'ref_int']-1)].sum()
    rolling_test_DF.loc[i,'full_revert']=rolling_test_DF.loc[i,'rolling_sums']-rolling_test_DF.loc[i,'rolling_parts']
rolling_test_DF
#Bottles sold and full_revert are equal, so the logic for reverting the data should work.

#Create a function that takes in the data, spits out a dataframe that contains the original data and the rolling data.
def create_rolling(dataframe, stat, window, min_periods=1, center=False):
    tmp_df=pd.DataFrame()
    tmp_df[stat]=dataframe[stat]
    tmp_df['rolling']=tmp_df[stat].rolling(window=window,min_periods=min_periods,center=center).mean()
    return tmp_df

#now define a function that will be able to revert functions back to their non-rolling values
def revert_back(DF,stat,tmp_time,window,min_periods=1,center=False):
    tmp_df=pd.DataFrame()
    tmp_df[stat]=DF[stat]
    tmp_df['rolling']=DF['rolling']
    tmp_df['ref_int']=range(1,len(DF)+1)
    tmp_df['prev_rows']=tmp_df[stat].rolling(window=window,min_periods=min_periods,center=center).count()
    tmp_sum=tmp_df.loc[tmp_time,'rolling']*(tmp_df.loc[tmp_time,'prev_rows']+1)
    part_sum=tmp_df[stat][int(tmp_df.loc[tmp_time,'ref_int']-(tmp_df.loc[tmp_time,'prev_rows']+1)):int(tmp_df.loc[tmp_time,'ref_int']-1)].sum()
    tmp_df.loc[tmp_time,stat]=tmp_sum-part_sum
    tmp_df=tmp_df.drop(['ref_int','prev_rows'],axis=1)
    return tmp_df

#Define the function for performing the ARIMA modeling using rolling
def evaluate_arima_rolling(DF, stat, arima_order, window, min_periods=1, center=False):
    # prepare training dataset
    train_size = int(len(DF) * 0.70)
    train, test = DF[0:train_size], DF[train_size:]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        train=create_rolling(train, stat, window, min_periods, center)
        model   = ARIMA(train['rolling'], order=arima_order,trend='n')
        model_fit = model.fit()
        yhat = model_fit.forecast()
        tmp_time=test.index[t]
        train.loc[tmp_time,'rolling']=float(yhat)
        train=revert_back(train,stat,tmp_time,window,min_periods,center)
        yhat = train.loc[tmp_time,stat]
        predictions.append(yhat)
        train.loc[tmp_time,stat]=test.loc[tmp_time,stat]
	# calculate out of sample error
    mse = mean_squared_error(test[stat], predictions)
    rmse = sqrt(mse)
    return rmse

#now test the evaluate rolling function
evaluate_arima_rolling(teq_by_date, 'Bottles Sold', (0,0,0), 66, min_periods=1, center=False)
#output is 289154.7123920531, waaaaay too high, need to evaluate what is going on

#pull the code out of the function and manually run it to try to break down why the rmse is so high
DF=teq_by_date
stat='Bottles Sold'
arima_order=(0,0,0)
window=66
min_periods=1
center=False

# prepare training dataset
train_size = int(len(DF) * 0.70)
train, test = DF[0:train_size], DF[train_size:]
# make predictions
predictions = list()
for t in range(len(test)):
        train=create_rolling(train, stat, window, min_periods, center)
        model   = ARIMA(train['rolling'], order=arima_order,trend='n')
        model_fit = model.fit()
        yhat = model_fit.forecast()
        tmp_time=test.index[t]
        train.loc[tmp_time,'rolling']=float(yhat)
        train=revert_back(train,stat,tmp_time,window,min_periods,center)
        yhat = train.loc[tmp_time,stat]
        predictions.append(yhat)
        train.loc[tmp_time,stat]=test.loc[tmp_time,stat]
	# calculate out of sample error
mse = mean_squared_error(test[stat], predictions)
rmse = sqrt(mse)

predictions #the predictions are way way off, need to go back to the create_rolling and revert_back functions to see what is going on

#test the create rolling function
test_rolling_DF=create_rolling(teq_by_date, 'Bottles Sold',66,1,False)
test_rolling_DF
#this output looks reasonable

#make up a number to use for the next iteration in the test_rolling function above so I can then test the revert function
test_rolling_DF.loc['2016-08-27','rolling']=4345.565656
print(test_rolling_DF)
test_revert_DF=revert_back(test_rolling_DF,'Bottles Sold','2016-08-27',66,min_periods=1,center=False)
test_revert_DF
#this appears to be the problem, when trying to revert the predicted rolling value back to a predicted normal value,
#the predicted normal value seems to be way too high. Need to break down the revert_back function now

#break down the revert_back function now
DF=test_rolling_DF
stat='Bottles Sold'
tmp_time='2016-08-27'
window=66
min_periods=1
center=False

tmp_df=pd.DataFrame()
tmp_df[stat]=DF[stat]
tmp_df['rolling']=DF['rolling']
tmp_df['ref_int']=range(1,len(DF)+1)
tmp_df['prev_rows']=tmp_df[stat].rolling(window=window,min_periods=min_periods,center=center).count()
tmp_sum=tmp_df.loc[tmp_time,'rolling']*(tmp_df.loc[tmp_time,'prev_rows']+1)
part_sum=tmp_df[stat][int(tmp_df.loc[tmp_time,'ref_int']-(tmp_df.loc[tmp_time,'prev_rows']+1)):int(tmp_df.loc[tmp_time,'ref_int']-1)].sum()
tmp_df.loc[tmp_time,stat]=tmp_sum-part_sum
tmp_df=tmp_df.drop(['ref_int','prev_rows'],axis=1)
tmp_df
#I think when the window is too big, it draws in too many dips and highs in the data,
# creating rolling values that are very innacurate, and making it very difficult to try to revert the predicted rolling value back to a basic predicted non-rolling value.
# It may be that the rolling technique can only be used with smaller windows.

#found this potential solution online at this link, giving it a shot:
#https://stackoverflow.com/questions/52456267/how-to-do-a-reverse-moving-average-in-pandas-rolling-mean-operation-on-pr
def cumsum_shift(s, shift = 1, init_values = [0]):
    s_cumsum = pd.Series(np.zeros(len(s)))
    for i in range(shift):
        s_cumsum.iloc[i] = init_values[i]
    for i in range(shift,len(s)):
        s_cumsum.iloc[i] = s_cumsum.iloc[i-shift] + s.iloc[i]
    return s_cumsum

s_diffed = 66 * tmp_df['rolling'].diff()
tmp_df['y_unrolled'] = list(cumsum_shift(s=s_diffed, shift = 66, init_values= tmp_df['Bottles Sold'].values[:66]))
tmp_df['y_unrolled_pred'] = list(cumsum_shift(s=s_diffed, shift = 66, init_values= tmp_df['rolling'].values[:66]))
#the results from this were very similar to what I was finding, big windows are still an issue.
#Going to try a couple more things.

#seeing if anything can be done with the diff between each value within the time window to make a large window work
#tmp_df['diff']=tmp_df[stat].diff()
#part_sum=tmp_df['rolling'][int(tmp_df.loc[tmp_time,'ref_int']-(tmp_df.loc[tmp_time,'prev_rows']+1)):int(tmp_df.loc[tmp_time,'ref_int']-1)].sum()
#med_diff=tmp_df['rolling'][int(tmp_df.loc[tmp_time,'ref_int']-(tmp_df.loc[tmp_time,'prev_rows']+1)):int(tmp_df.loc[tmp_time,'ref_int']-1)].diff().median()
#tmp_diff_sum=(tmp_df[stat][int(tmp_df.loc[tmp_time,'ref_int']-(tmp_df.loc[tmp_time,'prev_rows']+1)):int(tmp_df.loc[tmp_time,'ref_int']-1)]+(med_diff/(tmp_df.loc[tmp_time,'prev_rows']-1))).sum()
#tmp_diff_sum=(tmp_df['rolling'][int(tmp_df.loc[tmp_time,'ref_int']-(tmp_df.loc[tmp_time,'prev_rows']+1)):int(tmp_df.loc[tmp_time,'ref_int']-1)]+(med_diff/(tmp_df.loc[tmp_time,'prev_rows']-1))).sum()

#see if when the center=True option is used if the result is any better:
evaluate_arima_rolling(teq_by_date, 'Bottles Sold', (0,0,0), 64, min_periods=1, center=True)
#output is 145581.37604958905, so the result is better, but still way too high

#unfortunately it looks like using a large window is going to be an issue, going back to try to find another suitable window to use
rolling_btl_DF=rolling_iteration(teq_by_date,'Bottles Sold',1,False, 1, 366)
#view the top lowest ADF values when center=False
rolling_btl_DF.sort_values('ADF_stat').head(10) #a window of 34 should be checked

#check when center=True
rolling_btl_DF_cnt=rolling_iteration(teq_by_date,'Bottles Sold',1,True, 1, 366)
#view the top lowest ADF values when center=True
rolling_btl_DF_cnt.sort_values('ADF_stat').head(10) #34 may work here too

#evaluate how the model looks now when the window is 34, using both True and False for the center
print(evaluate_arima_rolling(teq_by_date, 'Bottles Sold', (0,0,0), 34, min_periods=1, center=False)) #150003.87296913174
evaluate_arima_rolling(teq_by_date, 'Bottles Sold', (0,0,0), 34, min_periods=1, center=True) #79383.0807431509
#These are still too high unfortunately

#looking again
print(rolling_btl_DF[rolling_btl_DF['interval']<=25].sort_values('ADF_stat').head(10))
print(rolling_btl_DF_cnt[rolling_btl_DF_cnt['interval']<=25].sort_values('ADF_stat').head(10)) #25,23,2,and 8 should be looked at

#now testing
print(evaluate_arima_rolling(teq_by_date, 'Bottles Sold', (0,0,0), 25, min_periods=1, center=False))
print(evaluate_arima_rolling(teq_by_date, 'Bottles Sold', (0,0,0), 25, min_periods=1, center=True))

print(evaluate_arima_rolling(teq_by_date, 'Bottles Sold', (0,0,0), 23, min_periods=1, center=False))
print(evaluate_arima_rolling(teq_by_date, 'Bottles Sold', (0,0,0), 23, min_periods=1, center=True))

print(evaluate_arima_rolling(teq_by_date, 'Bottles Sold', (0,0,0), 8, min_periods=1, center=False))
print(evaluate_arima_rolling(teq_by_date, 'Bottles Sold', (0,0,0), 8, min_periods=1, center=True))

print(evaluate_arima_rolling(teq_by_date, 'Bottles Sold', (1,1,1), 8, min_periods=1, center=False))
print(evaluate_arima_rolling(teq_by_date, 'Bottles Sold', (1,1,1), 8, min_periods=1, center=True))

print(evaluate_arima_rolling(teq_by_date, 'Bottles Sold', (0,0,0), 2, min_periods=1, center=False))
print(evaluate_arima_rolling(teq_by_date, 'Bottles Sold', (0,0,0), 2, min_periods=1, center=True))

print(evaluate_arima_rolling(teq_by_date, 'Bottles Sold', (1,1,1), 2, min_periods=1, center=False))
evaluate_arima_rolling(teq_by_date, 'Bottles Sold', (1,1,1), 2, min_periods=1, center=True)

#see if any other values between 2 and 10 produce results that could be better than a window of 2
for i in range(2,11):
    print(i, ' not-center: ', evaluate_arima_rolling(teq_by_date, 'Bottles Sold', (1,1,1), i, min_periods=1, center=False))
    print(i, ' center: ', evaluate_arima_rolling(teq_by_date, 'Bottles Sold', (1,1,1), i, min_periods=1, center=True))
#based on these results, going to use a window of 2, which means it will only be averaging two values at a time, but it will still be some smoothing

#now create the function that runs the models based on the inputted DF, p, d, and q values,
#then creates a DF storing all of the data which can be anaylzed to determine the best p,d,q values to use with the data
def evaluate_rolling(dataframe, p_values, d_values, q_values):
  file_num=0
  full_DF=pd.DataFrame()
  for p in p_values:
    for d in d_values:
      for q in q_values:
        order = (p,d,q)
        tmp_DF=pd.DataFrame()
        for stat in dataframe.columns:
          try:
            mse = evaluate_arima_rolling(dataframe, stat, order, 2, min_periods=1, center=False)
            tmp_DF.loc[0,'p']=p
            tmp_DF.loc[0,'d']=d
            tmp_DF.loc[0,'q']=q
            tmp_DF.loc[0,stat]=mse
          except:
            tmp_DF.loc[0,'p']=p
            tmp_DF.loc[0,'d']=d
            tmp_DF.loc[0,'q']=q
            tmp_DF.loc[0,stat]=0
            continue
        tmp_DF.to_csv(f'{default_directory}/datafiles/backups/backups_roll/tmp_DF_{file_num}_roll.csv',header=True,index=False)
        file_num=file_num+1
        temp_comb_DF=[full_DF,tmp_DF]
        full_DF=pd.concat(temp_comb_DF)
  return full_DF

'''#Make sure function works
p_values = range(0, 1)
d_values = range(0, 1)
q_values = range(0, 3)
arima_test_check_rolling=evaluate_rolling(teq_by_date, p_values, d_values, q_values)
arima_test_check_rolling
'''

'''Once again, this took 10+ hours to run, the output will be manually loaded below.
#now collect the data
p_values = range(0, 9)
d_values = range(0, 3)
q_values = range(0, 9)
arima_test_check_rolling=evaluate_rolling(teq_by_date, p_values, d_values, q_values)
arima_test_check_rolling
'''

'''This is how the individual p,d,q files were imported and compiled. Once again, the output will ultimately be imported below.
RMSE_tests_DF_roll=pd.read_csv(f'{default_directory}/datafiles/backups/backups_roll/tmp_DF_0_roll.csv')
for i in range(1,243):
  tmp_RMSE_DF=pd.read_csv(f'{default_directory}/datafiles/backups/backups_roll/tmp_DF_{i}_roll.csv')
  temp_comb_DF=[RMSE_tests_DF_roll,tmp_RMSE_DF]
  RMSE_tests_DF_roll=pd.concat(temp_comb_DF)
RMSE_tests_DF_roll=RMSE_tests_DF_roll.reset_index(drop=True)
RMSE_tests_DF_roll
'''

#RMSE_tests_DF_roll.to_csv('RMSE_tests_DF_roll.csv',index=False, header=True)
RMSE_tests_DF_roll=pd.read_csv('RMSE_tests_DF_roll.csv')
RMSE_tests_DF_roll

print(min(RMSE_tests_DF_roll['Bottles Sold'][RMSE_tests_DF_roll['Bottles Sold']!=0])) #1287.5151310587955
print(min(RMSE_tests_DF_roll['Sale (Dollars)'][RMSE_tests_DF_roll['Sale (Dollars)']!=0])) #22356.466569742424

best_RMSE_DF_roll=best_arima(RMSE_tests_DF_roll)
best_RMSE_DF_roll

#function for building the best model and saving/exporting it
def best_model_roll(timeseries_DF,best_arima_out,window, min_periods=1, center=False,use_bias='no'):
  for col in timeseries_DF.columns:
    #rolling/smooth the data
    rolled_data=create_rolling(timeseries_DF,col,window=window,min_periods=min_periods,center=center)
    # fit model
    model = ARIMA(rolled_data['rolling'], order=best_arima_out.loc[0,str(col)],trend='n')
    model_fit = model.fit()
    # bias
    if use_bias == 'yes':
        residuals=model_fit.resid
        bias = residuals.mean()
        np.save(f'{default_directory}/models/model_bias_{col}_roll.npy', [bias])
    # save model
    model_fit.save(f'{default_directory}/models/model_{col}_roll.pkl')

best_model_roll(teq_by_date,best_RMSE_DF_roll,2,use_bias='yes')
best_model_roll(teq_by_date,best_RMSE_DF_roll,window=2)

def arima_forecast_validate_roll(timeseries_DF,test_DF,best_arima_out,window, min_periods=1, center=False,use_bias='no'):
  pred_val_DF=pd.DataFrame()
  pred_val_DF.index=test_DF.index
  y_pred_dict={}
  for col in timeseries_DF.columns:
      rolled_data=create_rolling(timeseries_DF,col,window,min_periods,center)
      # load model
      model_fit = ARIMAResults.load(f'{default_directory}/models/model_{col}_roll.pkl')
      if use_bias=='yes':
          bias = np.load(f'{default_directory}/models/model_bias_{col}_roll.npy')
      else:
          bias=np.array([0])
      # make first prediction
      predictions = list()
      yhat = float(model_fit.forecast())
      tmp_time=test_DF.index[0]
      rolled_data.loc[tmp_time,'rolling']=float(yhat)
      rolled_data=revert_back(rolled_data,col,tmp_time,window,min_periods,center)
      yhat = bias+rolled_data.loc[tmp_time,col]
      predictions.append(yhat)
      rolled_data.loc[tmp_time,col]=test_DF.loc[tmp_time,col]
      pred_val_DF.loc[pred_val_DF.index[0],f'{str(col)}_Pred']="{:.2f}".format(yhat[0])
      pred_val_DF.loc[pred_val_DF.index[0],f'{str(col)}_Exp']="{:.2f}".format(test_DF[col][0])
      print('Stat: ', col)
      print('>Predicted=%.3f, Expected=%3.f' % (yhat, test_DF[col][0]))
      #make additional predictions
      for i in range(1, len(test_DF[col])):
        rolled_data=create_rolling(rolled_data,col,window,min_periods,center)
        model = ARIMA(rolled_data['rolling'], order=best_arima_out.loc[0,str(col)],trend='n')
        model.initialize_approximate_diffuse()
        model_fit = model.fit()
        yhat = float(model_fit.forecast())
        tmp_time=test_DF.index[i]
        rolled_data.loc[tmp_time,'rolling']=float(yhat)
        rolled_data=revert_back(rolled_data,col,tmp_time,window,min_periods,center)
        yhat = bias+rolled_data.loc[tmp_time,col]
        predictions.append(yhat)
        rolled_data.loc[tmp_time,col]=test_DF.loc[tmp_time,col]
        obs = test_DF[col][i]
        pred_val_DF.loc[pred_val_DF.index[i],f'{str(col)}_Pred']="{:.2f}".format(yhat[0])
        pred_val_DF.loc[pred_val_DF.index[i],f'{str(col)}_Exp']="{:.2f}".format(obs)
        print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
      # report performance
      mse = mean_squared_error(test_DF[col], predictions)
      rmse = sqrt(mse)
      print('RMSE: %.3f' % rmse)
      pred_val_DF.loc['RMSE',f'{str(col)}_Pred']=rmse
      pred_val_DF.loc['RMSE',f'{str(col)}_Exp']=rmse
      plt.plot(np.array(test_DF[col]),color='blue')
      plt.plot(predictions, color='red')
      plt.title(f'Stat: {col}')
      plt.show()
      y_pred_dict[f'y_{col}']=test_DF[col]
      y_pred_dict[f'predictions_{col}']=predictions
  return pred_val_DF, y_pred_dict['y_Bottles Sold'], y_pred_dict['predictions_Bottles Sold'],y_pred_dict['y_Sale (Dollars)'], y_pred_dict['predictions_Sale (Dollars)']

Preds_2017_roll_bias,exp_2017_btl_roll_bias,pred_2017_btl_roll_bias,exp_2017_sale_roll_bias,pred_2017_sale_roll_bias=\
arima_forecast_validate_roll(teq_by_date,teq_upto_2017,best_RMSE_DF_roll,window=2, min_periods=1, center=False,use_bias='yes')

#want to get a cleaner plot, going to try just plotting the 2016 prediction data
#find how many values are from 2016
print(Preds_2017_roll_bias[:-1][Preds_2017_roll_bias[:-1].index<pd.to_datetime('2017-01-01')]) #first 88 rows
#now plot just the 2016 data
plt.plot(np.array(exp_2017_btl_roll_bias[:88]),color='blue')
plt.plot(pred_2017_btl_roll_bias[:88], color='red')
plt.title('Stat: Btl, Just 2016')
plt.show()
plt.plot(np.array(exp_2017_sale_roll_bias[:88]),color='blue')
plt.plot(pred_2017_sale_roll_bias[:88], color='red')
plt.title('Stat: Sale, Just 2016')
plt.show()

#its still a little messy, try looking at just one months worth of 2016 data (December)
print(Preds_2017_roll_bias[:-1][(Preds_2017_roll_bias[:-1].index>pd.to_datetime('2016-11-30')) & (Preds_2017_roll_bias[:-1].index<pd.to_datetime('2017-01-01'))])
print(list(Preds_2017_roll_bias.index).index(pd.to_datetime('2016-12-01 00:00:00')))#index of first december date is 66
print(list(Preds_2017_roll_bias.index).index(pd.to_datetime('2016-12-30 00:00:00')))#index of first december date is 87
#now plot just the December 2016 data
plt.plot(np.array(exp_2017_btl_roll_bias[66:88]),color='blue')
plt.plot(pred_2017_btl_roll_bias[66:88], color='red')
plt.title('Stat: Btl, Just December 2016')
plt.show()
plt.plot(np.array(exp_2017_sale_roll_bias[66:88]),color='blue')
plt.plot(pred_2017_sale_roll_bias[66:88], color='red')
plt.title('Stat: Sale, Just December 2016')
plt.show()

#run again without bias
Preds_2017_roll,exp_2017_btl_roll,pred_2017_btl_roll,exp_2017_sale_roll,pred_2017_sale_roll=\
arima_forecast_validate_roll(teq_by_date,teq_upto_2017,best_RMSE_DF_roll,window=2, min_periods=1, center=False,use_bias='no')

#want to get a cleaner plot, going to try just plotting the 2016 prediction data
#find how many values are from 2016
print(Preds_2017_roll[:-1][Preds_2017_roll[:-1].index<pd.to_datetime('2017-01-01')]) #first 88 rows
#now plot just the 2016 data
plt.plot(np.array(exp_2017_btl_roll[:88]),color='blue')
plt.plot(pred_2017_btl_roll[:88], color='red')
plt.title('Stat: Btl, Just 2016')
plt.show()
plt.plot(np.array(exp_2017_sale_roll[:88]),color='blue')
plt.plot(pred_2017_sale_roll[:88], color='red')
plt.title('Stat: Sale, Just 2016')
plt.show()

#its still a little messy, try looking at just one months worth of 2016 data (December)
print(Preds_2017_roll[:-1][(Preds_2017_roll[:-1].index>pd.to_datetime('2016-11-30')) & (Preds_2017_roll[:-1].index<pd.to_datetime('2017-01-01'))])
print(list(Preds_2017_roll.index).index(pd.to_datetime('2016-12-01 00:00:00')))#index of first december date is 66
print(list(Preds_2017_roll.index).index(pd.to_datetime('2016-12-30 00:00:00')))#index of first december date is 87
#now plot just the December 2016 data
plt.plot(np.array(exp_2017_btl_roll[66:88]),color='blue')
plt.plot(pred_2017_btl_roll[66:88], color='red')
plt.title('Stat: Btl, Just December 2016')
plt.show()
plt.plot(np.array(exp_2017_sale_roll[66:88]),color='blue')
plt.plot(pred_2017_sale_roll[66:88], color='red')
plt.title('Stat: Sale, Just December 2016')
plt.show()

####preds 2017, roll, no bias, bottle: 1495.74 \
####preds 2017, roll, no bias, sale: 25967.1 \
####preds 2017, roll, bias, bottle: 1495.47 \
####preds 2017, roll, bias, sale: 25961.6

################################################################################
#### Modeling With Smoothed Data: Savgul Smoothing Method ######################
################################################################################

#Define the functions for using the savgul smoothing
def savgul_data(data, window_size,poly):
    return pd.Series(savgol_filter(data, window_size,poly))

def inverse_savgul_data(smoothed_data, original_data,mode='mean'):
    if mode=='mean':
        return smoothed_data + (original_data - smoothed_data).mean()
    elif mode == 'median':
        return smoothed_data + (original_data - smoothed_data).median()

#Do some testing of the savgul technique.
teq_smooth_test=teq_by_date.copy(deep=True)
savgol_data=savgul_data(teq_smooth_test['Bottles Sold'],30,3)
teq_smooth_test['savgol_data']=list(savgol_data)
teq_smooth_test['savgol_rev']=inverse_savgul_data(teq_smooth_test['savgol_data'],teq_smooth_test['Bottles Sold'])
teq_smooth_test.index
teq_smooth_test.loc['2016-08-27','savgol_data']=3472.34
teq_smooth_test['savgol_rev']=inverse_savgul_data(teq_smooth_test['savgol_data'],teq_smooth_test['Bottles Sold'])
teq_smooth_test

#Create a function that performs savgul on the data and then checks the ADF
def savgul_and_check(df, stat, win, poly):
    X = df[stat]
    X = X.astype('float32')
    # difference data
    tmp_win =win
    #stationary
    stationary = savgul_data(X,tmp_win,poly)
    stationary.index = df.index
    # check if stationary
    result = adfuller(stationary)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
    	print('\t%s: %.3f' % (key, value))
    for i in [2012,2013,2014,2015]:
        fig, ax = plt.subplots(figsize=(20,12))
        stationary[(stationary.index >= f'{i}-01-31') & (stationary.index <f'{i+1}-01-01')].plot()
        plt.title(f'Year: {i}, Window: {tmp_win}')
        plt.show()
    plot_acf(stationary)
    plt.title(f'ACF Interval: {tmp_win}')
    plt.show()
    plot_pacf(stationary)
    plt.title(f'PACF Interval: {tmp_win}')
    plt.show()
    return stationary

savgul_and_check(teq_by_date, 'Bottles Sold', 3, 2)
savgul_and_check(teq_by_date, 'Bottles Sold', 30, 2)

#Create a function for testing all possible values from 1 to 365 days, and 1 to 10 for polynomial
def savgul_iteration(df, stat,min_win,max_win,poly):
    X = df[stat]
    X = X.astype('float32')
    tmp_df=pd.DataFrame()
    count=0
    for p in range(1,poly+1):
        for i in range(min_win,max_win):
            if p>=i:
                continue
            else:
                tmp_win = i
                stationary = savgul_data(X,tmp_win,p)
                stationary.index = df.index
                # check if stationary
                result = adfuller(stationary)
                tmp_df.loc[count,'window']=i
                tmp_df.loc[count,'poly']=poly
                tmp_df.loc[count,'ADF_stat']=result[0]
                tmp_df.loc[count,'p-val']=result[1]
                for key, value in result[4].items():
                    tmp_df.loc[count,key]=value
                count=count+1
    return tmp_df

#try to pinpoint a good window to use for Bottles Sold
savgul_btl_DF=savgul_iteration(teq_by_date,'Bottles Sold',1, 366,5)
#view the top lowest ADF values
savgul_btl_DF.sort_values('ADF_stat').head(10) #a window of 256, poly of 5 should be checked

#testing
teq_smooth_test=teq_by_date.copy(deep=True)
savgol_data=savgul_data(teq_smooth_test['Bottles Sold'],256,5)
teq_smooth_test['savgol_data']=list(savgol_data)
teq_smooth_test['savgol_rev_mean']=inverse_savgul_data(teq_smooth_test['savgol_data'],teq_smooth_test['Bottles Sold'],'mean')
teq_smooth_test['savgol_rev_median']=inverse_savgul_data(teq_smooth_test['savgol_data'],teq_smooth_test['Bottles Sold'],'median')
teq_smooth_test.loc['2016-08-27','savgol_data']=3472.34
teq_smooth_test['savgol_rev_mean']=inverse_savgul_data(teq_smooth_test['savgol_data'],teq_smooth_test['Bottles Sold'],'mean')
teq_smooth_test['savgol_rev_median']=inverse_savgul_data(teq_smooth_test['savgol_data'],teq_smooth_test['Bottles Sold'],'median')
#large windows may work, going to look at some smaller windows too to be safe

#try to pinpoint a good window to use for Bottles Sold, checking smaller windows too based on what I saw before with other smoothing functions
savgul_btl_DF=savgul_iteration(teq_by_date,'Bottles Sold',1, 30,5)
#view the top lowest ADF values
savgul_btl_DF.sort_values('ADF_stat').head(10) #a window of 23, poly of 5 should be checked

#testing the window of 30/poly of 5
teq_smooth_test=teq_by_date.copy(deep=True)
savgol_data=savgul_data(teq_smooth_test['Bottles Sold'],23,5)
teq_smooth_test['savgol_data']=list(savgol_data)
teq_smooth_test['savgol_rev_mean']=inverse_savgul_data(teq_smooth_test['savgol_data'],teq_smooth_test['Bottles Sold'],'mean')
teq_smooth_test['savgol_rev_median']=inverse_savgul_data(teq_smooth_test['savgol_data'],teq_smooth_test['Bottles Sold'],'median')
teq_smooth_test.loc['2016-08-27','savgol_data']=3472.34
teq_smooth_test['savgol_rev_mean']=inverse_savgul_data(teq_smooth_test['savgol_data'],teq_smooth_test['Bottles Sold'],'mean')
teq_smooth_test['savgol_rev_median']=inverse_savgul_data(teq_smooth_test['savgol_data'],teq_smooth_test['Bottles Sold'],'median')
teq_smooth_test
#windows of 30 may work too

#now check sale dollars
savgul_sale_DF=savgul_iteration(teq_by_date,'Sale (Dollars)',1, 366,5)
#view the top lowest ADF values
savgul_sale_DF.sort_values('ADF_stat').head(10) #a window of 208 should be checked

#testing a window of 208
teq_smooth_test=teq_by_date.copy(deep=True)
savgol_data=savgul_data(teq_smooth_test['Sale (Dollars)'],208,5)
teq_smooth_test['savgol_data']=list(savgol_data)
teq_smooth_test['savgol_rev_mean']=inverse_savgul_data(teq_smooth_test['savgol_data'],teq_smooth_test['Sale (Dollars)'],'mean')
teq_smooth_test['savgol_rev_median']=inverse_savgul_data(teq_smooth_test['savgol_data'],teq_smooth_test['Sale (Dollars)'],'median')
teq_smooth_test.loc['2016-08-27','savgol_data']=37472.34
teq_smooth_test['savgol_rev_mean']=inverse_savgul_data(teq_smooth_test['savgol_data'],teq_smooth_test['Sale (Dollars)'],'mean')
teq_smooth_test['savgol_rev_median']=inverse_savgul_data(teq_smooth_test['savgol_data'],teq_smooth_test['Sale (Dollars)'],'median')
teq_smooth_test

#looking at smaller windows too
savgul_sale_DF=savgul_iteration(teq_by_date,'Sale (Dollars)',1, 30,5)
#view the top lowest ADF values
savgul_sale_DF.sort_values('ADF_stat').head(10) #a window of 23 should be checked

#testing a window of 23
teq_smooth_test=teq_by_date.copy(deep=True)
savgol_data=savgul_data(teq_smooth_test['Sale (Dollars)'],23,5)
teq_smooth_test['savgol_data']=list(savgol_data)
teq_smooth_test['savgol_rev_mean']=inverse_savgul_data(teq_smooth_test['savgol_data'],teq_smooth_test['Sale (Dollars)'],'mean')
teq_smooth_test['savgol_rev_median']=inverse_savgul_data(teq_smooth_test['savgol_data'],teq_smooth_test['Sale (Dollars)'],'median')
teq_smooth_test.loc['2016-08-27','savgol_data']=37472.34
teq_smooth_test['savgol_rev_mean']=inverse_savgul_data(teq_smooth_test['savgol_data'],teq_smooth_test['Sale (Dollars)'],'mean')
teq_smooth_test['savgol_rev_median']=inverse_savgul_data(teq_smooth_test['savgol_data'],teq_smooth_test['Sale (Dollars)'],'median')
teq_smooth_test

#now create the functions for testing the best models
#Define the function for performing the ARIMA modeling with savgul smoothing
def evaluate_arima_savgul(DF,stat,arima_order,window,poly,mode):
    # prepare training dataset
    tmp_DF=pd.DataFrame()
    tmp_DF.index=DF.index
    tmp_DF[stat]=DF[stat]
    train_size = int(len(tmp_DF) * 0.70)
    train, test = tmp_DF[0:train_size], tmp_DF[train_size:]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        sav_data = savgul_data(train[stat], window, poly)
        train['sav_data']=list(sav_data)
        model   = ARIMA(train['sav_data'], order=arima_order,trend='n')
        model_fit = model.fit()
        yhat = model_fit.forecast()
        tmp_time=test.index[t]
        train.loc[tmp_time,'sav_data']=float(yhat)
        train['inverse']=inverse_savgul_data(train['sav_data'], train[stat],mode)
        yhat = train.loc[tmp_time,'inverse']
        predictions.append(yhat)
        train.loc[tmp_time,stat]=test.loc[tmp_time,stat]
        train=train.drop('inverse',axis=1)
	# calculate out of sample error
    mse = mean_squared_error(test, predictions)
    rmse = sqrt(mse)
    return rmse

#now test the function to make sure it works
print(evaluate_arima_savgul(teq_by_date,'Bottles Sold',(0,0,0),256,5,'mean')) #4576.657185161676
print(evaluate_arima_savgul(teq_by_date,'Bottles Sold',(0,0,0),256,5,'median')) #4527.8701348561435

print(evaluate_arima_savgul(teq_by_date,'Bottles Sold',(1,1,1),256,5,'mean')) #1420.5116183613525
print(evaluate_arima_savgul(teq_by_date,'Bottles Sold',(1,1,1),256,5,'median')) #1420.272884317896

print(evaluate_arima_savgul(teq_by_date,'Bottles Sold',(0,0,0),23,5,'mean')) #4575.051401336516
print(evaluate_arima_savgul(teq_by_date,'Bottles Sold',(0,0,0),23,5,'median')) #4474.285803550747

print(evaluate_arima_savgul(teq_by_date,'Bottles Sold',(1,1,1),23,5,'mean')) #1935.8130183933222
print(evaluate_arima_savgul(teq_by_date,'Bottles Sold',(1,1,1),23,5,'median')) #1939.1250870302226
#going to go with 256 and median for Bottles Sold

print(evaluate_arima_savgul(teq_by_date,'Sale (Dollars)',(0,0,0),208,5,'mean')) #78642.61954784328
print(evaluate_arima_savgul(teq_by_date,'Sale (Dollars)',(0,0,0),208,5,'median')) #77571.3638191473

print(evaluate_arima_savgul(teq_by_date,'Sale (Dollars)',(1,1,1),208,5,'mean')) #24758.247499337846
print(evaluate_arima_savgul(teq_by_date,'Sale (Dollars)',(1,1,1),208,5,'median')) #24754.66772580419

print(evaluate_arima_savgul(teq_by_date,'Sale (Dollars)',(0,0,0),23,5,'mean')) #78634.78865714741
print(evaluate_arima_savgul(teq_by_date,'Sale (Dollars)',(0,0,0),23,5,'median')) #76707.61821852658

print(evaluate_arima_savgul(teq_by_date,'Sale (Dollars)',(1,1,1),23,5,'mean')) #33050.02395955885
evaluate_arima_savgul(teq_by_date,'Sale (Dollars)',(1,1,1),23,5,'median') #33116.41031406413
#going with 208 and median for Sale Dollars

#now create the function that runs the models based on the inputted DF, p, d, and q values,
#then creates a DF storing all of the data which can be anaylzed to determine the best p,d,q values to use with the data
def evaluate_savgul(dataframe, p_values, d_values, q_values):
  file_num=0
  #file_num=135
  full_DF=pd.DataFrame()
  for p in p_values:
    for d in d_values:
      for q in q_values:
        order = (p,d,q)
        tmp_DF=pd.DataFrame()
        for stat in dataframe.columns:
            if stat=='Bottles Sold':
                win_int=256
            elif stat=='Sale (Dollars)':
                win_int=208
            try:
                mse = evaluate_arima_savgul(dataframe,stat,order,win_int,5,'median')
                tmp_DF.loc[0,'p']=p
                tmp_DF.loc[0,'d']=d
                tmp_DF.loc[0,'q']=q
                tmp_DF.loc[0,stat]=mse
            except:
                tmp_DF.loc[0,'p']=p
                tmp_DF.loc[0,'d']=d
                tmp_DF.loc[0,'q']=q
                tmp_DF.loc[0,stat]=0
                continue
        tmp_DF.to_csv(f'{default_directory}/datafiles/backups/backups_sav/tmp_DF_{file_num}_sav.csv',header=True,index=False)
        file_num=file_num+1
        temp_comb_DF=[full_DF,tmp_DF]
        full_DF=pd.concat(temp_comb_DF)
  return full_DF

'''#Make sure function works
p_values = range(0, 1)
d_values = range(0, 1)
q_values = range(0, 3)
arima_test_check_savgul=evaluate_savgul(teq_by_date, p_values, d_values, q_values)
arima_test_check_savgul
'''

'''Once again, the final output will be imported below.
#now collect the data
p_values = range(0, 5)
d_values = range(0, 3)
q_values = range(0, 9)
arima_test_check_savgul=evaluate_savgul(teq_by_date, p_values, d_values, q_values)
arima_test_check_savgul
'''

#now import and combine all of the tests to determine which combination of p,d,q values produces the best RMSE value
#The overall file will be imported below, no need to run this code again.
'''
RMSE_tests_DF_sav=pd.read_csv(f'{default_directory}/datafiles/backups/backups_sav/tmp_DF_0_sav.csv')
for i in range(1,243):
  tmp_RMSE_DF=pd.read_csv(f'{default_directory}/datafiles/backups/backups_sav/tmp_DF_{i}_sav.csv')
  temp_comb_DF=[RMSE_tests_DF_sav,tmp_RMSE_DF]
  RMSE_tests_DF_sav=pd.concat(temp_comb_DF)
RMSE_tests_DF_sav=RMSE_tests_DF_sav.reset_index(drop=True)
RMSE_tests_DF_sav'''

#RMSE_tests_DF_sav.to_csv(f'{default_directory}/datafiles/backups/backups_sav/RMSE_tests_DF_sav.csv',index=False, header=True)
RMSE_tests_DF_sav=pd.read_csv(f'{default_directory}/datafiles/backups/backups_sav/RMSE_tests_DF_sav.csv')
RMSE_tests_DF_sav

print(min(RMSE_tests_DF_sav['Bottles Sold'][RMSE_tests_DF_sav['Bottles Sold']!=0])) #1407.0792274899743
min(RMSE_tests_DF_sav['Sale (Dollars)'][RMSE_tests_DF_sav['Sale (Dollars)']!=0])    #24551.15253238082

best_RMSE_DF_sav=best_arima(RMSE_tests_DF_sav)
best_RMSE_DF_sav

#Create the function for determining the best models and then exporting them for later use.
def best_model_sav(timeseries_DF,best_arima_out,use_bias='no'):
  for col in timeseries_DF.columns:
    tmp_DF=pd.DataFrame()
    tmp_DF.index=timeseries_DF.index
    tmp_DF[col]=timeseries_DF[col]
    if col=='Bottles Sold':
        win_int=256
    elif col=='Sale (Dollars)':
        win_int=208
    #rolling/smooth the data
    sav_data=savgul_data(tmp_DF[col],win_int,5)
    tmp_DF['sav_data']=list(sav_data)
    # fit model
    model = ARIMA(tmp_DF['sav_data'], order=best_arima_out.loc[0,str(col)],trend='n')
    model_fit = model.fit()
    # bias
    if use_bias == 'yes':
        residuals=model_fit.resid
        bias = residuals.mean()
        np.save(f'{default_directory}/models/model_bias_{col}_sav.npy', [bias])
    # save model
    model_fit.save(f'{default_directory}/models/model_{col}_sav.pkl')

best_model_sav(teq_by_date,best_RMSE_DF_sav,'yes')
best_model_sav(teq_by_date,best_RMSE_DF_sav)

#now define the function for testing our model
def arima_forecast_validate_sav(timeseries_DF,test_DF,best_arima_out,mode,use_bias='no'):
  pred_val_DF=pd.DataFrame()
  pred_val_DF.index=test_DF.index
  y_pred_dict={}
  for col in timeseries_DF.columns:
      if col=='Bottles Sold':
          win_int=256
      elif col=='Sale (Dollars)':
          win_int=208
      tmp_DF=pd.DataFrame()
      tmp_DF.index=timeseries_DF.index
      tmp_DF[col]=timeseries_DF[col]
      sav_data=savgul_data(tmp_DF[col],win_int,poly=5)
      tmp_DF['sav_data']=list(sav_data)
      # load model
      model_fit = ARIMAResults.load(f'{default_directory}/models/model_{col}_sav.pkl')
      if use_bias=='yes':
          bias = np.load(f'{default_directory}/models/model_bias_{col}_sav.npy')
      else:
          bias=np.array([0])
      # make first prediction
      predictions = list()
      yhat = float(model_fit.forecast())
      tmp_time=test_DF.index[0]
      tmp_DF.loc[tmp_time,'sav_data']=float(yhat)
      tmp_DF['inverse']=inverse_savgul_data(tmp_DF['sav_data'], tmp_DF[col],mode)
      yhat = bias+tmp_DF.loc[tmp_time,'inverse']
      predictions.append(yhat)
      tmp_DF.loc[tmp_time,col]=test_DF.loc[tmp_time,col]
      pred_val_DF.loc[pred_val_DF.index[0],f'{str(col)}_Pred']="{:.2f}".format(yhat[0])
      pred_val_DF.loc[pred_val_DF.index[0],f'{str(col)}_Exp']="{:.2f}".format(test_DF[col][0])
      print('Stat: ', col)
      print('>Predicted=%.3f, Expected=%3.f' % (yhat, test_DF[col][0]))
      #make additional predictions
      for i in range(1, len(test_DF[col])):
        sav_data=savgul_data(tmp_DF[col],win_int,poly=5)
        tmp_DF['sav_data']=list(sav_data)
        model = ARIMA(tmp_DF['sav_data'], order=best_arima_out.loc[0,str(col)],trend='n')
        model.initialize_approximate_diffuse()
        model_fit = model.fit()
        yhat = float(model_fit.forecast())
        tmp_time=test_DF.index[i]
        tmp_DF.loc[tmp_time,'sav_data']=float(yhat)
        tmp_DF['inverse']=inverse_savgul_data(tmp_DF['sav_data'], tmp_DF[col],mode)
        yhat = bias+tmp_DF.loc[tmp_time,'inverse']
        predictions.append(yhat)
        tmp_DF.loc[tmp_time,col]=test_DF.loc[tmp_time,col]
        obs = test_DF[col][i]
        pred_val_DF.loc[pred_val_DF.index[i],f'{str(col)}_Pred']="{:.2f}".format(yhat[0])
        pred_val_DF.loc[pred_val_DF.index[i],f'{str(col)}_Exp']="{:.2f}".format(obs)
        print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
      # report performance
      mse = mean_squared_error(test_DF[col], predictions)
      rmse = sqrt(mse)
      print('RMSE: %.3f' % rmse)
      pred_val_DF.loc['RMSE',f'{str(col)}_Pred']=rmse
      pred_val_DF.loc['RMSE',f'{str(col)}_Exp']=rmse
      plt.plot(np.array(test_DF[col]),color='blue')
      plt.plot(predictions, color='red')
      plt.title(f'Stat: {col}')
      plt.show()
      y_pred_dict[f'y_{col}']=test_DF[col]
      y_pred_dict[f'predictions_{col}']=predictions
  return pred_val_DF, y_pred_dict['y_Bottles Sold'], y_pred_dict['predictions_Bottles Sold'],y_pred_dict['y_Sale (Dollars)'], y_pred_dict['predictions_Sale (Dollars)']

#test the model with bias
Preds_2017_sav_bias,exp_2017_btl_sav_bias,pred_2017_btl_sav_bias,exp_2017_sale_sav_bias,pred_2017_sale_sav_bias=\
arima_forecast_validate_sav(teq_by_date,teq_upto_2017,best_RMSE_DF_sav, mode='median',use_bias='yes')

#want to get a cleaner plot, going to try just plotting the 2016 prediction data
#find how many values are from 2016
print(Preds_2017_sav_bias[:-1][Preds_2017_sav_bias[:-1].index<pd.to_datetime('2017-01-01')]) #first 88 rows
#now plot just the 2016 data
plt.plot(np.array(exp_2017_btl_sav_bias[:88]),color='blue')
plt.plot(pred_2017_btl_sav_bias[:88], color='red')
plt.title('Stat: Btl, Just 2016')
plt.show()
plt.plot(np.array(exp_2017_sale_sav_bias[:88]),color='blue')
plt.plot(pred_2017_sale_sav_bias[:88], color='red')
plt.title('Stat: Sale, Just 2016')
plt.show()

#its still a little messy, try looking at just one months worth of 2016 data (December)
print(Preds_2017_sav_bias[:-1][(Preds_2017_sav_bias[:-1].index>pd.to_datetime('2016-11-30')) & (Preds_2017_sav_bias[:-1].index<pd.to_datetime('2017-01-01'))])
print(list(Preds_2017_sav_bias.index).index(pd.to_datetime('2016-12-01 00:00:00')))#index of first december date is 66
print(list(Preds_2017_sav_bias.index).index(pd.to_datetime('2016-12-30 00:00:00')))#index of first december date is 87
#now plot just the December 2016 data
plt.plot(np.array(exp_2017_btl_sav_bias[66:88]),color='blue')
plt.plot(pred_2017_btl_sav_bias[66:88], color='red')
plt.title('Stat: Btl, Just December 2016')
plt.show()
plt.plot(np.array(exp_2017_sale_sav_bias[66:88]),color='blue')
plt.plot(pred_2017_sale_sav_bias[66:88], color='red')
plt.title('Stat: Sale, Just December 2016')
plt.show()

#run again without bias
Preds_2017_sav,exp_2017_btl_sav,pred_2017_btl_sav,exp_2017_sale_sav,pred_2017_sale_sav=\
arima_forecast_validate_sav(teq_by_date,teq_upto_2017,best_RMSE_DF_sav,mode='median',use_bias='no')

#want to get a cleaner plot, going to try just plotting the 2016 prediction data
#find how many values are from 2016
print(Preds_2017_sav[:-1][Preds_2017_sav[:-1].index<pd.to_datetime('2017-01-01')]) #first 88 rows
#now plot just the 2016 data
plt.plot(np.array(exp_2017_btl_sav[:88]),color='blue')
plt.plot(pred_2017_btl_sav[:88], color='red')
plt.title('Stat: Btl, Just 2016')
plt.show()
plt.plot(np.array(exp_2017_sale_sav[:88]),color='blue')
plt.plot(pred_2017_sale_sav[:88], color='red')
plt.title('Stat: Sale, Just 2016')
plt.show()

#its still a little messy, try looking at just one months worth of 2016 data (December)
print(Preds_2017_sav[:-1][(Preds_2017_sav[:-1].index>pd.to_datetime('2016-11-30')) & (Preds_2017_sav[:-1].index<pd.to_datetime('2017-01-01'))])
print(list(Preds_2017_sav.index).index(pd.to_datetime('2016-12-01 00:00:00')))#index of first december date is 66
print(list(Preds_2017_sav.index).index(pd.to_datetime('2016-12-30 00:00:00')))#index of first december date is 87
#now plot just the December 2016 data
plt.plot(np.array(exp_2017_btl_sav[66:88]),color='blue')
plt.plot(pred_2017_btl_sav[66:88], color='red')
plt.title('Stat: Btl, Just December 2016')
plt.show()
plt.plot(np.array(exp_2017_sale_sav[66:88]),color='blue')
plt.plot(pred_2017_sale_sav[66:88], color='red')
plt.title('Stat: Sale, Just December 2016')
plt.show()

################################################################################
#### Modeling Using Years 2012 Through 2023 To Try To Create Predictions #######
################################################################################

##The model created using the raw, unsmoothed data provided the best output. I was able to pull together the data from 2018 up through November 2023. This data will be imported, combined with the 2012 through 2017 data, then used to build another model using the raw data.

##First it will be built using everything through 2022, and then tested using the 2023 data.

##Then a model will be built using all of the data through 2023, then I will try to forecast out predictions for the 31 days of December 2023.

#import the 2018-2023 data
# I needed to download the data in yearly chunks, otherwise downloading them all at once would have created a file that was too big
# Below is the code I used to import all of the years and combine them together. the final output will be imported below
'''tequila_2018_2023=pd.read_csv(f'{default_directory}/datafiles/2018_Iowa_Liquor_Sales.csv')
for i in range(2018,2024):
  tmp_liquor_DF=pd.read_csv(f'{default_directory}/datafiles/{i}_Iowa_Liquor_Sales.csv')
  temp_comb_DF=[tequila_2018_2023,tmp_liquor_DF]
  tequila_2018_2023=pd.concat(temp_comb_DF)
tequila_2018_2023=tequila_2018_2023.reset_index(drop=True)
tequila_2018_2023
'''

#tequila_2018_2023.to_csv(f'{default_directory}/datafiles/tequila_2018_2023.csv',index=False, header=True)
tequila_2018_2023=pd.read_csv(f'{default_directory}/datafiles/tequila_2018_2023.csv')
tequila_2018_2023

#need to convert to datetime
tequila_2018_2023.dtypes
tequila_2018_2023['Date']=pd.to_datetime(tequila_2018_2023['Date'])

#remove rows in the category name that have NA values
tequila_2018_2023=tequila_2018_2023.dropna(subset=['Category Name'])
#Make sure it is only made up of Tequila products
tequila_2018_2023=tequila_2018_2023[tequila_2018_2023['Category Name'].str.contains('TEQUILA')]

#now groupby date and isolate Bottles Sold and Sale (Dollars)
teq_18toNov23=tequila_2018_2023.groupby(["Date"])[["Bottles Sold","Sale (Dollars)"]].sum()
teq_18toNov23 #See what the most recent date is

#Now seperate out the 2023 data, that will be the test data
teq_23UpToNov=teq_18toNov23[teq_18toNov23.index>'2022-12-31']
print(teq_23UpToNov)
teq_18to22=teq_18toNov23[teq_18toNov23.index<='2022-12-31']
teq_18to22

#now combine the 2012 to 2016, the 2016/2017, and the 2018 to 2022 data altogether
temp_comb_DF=[teq_by_date,teq_upto_2017,teq_18to22]
teq_2012_2022=pd.concat(temp_comb_DF)
teq_2012_2022

#bring back the function for testing the arima models with raw data
def evaluate_arima_raw(X, arima_order):
    # prepare training dataset
    X = X.astype('float32')
    train_size = int(len(X) * 0.70)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model   = ARIMA(pd.Series(history), order=arima_order,trend='n')
        model_fit = model.fit()
        yhat = model_fit.forecast()
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    mse = mean_squared_error(test, predictions)
    rmse = sqrt(mse)
    return rmse

print(evaluate_arima_raw(teq_2012_2022['Bottles Sold'],(0,0,0))) #6109.945143429502
evaluate_arima_raw(teq_2012_2022['Bottles Sold'],(1,1,1)) #2554.332010691224

#now bring back the function that runs the models based on the inputted DF, p, d, and q values, then creates a DF storing all of the data.
def evaluate_raw_models(dataframe, p_values, d_values, q_values):
  file_num=0
  full_DF=pd.DataFrame()
  for p in p_values:
    for d in d_values:
      for q in q_values:
        order = (p,d,q)
        tmp_DF=pd.DataFrame()
        for stat in dataframe.columns:
            dataset=dataframe[stat]
            try:
              mse = evaluate_arima_raw(dataset, order)
              tmp_DF.loc[0,'p']=p
              tmp_DF.loc[0,'d']=d
              tmp_DF.loc[0,'q']=q
              tmp_DF.loc[0,stat]=mse
            except:
              tmp_DF.loc[0,'p']=p
              tmp_DF.loc[0,'d']=d
              tmp_DF.loc[0,'q']=q
              tmp_DF.loc[0,stat]=0
              continue
        tmp_DF.to_csv(f'{default_directory}/datafiles/backups/backups_raw_all/tmp_DF_{file_num}_raw_all.csv',header=True,index=False)
        file_num=file_num+1
        temp_comb_DF=[full_DF,tmp_DF]
        full_DF=pd.concat(temp_comb_DF)
  return full_DF

'''#I wanted to go back and try collecting the p,d,q values again for this model, now that it has specifically been trained using the additional data
#from 2017 to 2023. This is taking even longer then the p,d,q functionw as before, so for now I am just going to go forward with the p,d,q values
#that I determined when the model was created with data from 2012 through 2016.
p_values = range(0, 9)
d_values = range(0, 3)
q_values = range(0, 9)
arima_test_check_raw=evaluate_raw_models(teq_2012_2022, p_values, d_values, q_values)
arima_test_check_raw
'''

'''If I am able to get the new p,d,q, values collected, I will run the code in this cell to import all the values and determine the best p,d,q.
#For now, the values found in the first model created from the raw data will be used.
#now import and combine all of the tests to determine which combination of p,d,q values produces the best RMSE value
RMSE_tests_DF_raw_all=pd.read_csv(f'{default_directory}/datafiles/backups/backups_raw_all/tmp_DF_0_raw_all.csv')
for i in range(1,243):
  tmp_RMSE_DF_raw_all=pd.read_csv(f'{default_directory}/datafiles/backups/backups_raw_all/tmp_DF_{i}_raw_all.csv')
  temp_comb_DF_raw_all=[RMSE_tests_DF_raw_all,tmp_RMSE_DF_raw_all]
  RMSE_tests_DF_raw_all=pd.concat(temp_comb_DF_raw_all)
RMSE_tests_DF_raw_all=RMSE_tests_DF_raw_all.reset_index(drop=True)
RMSE_tests_DF_raw_all

#RMSE_tests_DF_raw_all.to_csv(f'{default_directory}/datafiles/backups/backups_raw_all/RMSE_tests_DF_raw_all.csv',index=False, header=True)
RMSE_tests_DF_raw_all=pd.read_csv(f'{default_directory}/datafiles/backups/backups_raw_all/RMSE_tests_DF_raw_all.csv')
RMSE_tests_DF_raw_all

#now run the function to create the DF containing the best p,d,q combinations for each stat
best_RMSE_DF_raw_all=best_arima(RMSE_tests_DF_raw_all)
best_RMSE_DF_raw_all
'''

#now create the function that builds and saves the model using best p,d,q combinations
def best_model_raw_all(timeseries_DF,best_arima_out,use_bias='no'):
  for col in timeseries_DF.columns:
    # fit model
    model = ARIMA(timeseries_DF[col], order=best_arima_out.loc[0,str(col)],trend='n')
    model_fit = model.fit()
    # bias
    if use_bias == 'yes':
        residuals=model_fit.resid
        bias = residuals.mean()
        np.save(f'{default_directory}/models/model_bias_{col}_raw_all.npy', [bias])
    # save model
    model_fit.save(f'{default_directory}/models/model_{col}_raw_all.pkl')

best_model_raw_all(teq_2012_2022,best_RMSE_DF_raw,'yes')
best_model_raw_all(teq_2012_2022,best_RMSE_DF_raw)

#now create the function for making the predictions
def arima_forecast_validate_raw_all(timeseries_DF,test_DF,best_arima_out,use_bias='no'):
  pred_val_DF=pd.DataFrame()
  pred_val_DF.index=test_DF.index
  y_pred_dict={}
  for col in timeseries_DF.columns:
      dataset=timeseries_DF[col]
      X = dataset.values.astype('float32')
      history = [x for x in X]
      validation = test_DF[col]
      y = validation.values.astype('float32')
      # load model
      model_fit = ARIMAResults.load(f'{default_directory}/models/model_{col}_raw_all.pkl')
      if use_bias=='yes':
          bias = np.load(f'{default_directory}/models/model_bias_{col}_raw_all.npy')
      else:
          bias=np.array([0])
      # make first prediction
      predictions = list()
      yhat = float(model_fit.forecast())
      yhat = bias + yhat
      predictions.append(yhat)
      history.append(y[0])
      pred_val_DF.loc[pred_val_DF.index[0],f'{str(col)}_Pred']="{:.2f}".format(yhat[0])
      pred_val_DF.loc[pred_val_DF.index[0],f'{str(col)}_Exp']="{:.2f}".format(y[0])
      print('Stat: ', col)
      print('>Predicted=%.3f, Expected=%3.f' % (yhat, y[0]))
      #make additional predictions
      for i in range(1, len(y)):
        model = ARIMA(pd.Series(history), order=best_arima_out.loc[0,str(col)],trend='n')
        model.initialize_approximate_diffuse()
        model_fit = model.fit()
        yhat = float(model_fit.forecast())
        yhat = bias + yhat
        predictions.append(yhat)
        obs = y[i]
        history.append(obs)
        pred_val_DF.loc[pred_val_DF.index[i],f'{str(col)}_Pred']="{:.2f}".format(yhat[0])
        pred_val_DF.loc[pred_val_DF.index[i],f'{str(col)}_Exp']="{:.2f}".format(obs)
        print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
      # report performance
      mse = mean_squared_error(test_DF[col], predictions)
      rmse = sqrt(mse)
      print('RMSE: %.3f' % rmse)
      pred_val_DF.loc['RMSE',f'{str(col)}_Pred']=rmse
      pred_val_DF.loc['RMSE',f'{str(col)}_Exp']=rmse
      plt.plot(np.array(test_DF[col]),color='blue')
      plt.plot(predictions, color='red')
      plt.title(f'Stat: {col}')
      plt.show()
      y_pred_dict[f'y_{col}']=test_DF[col]
      y_pred_dict[f'predictions_{col}']=predictions
  return pred_val_DF, y_pred_dict['y_Bottles Sold'], y_pred_dict['predictions_Bottles Sold'],y_pred_dict['y_Sale (Dollars)'], y_pred_dict['predictions_Sale (Dollars)']

Preds_2023_raw_bias,exp_2023_btl_raw_bias,pred_2023_btl_raw_bias,exp_2023_sale_raw_bias,pred_2023_sale_raw_bias=\
arima_forecast_validate_raw_all(teq_2012_2022,teq_23UpToNov,best_RMSE_DF_raw,'yes')

#want to get cleaner plots, break it down into ~3 month intervals
#find the indexs' from January through March
print(Preds_2023_raw_bias[:-1][Preds_2023_raw_bias[:-1].index<pd.to_datetime('2023-04-01')])   #first 70 rows
#now plot just the January through March
plt.plot(np.array(exp_2023_btl_raw_bias[:70]),color='blue')
plt.plot(pred_2023_btl_raw_bias[:70], color='red')
plt.title('Stat: Btl, Jan-Mar 2023')
plt.show()
plt.plot(np.array(exp_2023_sale_raw_bias[:70]),color='blue')
plt.plot(pred_2023_sale_raw_bias[:70], color='red')
plt.title('Stat: Sale, Jan-Mar 2023')
plt.show()

#Now April through June
print(Preds_2023_raw_bias[:-1][(Preds_2023_raw_bias[:-1].index>pd.to_datetime('2023-3-31')) & (Preds_2023_raw_bias[:-1].index<pd.to_datetime('2023-07-01'))])
#next 68 rows make up April through June.
#now plot just the April through June data
plt.plot(np.array(exp_2023_btl_raw_bias[70:138]),color='blue')
plt.plot(pred_2023_btl_raw_bias[70:138], color='red')
plt.title('Stat: Btl, April-June 2023')
plt.show()
plt.plot(np.array(exp_2023_sale_raw_bias[70:138]),color='blue')
plt.plot(pred_2023_sale_raw_bias[70:138], color='red')
plt.title('Stat: Sale, April-June 2023')
plt.show()

#Now July through September
print(Preds_2023_raw_bias[:-1][(Preds_2023_raw_bias[:-1].index>pd.to_datetime('2023-6-30')) & (Preds_2023_raw_bias[:-1].index<pd.to_datetime('2023-10-01'))])#next 69 rows
#now plot just the July through September data
plt.plot(np.array(exp_2023_btl_raw_bias[138:207]),color='blue')
plt.plot(pred_2023_btl_raw_bias[138:207], color='red')
plt.title('Stat: Btl, July-Sept 2023')
plt.show()
plt.plot(np.array(exp_2023_sale_raw_bias[138:207]),color='blue')
plt.plot(pred_2023_sale_raw_bias[138:207], color='red')
plt.title('Stat: Sale, July-Sept 2023')
plt.show()

#Now October and November
print(Preds_2023_raw_bias[:-1][Preds_2023_raw_bias[:-1].index>pd.to_datetime('2023-9-30')])  #next 53 rows
#now plot just the July through September data
plt.plot(np.array(exp_2023_btl_raw_bias[207:]),color='blue')
plt.plot(pred_2023_btl_raw_bias[207:], color='red')
plt.title('Stat: Btl, Oct,Nov 2023')
plt.show()
plt.plot(np.array(exp_2023_sale_raw_bias[207:]),color='blue')
plt.plot(pred_2023_sale_raw_bias[207:], color='red')
plt.title('Stat: Sale, Oct,Nov 2023')
plt.show()

Preds_2023_raw,exp_2023_btl_raw,pred_2023_btl_raw,exp_2023_sale_raw,pred_2023_sale_raw=\
arima_forecast_validate_raw_all(teq_2012_2022,teq_23UpToNov,best_RMSE_DF_raw)

#want to get cleaner plots, break it down into ~3 month intervals
#find the indexs from January through March
print(Preds_2023_raw[:-1][Preds_2023_raw[:-1].index<pd.to_datetime('2023-04-01')])   #first 70 rows
#now plot just the January through March
plt.plot(np.array(exp_2023_btl_raw[:70]),color='blue')
plt.plot(pred_2023_btl_raw[:70], color='red')
plt.title('Stat: Btl, Jan-Man 2023')
plt.show()
plt.plot(np.array(exp_2023_sale_raw[:70]),color='blue')
plt.plot(pred_2023_sale_raw[:70], color='red')
plt.title('Stat: Sale, Jan-Man 2023')
plt.show()

#Now April through June
print(Preds_2023_raw[:-1][(Preds_2023_raw[:-1].index>pd.to_datetime('2023-3-31')) & (Preds_2023_raw[:-1].index<pd.to_datetime('2023-07-01'))])  #next 68 rows
#now plot just the April through June data
plt.plot(np.array(exp_2023_btl_raw[70:138]),color='blue')
plt.plot(pred_2023_btl_raw[70:138], color='red')
plt.title('Stat: Btl, April-June 2023')
plt.show()
plt.plot(np.array(exp_2023_sale_raw[70:138]),color='blue')
plt.plot(pred_2023_sale_raw[70:138], color='red')
plt.title('Stat: Sale, April-June 2023')
plt.show()

#Now July through September
print(Preds_2023_raw[:-1][(Preds_2023_raw[:-1].index>pd.to_datetime('2023-06-30')) & (Preds_2023_raw[:-1].index<pd.to_datetime('2023-10-01'))])  #next 69 rows
#now plot just the July through September data
plt.plot(np.array(exp_2023_btl_raw[138:207]),color='blue')
plt.plot(pred_2023_btl_raw[138:207], color='red')
plt.title('Stat: Btl, July-Sept 2023')
plt.show()
plt.plot(np.array(exp_2023_sale_raw[138:207]),color='blue')
plt.plot(pred_2023_sale_raw[138:207], color='red')
plt.title('Stat: Sale, July-Sept 2023')
plt.show()

#Now October and November
print(Preds_2023_raw[:-1][Preds_2023_raw[:-1].index>pd.to_datetime('2023-9-30')])  #final rows
#now plot just the July through September data
plt.plot(np.array(exp_2023_btl_raw[207:]),color='blue')
plt.plot(pred_2023_btl_raw[207:], color='red')
plt.title('Stat: Btl, Oct,Nov 2023')
plt.show()
plt.plot(np.array(exp_2023_sale_raw[207:]),color='blue')
plt.plot(pred_2023_sale_raw[207:], color='red')
plt.title('Stat: Sale, Oct,Nov 2023')
plt.show()

####Bottle RMSE, raw data, all, no bias: 2490.64
####Bottle RMSE, raw data, all, with bias: 2490.87
####Revenue RMSE, raw data, all, no bias: 59343.8
####Revenue RMSE, raw data, all, with bias: 59343.5

################################################################################
#### Now Create The Model For Trying To Forecast December 2023 #################
################################################################################

#now build a model for trying to predict the total bottles sold and total revenue for december 2023
#combine together all data from 2012 to 2023
temp_comb_DF=[teq_2012_2022, teq_23UpToNov]
teq_2012_2023=pd.concat(temp_comb_DF)

#build the function
def arima_forecast_future(timeseries_DF,best_arima_out,days):
  pred_val_DF=pd.DataFrame()
  for col in timeseries_DF.columns:
    print('Stat: ', col)
    model = ARIMA(timeseries_DF[col], order=best_arima_out.loc[0,str(col)],trend='n')
    model.initialize_approximate_diffuse()
    model_fit = model.fit()
    forecast=model_fit.forecast(steps=days)
    predictions = list()
    count=1
    for yhat in forecast:
      day='day_'+str(count)
      print(f'{day} >Predicted=%.3f' % yhat)
      predictions.append(yhat)
      pred_val_DF.loc[day,f'{str(col)}_Pred']="{:.2f}".format(yhat)
      count =count + 1
    fig, ax = plt.subplots(figsize=(20,12))
    plt.plot(predictions, color='red')
    plt.title(f'Stat: {col}')
    plt.show()
  return pred_val_DF, predictions

#forecast out 31 days
December_23_DF, December_23_Pred=arima_forecast_future(teq_2012_2023,best_RMSE_DF_raw,31)

December_23_DF['Bottles Sold_Pred'].sum() #161038.21999999997

December_23_DF['Bottles Sold_Pred'].mean() #5194.7812903225795

December_23_DF['Sale (Dollars)_Pred'].sum() #3484176.7599999993

December_23_DF['Sale (Dollars)_Pred'].mean() #112392.7987096774

#now plot the data
print(December_23_DF.dtypes)
December_23_DF=December_23_DF.astype('float')

teq_2012_2023.index=pd.to_datetime(teq_2012_2023.index)
teq_date_plot=teq_2012_2023.copy(deep=True)
teq_date_plot['Dec_Day']=teq_date_plot.index.day

#plot the different decembers over the years altogether including our predicted December to see how it compares
#first total bottles
fig, ax = plt.subplots(figsize=(20,12))
for i in range(2012,2023):
    plt.plot(teq_date_plot['Dec_Day'][(teq_date_plot.index>pd.to_datetime(f'{i}-11-30')) & (teq_date_plot.index<=pd.to_datetime(f'{i}-12-31'))],teq_date_plot['Bottles Sold'][(teq_date_plot.index>pd.to_datetime(f'{i}-11-30')) & (teq_date_plot.index<=pd.to_datetime(f'{i}-12-31'))],label=i)
plt.plot(range(1,32),list(December_23_DF['Bottles Sold_Pred']),label=2023,linewidth=4)
fig.autofmt_xdate()
plt.legend()
#plt.tight_layout()
plt.title('December 2023 Total Bottles Sold Predictions With 2012-2022 December Numbers',fontsize=30)
plt.ylabel('Total Bottles Sold',fontsize=20)
plt.xlabel('Day Of The Month',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

#now plot revenue
fig, ax = plt.subplots(figsize=(20,12))
for i in range(2012,2023):
    plt.plot(teq_date_plot['Dec_Day'][(teq_date_plot.index>pd.to_datetime(f'{i}-11-30')) & (teq_date_plot.index<=pd.to_datetime(f'{i}-12-31'))],teq_date_plot['Sale (Dollars)'][(teq_date_plot.index>pd.to_datetime(f'{i}-11-30')) & (teq_date_plot.index<=pd.to_datetime(f'{i}-12-31'))],label=i)
plt.plot(range(1,32),list(December_23_DF['Sale (Dollars)_Pred']),label=2023,linewidth=4)
fig.autofmt_xdate()
plt.legend()
#plt.tight_layout()
plt.title('December 2023 Sale (Dollars) Predictions With 2012-2022 December Number',fontsize=30)
plt.ylabel('Sale (Dollars)',fontsize=20)
plt.xlabel('Day Of The Month',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

#now plot our 2023 predictions versus the average bottles sold and the average revenue on each day in December between 2012 and 2022
teq_date_mean=teq_date_plot.groupby('Dec_Day').mean()

#first bottles
fig, ax = plt.subplots(figsize=(20,12))
plt.plot(teq_date_mean.index,teq_date_mean['Bottles Sold'],label='Mean For 2012-2022')
plt.plot(teq_date_mean.index,list(December_23_DF['Bottles Sold_Pred']),label=2023,linewidth=4)
fig.autofmt_xdate()
plt.legend()
#plt.tight_layout()
plt.title('December 2023 Total Bottles Sold Predictions Versus 2012-2022 Average',fontsize=30)
plt.ylabel('Total Bottles Sold',fontsize=20)
plt.xlabel('Day Of The Month',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()


#Sale (Dollars)
fig, ax = plt.subplots(figsize=(20,12))
plt.plot(teq_date_mean.index,teq_date_mean['Sale (Dollars)'],label='Mean For 2012-2022')
plt.plot(teq_date_mean.index,list(December_23_DF['Sale (Dollars)_Pred']),label=2023,linewidth=4)
fig.autofmt_xdate()
plt.legend()
#plt.tight_layout()
plt.title('December 2023 Sale (Dollars) Sold Predictions Versus 2012-2022 Average',fontsize=30)
plt.ylabel('Sale (Dollars)',fontsize=20)
plt.xlabel('Day Of The Month',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

"""##The script finished running for collecting the p,d,q values for the model that was trained using all of the data up to 2022, so now going to check their values and see if new predictions should be made."""

'''
#now import and combine all of the tests to determine which combination of p,d,q values produces the best RMSE value
RMSE_tests_DF_raw_all=pd.read_csv(f'{default_directory}/datafiles/backups/backups_raw_all/tmp_DF_0_raw_all.csv')
for i in range(1,243):
  tmp_RMSE_DF_raw_all=pd.read_csv(f'{default_directory}/datafiles/backups/backups_raw_all/tmp_DF_{i}_raw_all.csv')
  temp_comb_DF_raw_all=[RMSE_tests_DF_raw_all,tmp_RMSE_DF_raw_all]
  RMSE_tests_DF_raw_all=pd.concat(temp_comb_DF_raw_all)
RMSE_tests_DF_raw_all=RMSE_tests_DF_raw_all.reset_index(drop=True)
RMSE_tests_DF_raw_all
'''

#RMSE_tests_DF_raw_all.to_csv(f'{default_directory}/datafiles/backups/backups_raw_all/RMSE_tests_DF_raw_all.csv',index=False, header=True)
RMSE_tests_DF_raw_all=pd.read_csv(f'{default_directory}/datafiles/backups/backups_raw_all/RMSE_tests_DF_raw_all.csv')
RMSE_tests_DF_raw_all

#now run the function to create the DF containing the best p,d,q combinations for each stat
best_RMSE_DF_raw_all=best_arima(RMSE_tests_DF_raw_all)
print('p,d,q values for the model using all raw data up to 2022:')
print(best_RMSE_DF_raw_all)

#compare it to the p,d,q values for the model that was trained on the raw data going up to 2016
RMSE_tests_DF_raw=pd.read_csv(f'{default_directory}/datafiles/backups/backups_raw/RMSE_tests_DF_raw.csv')
best_RMSE_DF_raw=best_arima(RMSE_tests_DF_raw)
print('\n')
print('p,d,q values for the model using raw data between 2012 and 2016:')
print(best_RMSE_DF_raw)

#The p,d,q values are all the same whether the full set of data is used up to 2022 or the data that went from 2012 to 2016, so no need to do any further testing.