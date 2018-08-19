# -*- coding: utf-8 -*-
"""
Created on Wed Aug 09 19:25:04 2017

@author: ram
"""




import warnings
import itertools

import pandas as pd
import numpy as np

import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('bmh')

#Visualize Tractor Sales data as time series
from io import StringIO
import requests

url = 'http://www.ucanalytics.com/blogs/wp-content/uploads/2015/06/Tractor-Sales.csv'
s = requests.get(url).text

sales_data = pd.read_csv(StringIO(s))
sales_data.head(5)

# since the complete date was not mentioned, we assume that it was the first of every month
dates = pd.date_range(start='2003-01-01', freq='MS', periods=len(sales_data))

import calendar
sales_data['Month'] = dates.month
sales_data['Month'] = sales_data['Month'].apply(lambda x: calendar.month_abbr[x])
sales_data['Year'] = dates.year

sales_data.head()

sales_data.drop(['Month-Year'], axis=1, inplace=True)
sales_data.rename(columns={'Number of Tractor Sold':'Tractor-Sales'}, inplace=True)
sales_data = sales_data[['Month', 'Year', 'Tractor-Sales']]

# set the dates as the index of the dataframe, so that it can be treated as a time-series dataframe
sales_data.set_index(dates, inplace=True)

# extract out the time-series
sales_ts = sales_data['Tractor-Sales']

plt.figure(figsize=(10, 5))
plt.plot(sales_ts)
plt.xlabel('Years')
plt.ylabel('Tractor Sales')


fig, axes = plt.subplots(2, 2, sharey=False, sharex=False)
fig.set_figwidth(14)
fig.set_figheight(8)
axes[0][0].plot(sales_ts.index, sales_ts, label='Original')
axes[0][0].plot(sales_ts.index, sales_ts.rolling(window=4).mean(), label='4-Months Rolling Mean')
axes[0][0].set_xlabel("Years")
axes[0][0].set_ylabel("Number of Tractor's Sold")
axes[0][0].set_title("4-Months Moving Average")
axes[0][0].legend(loc='best')
axes[0][1].plot(sales_ts.index, sales_ts, label='Original')
axes[0][1].plot(sales_ts.index, sales_ts.rolling(window=6).mean(), label='6-Months Rolling Mean')
axes[0][1].set_xlabel("Years")
axes[0][1].set_ylabel("Number of Tractor's Sold")
axes[0][1].set_title("6-Months Moving Average")
axes[0][1].legend(loc='best')
axes[1][0].plot(sales_ts.index, sales_ts, label='Original')
axes[1][0].plot(sales_ts.index, sales_ts.rolling(window=8).mean(), label='8-Months Rolling Mean')
axes[1][0].set_xlabel("Years")
axes[1][0].set_ylabel("Number of Tractor's Sold")
axes[1][0].set_title("8-Months Moving Average")
axes[1][0].legend(loc='best')
axes[1][1].plot(sales_ts.index, sales_ts, label='Original')
axes[1][1].plot(sales_ts.index, sales_ts.rolling(window=12).mean(), label='12-Months Rolling Mean')
axes[1][1].set_xlabel("Years")
axes[1][1].set_ylabel("Number of Tractor's Sold")
axes[1][1].set_title("12-Months Moving Average")
axes[1][1].legend(loc='best')
plt.tight_layout()
plt.show()


#As we could see in the above plots, 12-month moving average could produce a wrinkle free curve as desired. 
#This on some level is expected since we are using month-wise data for our analysis and there is expected monthly-seasonal 
#effect in our data.
#Let's plot the rolling mean and standard deviation on window of 12 months.

#Determing rolling statistics
rolmean = sales_ts.rolling(window=12).mean()
rolstd = sales_ts.rolling(window=12).std()

#Plot rolling statistics:
orig = plt.plot(sales_ts, label='Original')
mean = plt.plot(rolmean, label='Rolling Mean')
std = plt.plot(rolstd, label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)


#Dickey-Fuller Test - Let's run the Dicky Fuller Test on the timeseries and verify the null hypothesis that 
#the TS is non-stationary.


# Perform Dickey-Fuller test:
from statsmodels.tsa.stattools import adfuller
print('Results of Dickey-Fuller Test:')
dftest = adfuller(sales_ts, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#lags Used', 'Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)

#Though the variation in standard deviation is small, rolling mean is clearly increasing with time and this is 
#not a stationary series. Also, the test statistic is way more than the critical values.
#As we observed while plotting the moving average over months that there is a monhly pattern, now, 
#let’s decipher the seasonal component.

#Seasonality – Time Series Decomposition
#The first thing to do is to see how number of tractors sold vary on a month on month basis. 
#We will plot a stacked annual plot to observe seasonality in our data.

monthly_sales_data = pd.pivot_table(sales_data, values = "Tractor-Sales", columns = "Year", index = "Month")
monthly_sales_data = monthly_sales_data.reindex(index = ['Jan','Feb','Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
monthly_sales_data

monthly_sales_data.plot()


yearly_sales_data = pd.pivot_table(sales_data, values = "Tractor-Sales", columns = "Month", index = "Year")
yearly_sales_data = yearly_sales_data[['Jan','Feb','Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]
yearly_sales_data

yearly_sales_data.boxplot()

#Important Inferences
#The tractor sales have been increasing without fail every year.
#July and August are the peak months for tractor sales and the variance and 
#the mean value in July and August are also much higher than any of the other months.
#We can see a seasonal cycle of 12 months where the mean value of each month starts 
#with a increasing trend in the beginning of the year and drops down towards the end of the year. 
#We can see a seasonal effect with a cycle of 12 months.

#Irregular Remainder – Time Series Decomposition
#To decipher underlying patterns in tractor sales, we build a multiplicative time series 
#decomposition model with the following equation:
                        Yt=Trend_t×Seasonality_t×Remainder_t

#Instead of multiplicative model we could have chosen additive model as well. In that case 
#the equaltion would have been:
                        Yt=Trend_t+Seasonality_t+Remainder_t 
#However, it would have made very little difference in terms of conclusion we will draw 
#from this time series decomposition exercise. Additionally, we are also aware that plain 
#vanilla decomposition models like these are rarely used for forecasting. 
#Their primary purpose is to understand underlying patterns in temporal data to use in 
#more sophisticated analysis like Holt-Winters seasonal method or ARIMA.

decomposition = sm.tsa.seasonal_decompose(sales_ts, model='multiplicative')

fig = decomposition.plot()
fig.set_figwidth(12)
fig.set_figheight(8)
fig.suptitle('Decomposition of multiplicative time series')
plt.show()


#The following are some of our key observations from this analysis:
#1) Trend: 12-months moving average looks quite similar to a straight line hence we 
            #could have easily used linear regression to estimate the trend in this data.
#2) Seasonality: As discussed, seasonal plot displays a fairly consistent month-on-month pattern. 
            #The monthly seasonal components are average values for a month after removal of trend. Trend is removed from the time series using the following formula:
#        Seasonality_t×Remainder_t=Yt/Trend_t
 
#3) Irregular Remainder (random): is the residual left in the series after removal of trend and 
        #seasonal components. Remainder is calculated using the following formula:
#       Remainder_t=Yt/(Trend_t×Seasonality_t)
 
#The expectations from remainder component is that it should look like a white noise i.e. 
#displays no pattern at all. However, for our series residual display some pattern with high 
#variation on the edges of data i.e. near the beginning (2004-07) 
#and the end (2013-14) of the series.


#ARIMA Modeling
#ARIMA is a combination of 3 parts i.e. AR (AutoRegressive), I (Integrated), 
#and MA (Moving Average). A convenient notation for ARIMA model is ARIMA(p,d,q). 
#Here p,d, and q are the levels for each of the AR, I, and MA parts. 
#Each of these three parts is an effort to make the final residuals display a white noise 
#pattern (or no pattern at all). In each step of ARIMA modeling, 
#time series data is passed through these 3 parts like a sugar cane through a sugar 
#cane juicer to produce juice-less residual. The sequence of three passes for 
#ARIMA analysis is as following:
#1st Pass of ARIMA to Extract Juice / Information
#Integrated (I) – subtract time series with its lagged series to extract trends from the data
#In this pass of ARIMA juicer, we extract trend(s) from the original time series data. Differencing is one of the most commonly used mechanisms for extraction of trends. Here, the original series is subtracted with it’s lagged series e.g. November’s sales values are subtracted with October’s values to produce trend-less residual series. The formulae for different orders of differencing are as follow:

#No Differencing (d=0) |  Y′t=Yt 
#1st Differencing (d=1) |  Y′t=Yt−Yt_1 
#2nd Differencing (d=1) |  Y′t=Yt-Yt_1−(Yt_1−Yt_2)=Yt_2×Yt_1+Yt_2 

#For example, in the below plot a time series data with a linearly 
#upward trend is displayed. Adjacent to that plot is the 1st order differenced 
#plot for the same data. As we can notice after 1st order differencing, trend part 
#of the series is extracted and the difference data (residual) does not display any trend.

# convert to a dataframe for easy plot and differenciation

url = 'http://ucanalytics.com/blogs/wp-content/uploads/2017/08/dummy-sales.csv'
s = requests.get(url).text

random_df = pd.read_csv(StringIO(s))

random_df.set_index(pd.date_range(start='2012-01-01', freq='MS', periods=len(random_df)), inplace=True)

fig, axes = plt.subplots(1, 2, sharey=False, sharex=False)
fig.set_figwidth(12)
fig.set_figheight(4)
axes[0].plot(random_df.index, random_df['Sales'])
axes[0].set_xlabel("Years")
axes[0].set_ylabel("Sales")
axes[0].set_title("Original Data")

axes[1].plot(random_df.index, random_df['Sales'].diff(periods=1))
axes[1].set_xlabel("Years")
axes[1].set_ylabel("Differenced Sales")
axes[1].set_title("1$^{st}$ Order Differenced Data")
plt.tight_layout()
plt.show()


#The residual data of most time series usually become trend-less after the first order 
#differencing which is represented as ARIMA(0,1,0). Notice, AR (p), and MA (q) values in 
#this notation are 0 and the integrated (I) value has order one. If the residual series 
#still has a trend it is further differenced and is called 2nd order differencing. 
#This trend-less series is called stationary on mean series i.e. mean or average value 
#for series does not change over time.
#2nd Pass of ARIMA to Extract Juice / Information
#AutoRegressive (AR) – extract the influence of the previous periods’ values on the current period
#After the time series data is made stationary through the integrated (I) pass, 
#the AR part of the ARIMA juicer gets activated. As the name auto-regression suggests, 
#here we try to extract the influence of the values of previous periods on the current 
#period e.g. the influence of the September and October’s sales value on the November’s sales. 
#This is done through developing a regression model with the time lagged period values as 
#independent or predictor variables. The general form of the equation for this regression 
#model is shown below.
                Yt=c+α1Yt_1+α_2*Yt_2+........+α_p*Yt_p+e_t
#AR model of order 1 i.e. p=1 or ARIMA(1,0,0) is represented by the following regression equation
                Yt=c+α_1*Y_t−1+e_t
#3rd Pass of ARIMA to Extract Juice / Information
#Moving Average (MA) – extract the influence of the previous period’s error terms on the 
#current period’s error
#Finally, the last component of ARIMA juicer i.e. MA involves finding relationships between the 
#previous periods' error terms on the current period's error term. Keep in mind, this moving average 
#(MA) has nothing to do with moving average we learned about in the previous article on time series 
#decomposition. Moving Average (MA) part of ARIMA is developed with the following simple multiple 
#linear regression values with the lagged error values as independent or predictor variables.
                Yt=c+e_t+β1*e_t_1+β_2*e_t_2+......+β_q*e_t_q 
#MA model of order 1 i.e. q=1 or ARIMA(0,0,1) is represented by the following regression equation
                Yt=c+e_t+β_1*e_t_1 
#White Noise & ARIMA
#White noise is a funny thing, if we look at it for long we will start seeing some false patterns. 
#This is because the human brain is wired to find patterns, and at times confuses noises with signals.
# The biggest proof of this is how people lose money every day on the stock market. 
# This is precisely the reason why we need a mathematical or logical process to distinguish between 
# a white noise and a signal (juice / information). For example, consider the following simulated 
# white noise:
#    
  
# generate white noise
np.random.seed(1)
# plot of discrete white noise
plt.figure(figsize=(10, 5))
white_noise = np.random.normal(size=1000)
plt.plot(white_noise)
plt.xlabel('Data')
plt.ylabel('Time Index')
plt.show()

#If we stare at the above graph for a reasonably long time we may start seeing some false patterns. 
#A good way to distinguish between signal and noise is ACF (AutoCorrelation Function). 
#This is developed by finding the correlation between a series of its lagged values. 
#In the following ACF plot, we could see that for lag = 0 the 
#ACF plot has the perfect correlation i.e.  ρρ =1. This makes sense because any data with 
#itself will always have the perfect correlation. However as expected, our white noise doesn't 
#have a significant correlation with its historic values (lag >= 1). The dotted horizontal 
#lines in the plot show the threshold for the insignificant region i.e. for a significant 
#correlation the vertical bars should fall outside the horizontal dotted lines.
 

plt.figure()
smt.graphics.plot_acf(white_noise, lags=30, alpha=0.5)
plt.show()

#Difference data to make data stationary on mean (remove trend)
#Clearly the tractor sales has an upward trend for tractors sales and there is also a seasonal 
#component that we have already analyzed.
#The next thing to do is to make the series stationary by removing the upward trend through 
#1st order differencing of the series using the following formula:
#1st Differencing (d=1) |  Y′t=Yt−Yt_1


plt.figure(figsize=(10, 5))
plt.plot(sales_ts.diff(periods=1))
plt.xlabel('Years')
plt.ylabel('Tractor Sales')

#Okay so the above series is not stationary on variance i.e. variation in the plot is 
#increasing as we move towards the right of the chart. We need to make the series stationary on 
#variance to produce reliable forecasts through ARIMA models.

#log transform data to make data stationary on variance
#One of the best ways to make a series stationary on variance is through transforming the 
#original series through log transform. We will go back to our original tractor sales series 
#and log transform it to make it stationary on variance. The following equation represents 
#the process of log transformation mathematically:
#Log of sales :  Ytnew=log10(Yt)
#Notice, this series is not stationary on mean since we are using the original data without 
#differencing. But now the series looks stationary on variance.

plt.figure(figsize=(10, 5))
plt.plot(np.log10(sales_ts))
plt.xlabel('Years')
plt.ylabel('Log (Tractor Sales)')

#Difference log transform data to make data stationary on both mean and variance
#Let us look at the differenced plot for log transformed series to reconfirm 
#if the series is actually stationary on both mean and variance.
#1st Differencing (d=1) of Log of sales :  Ytnew=log10(Yt)−log10(Yt−1)

plt.figure(figsize=(10, 5))
plt.plot(np.log10(sales_ts).diff(periods=1))
plt.xlabel('Years')
plt.ylabel('Differenced Log (Tractor Sales)')

#Yes, now this series looks stationary on both mean and variance. 
#This also gives us the clue that I or integrated part of our ARIMA model will be equal to 1 
#as 1st difference is making the series stationary.
#Step 5: Plot ACF and PACF to identify potential AR and MA model
#Now, let us create autocorrelation factor (ACF) and partial autocorrelation factor (PACF) 
#plots to identify patterns in the above data which is stationary on both mean and variance. 
#The idea is to identify presence of AR and MA components in the residuals.

sales_ts_log = np.log10(sales_ts)
sales_ts_log.dropna(inplace=True)

sales_ts_log_diff = sales_ts_log.diff(periods=1) # same as ts_log_diff = ts_log - ts_log.shift(periods=1)
sales_ts_log_diff.dropna(inplace=True)

fig, axes = plt.subplots(1, 2, sharey=False, sharex=False)
fig.set_figwidth(12)
fig.set_figheight(4)
smt.graphics.plot_acf(sales_ts_log_diff, lags=30, ax=axes[0], alpha=0.5)
smt.graphics.plot_pacf(sales_ts_log_diff, lags=30, ax=axes[1], alpha=0.5)
plt.tight_layout()

#Since, there are enough spikes in the plots outside the insignificant zone 
#(dotted horizontal lines) we can conclude that the residuals are not random. 
#This implies that there is juice or information available in residuals to be extracted by 
#AR and MA models. Also, there is a seasonal component available in the residuals at the 
#lag 12 (represented by spikes at lag 12). This makes sense since we are analyzing monthly 
#data that tends to have seasonality of 12 months because of patterns in tractor sales.
#Step 6: Identification of best fit ARIMA model
#In order to fit the time series data with a seasonal ARIMA model, we need to first 
#find the the values of ARIMA(p,d,q)(P,D,Q)s that optimize a metric of interest such as AIC or BIC. 
#There are many guidelines and best practices to achieve this goal, yet the correct 
#parametrization of ARIMA models can be a painstaking manual process that requires domain 
#expertise and time. Auto arima function in forecast package in R helps us identify the best 
#fit ARIMA model on the fly but in Python we will generate combination of p,d and q to select 
#the optimal parameter values for our ARIMA(p,d,q)(P,D,Q)s time series model.
#This technique is known as "grid search" where we iteratively explore different combinations 
#of parameters. For each such combination of parameters, we try to fit a new seasonal ARIMA model 
#with the SARIMAX() function from the statsmodels module and assess AIC or BIC score. 
#The model with the best score wins and the parmeters for that model are the optimal parmeters.

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, d and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

import sys
warnings.filterwarnings("ignore") # specify to ignore warning messages

best_aic = np.inf
best_pdq = None
best_seasonal_pdq = None
temp_model = None

for param in pdq:
    for param_seasonal in seasonal_pdq:
        
        try:
            temp_model = sm.tsa.statespace.SARIMAX(sales_ts_log,
                                             order = param,
                                             seasonal_order = param_seasonal,
                                             enforce_stationarity=True,
                                             enforce_invertibility=True)
            results = temp_model.fit()

            # print("SARIMAX{}x{}12 - AIC:{}".format(param, param_seasonal, results.aic))
            if results.aic < best_aic:
                best_aic = results.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal
        except:
            #print("Unexpected error:", sys.exc_info()[0])
            continue
print("Best SARIMAX{}x{}12 model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))


#The best fit model is selected based on Akaike Information Criterion (AIC) , and Bayesian 
#Information Criterion (BIC) values. The idea is to choose a model with minimum AIC and BIC values.
#Akaike Information Criterion (AIC) - AIC is an effort to balance the model between goodness-of-fit 
#and number of parameters used in the model. This is similar to the balancing act between income 
#and cost of a company so that the debs of the company is optimized (Debt = Cost - Income). 
#As a modeler, we care about the maximum goodness of fit (income) with the minimum number of 
#parameters (cost).
#AIC=2K−2ln(L) 
#For the given model, L in the above formula is the maximized value of the likelihood function 
#representing goodness-of-fit, and K the number of estimated parameters. Like our debts, we want to 
#keep AIC value at the minimum to choose the best possible model. Bayesian Information Criterion (BIC) 
#is another variant of AIC and is used for the same purpose of best fit model selection. For the best 
#possible model selection, we want to look at AIC, BIC, and AICc (AIC with sample correction) if all 
#these values are minimum for a given model. With increasing parameters K will increase and hence AIC 
#increases. While with the goodness of the fit L increases thus decreasing AIC.
#As expected, our model has I (or integrated) component equal to 1. This represents differencing of 
#order 1. There is additional differencing of lag 12 in the above best fit model. Moreover, the best 
#fit model has MA value of order 1. Also, there is seasonal MA with lag 12 of order 1.
#The model that we got here is different from the one that was discussed in the blog. Let us validate 
#that model with ours how does that perform with statsmodel as well as our model.




# comparing our model with the best model reported in the blog
blog_param_order = (0, 1, 1)
blog_param_seasonal_order = (0, 1, 1, 12)
blog_model = sm.tsa.statespace.SARIMAX(sales_ts_log,
                                      order=blog_param_order,
                                      seasonal_order=blog_param_seasonal_order,
                                      enforce_stationarity=True,
                                      enforce_invertibility=True)
blog_results = blog_model.fit()
print("Blog SARIMAX{}x{}12 model - AIC:{}".format(blog_param_order, blog_param_seasonal_order, 
      blog_results.aic))

#Step 7: Predict sales on in-sample date using the best fit ARIMA model
#The next step is to predict tractor sales for in-sample data and find out how close is the 
#model prediction on the in-sample data to the actual truth.

best_model = sm.tsa.statespace.SARIMAX(sales_ts_log,
                                      order=(0, 1, 1),
                                      seasonal_order=(1, 0, 1, 12),
                                      enforce_stationarity=True,
                                      enforce_invertibility=True)
best_results = best_model.fit()

pred_dynamic = best_results.get_prediction(start=pd.to_datetime('2012-01-01'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()

# Extract the predicted and true values of our time series
sales_ts_forecasted = pred_dynamic.predicted_mean
sales_ts_truth = sales_ts_log['2012-01-01':]

# Compute the mean square error
mse = ((sales_ts_forecasted - sales_ts_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 4)))

axis = sales_ts['2005':].plot(label='Observed', figsize=(10, 6))
np.power(10, pred_dynamic.predicted_mean).plot(ax=axis, label='Dynamic Forecast', alpha=0.7)
axis.fill_between(pred_dynamic_ci.index, pred_dynamic_ci.iloc[:, 0], pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)
axis.fill_betweenx(axis.get_ylim(), pd.to_datetime('2012-01-01'), sales_ts.index[-1], alpha=.1, zorder=-1)
axis.set_xlabel('Years')
axis.set_ylabel('Tractor Sales')
plt.legend(loc='best')
plt.show()
plt.close()


#Step 8: Forecast sales using the best fit ARIMA model
#The next step is to predict tractor sales for next 3 years i.e. for 2015, 2016, and 2017 through the 
#above model.

# Get forecast 36 steps (3 years) ahead in future
n_steps = 36
pred_uc_99 = best_results.get_forecast(steps=36, alpha=0.01) # alpha=0.01 signifies 99% confidence interval
pred_uc_95 = best_results.get_forecast(steps=36, alpha=0.05) # alpha=0.05 95% CI

# Get confidence intervals 95% & 99% of the forecasts
pred_ci_99 = pred_uc_99.conf_int()
pred_ci_95 = pred_uc_95.conf_int()

n_steps = 36
idx = pd.date_range(sales_ts.index[-1], periods=n_steps, freq='MS')
fc_95 = pd.DataFrame(np.column_stack([np.power(10, pred_uc_95.predicted_mean), np.power(10, pred_ci_95)]), 
                     index=idx, columns=['forecast', 'lower_ci_95', 'upper_ci_95'])
fc_99 = pd.DataFrame(np.column_stack([np.power(10, pred_ci_99)]), 
                     index=idx, columns=['lower_ci_99', 'upper_ci_99'])
fc_all = fc_95.combine_first(fc_99)
fc_all = fc_all[['forecast', 'lower_ci_95', 'upper_ci_95', 'lower_ci_99', 'upper_ci_99']] # just reordering columns
fc_all.head()

# plot the forecast along with the confidence band
axis = sales_ts.plot(label='Observed', figsize=(15, 6))
fc_all['forecast'].plot(ax=axis, label='Forecast', alpha=0.7)
axis.fill_between(fc_all.index, fc_all['lower_ci_95'], fc_all['upper_ci_95'], color='k', alpha=.25)
#axis.fill_between(fc_all.index, fc_all['lower_ci_99'], fc_all['upper_ci_99'], color='k', alpha=.75)
axis.set_xlabel('Years')
axis.set_ylabel('Tractor Sales')
plt.legend(loc='best')
plt.show()


#Step 9: Plot ACF and PACF for residuals of ARIMA model to ensure no more information is left 
#for extraction
#Finally, let’s create an ACF and PACF plot of the residuals of our best fit ARIMA model i.e. 
#ARIMA(0,1,1)(1,0,1)[12].

best_results.plot_diagnostics(lags=30, figsize=(16,12))
plt.show()

#We need to ensure that the residuals of our model are uncorrelated and normally distributed with 
#zero-mean. If it is not that it signifies that the model can be further improved and we repeat the 
#process with the residuals.
#In this case, our model diagnostics suggests that the model residuals are normally distributed based 
#on the following:
#The KDE plot of the residuals on the top right is almost similar with the normal distribution.
#The qq-plot on the bottom left shows that the ordered distribution of residuals (blue dots) follows 
#the linear trend of the samples taken from a standard normal distribution with N(0, 1). Again, 
#this is a strong indication that the residuals are normally distributed.
#The residuals over time (top left plot) don't display any obvious seasonality and appear to be 
#white noise. This is confirmed by the autocorrelation (i.e. correlogram) plot on the bottom right, 
#which shows that the time series residuals have low correlation with lagged versions of itself.
#Those observations coupled with the fact that there are no spikes outside the insignificant zone 
#for both ACF and PACF plots lead us to conclude that that residuals are random with no information 
#or juice in them and our model produces a satisfactory fit that could help us understand our time 
#series data and forecast future values. It sems that our ARIMA model is working fine.

#Step 10: Regression with ARIMA Errors (Work in Progress)
#For the last 4 years, PowerHorse tractors is running an expensive marketing and farmer connect 
#program to boost their sales. They are interested in learning the impact of this program on overall 
#sales. As a data science consultant we are helping them with this effort. This is an interesting 
#problem and requires a thorough analysis followed by creative solutions and scientific monitoring 
#mechanism. To begin with we will build models based on regression with ARIMA errors and compare them 
#with the pure play ARIMA model. This analysis will provide some clues towards effectiveness of the 
#marketing program. However, this analysis will not be conclusive for finding shortcomings and 
#enhancements for the program which will require further analysis and creative solutions.


url = 'http://ucanalytics.com/blogs/wp-content/uploads/2017/08/sales-and-marketing.csv'
s = requests.get(url).text

mkt_sales_data = pd.read_csv(StringIO(s))

# since the complete date was not mentioned, we assume that it was the first of every month
mkt_sales_data.set_index(pd.date_range(start='2011-01-01', freq='MS', periods=len(mkt_sales_data)), inplace=True)
mkt_sales_data.drop(['Time Period'], axis=1, inplace=True)
mkt_sales_data.rename(columns={'Sales':'Tractor-Sales'}, inplace=True)
mkt_sales_data.head(5)

plt.figure()
plt.scatter(x=mkt_sales_data['Marketing Expense'], y=mkt_sales_data['Tractor-Sales'])
plt.xlabel('Monthly Marketing Expense (in 100$)')
plt.ylabel('Monthly Tractor Sales')

print("Correlation for Marketing Data: {}".format(round(mkt_sales_data.corr()['Marketing Expense']['Tractor-Sales'], 2)))

#This looks promising with quite a high correlation coefficient ( ρρ  > 0.8). However, 
#there is a lurking danger in analyzing non-stationary time series data. Since two uncorrelated 
#series can display high correlation because of time series trend in data. In this case, 
#PowerHorse is a growing company and the latent factor is 'growth' of the company. 
#Hence both its sales and marketing expenses can be on an upward curve independent of each other. 
#To investigat that a better way is to find the correlation between stationary data obtained through 
#differencing of marketing expediture and the trator sales data individually.
#The following is the correlation plot for stationary data:

mkt_sales_data_diff = mkt_sales_data.diff(periods=1)
mkt_sales_data_diff.dropna(inplace=True)

plt.figure()
plt.scatter(x=mkt_sales_data_diff['Marketing Expense'], y=mkt_sales_data_diff['Tractor-Sales'])
plt.xlabel('Differenced Marketing Expense')
plt.ylabel('Differenced Tractor Sales')

print("Correlation for Stationary Data: {}".format(round(mkt_sales_data_diff.corr()['Marketing Expense']['Tractor-Sales'], 2)))

#Ok, so that near perfect correlation has now disappeared though there is still some correlation 
#in this data ( ρρ  = 0.41).
#Typically, the marketing effort for the previous few months needs to have a good correlation with 
#sales for an effective marketing program. The marketing expense for the last month as displayed below 
#has very little correlation ( ρρ  = 0.17):
    
len(mkt_sales_data_diff)
prev_months_corr_data = pd.DataFrame(np.column_stack((\
                                        mkt_sales_data_diff['Marketing Expense'][0:46].values,\
                                        mkt_sales_data_diff['Tractor-Sales'][1:47].values)), \
                                     columns=["Previous Month's Differenced Marketing Expense" , 
                                              "Differenced Tractor Sales"])

plt.figure()
plt.scatter(x=prev_months_corr_data["Previous Month's Differenced Marketing Expense"], \
            y=prev_months_corr_data["Differenced Tractor Sales"])
plt.xlabel("Previous Month Difference Marketing Expense")
plt.ylabel("Differenced Tractor Sales")
plt.show()

print(
    "Correlation of current Months's Differenced Tractor Sales with Previous Month's Differenced Marketing Expense: {}"\
    .format(round(prev_months_corr_data.corr().iloc[0,1], 2)))

clip_length = len(mkt_sales_data_diff)
clip_length

lagged_mkt_exp_diff_sales_data = pd.DataFrame(np.column_stack((\
                                        mkt_sales_data_diff['Marketing Expense'],\
                                        mkt_sales_data_diff['Marketing Expense'].shift(periods=1)[0:clip_length],\
                                        mkt_sales_data_diff['Marketing Expense'].shift(periods=2)[0:clip_length],\
                                        mkt_sales_data_diff['Marketing Expense'].shift(periods=3)[0:clip_length],\
                                        mkt_sales_data_diff['Tractor-Sales'][0:47].values)), \
                                     columns=["Diff_Mkt_Exp", "Diff_Mkt_Exp_Lag_1", "Diff_Mkt_Exp_Lag_2", "Diff_Mkt_Exp_Lag_3", "Diff_Sales"])

lagged_mkt_exp_diff_sales_data.set_index(mkt_sales_data_diff.index, inplace=True)
lagged_mkt_exp_diff_sales_data.head(10)

#We now want to create different by considering the different lagged marketing expenditure 
#attributes as regressors and investigate whether previous motnh's/quarter's affect the tractor 
#sales or not (in other words if we incoporate the marketing efforts does they lower the AIC 
#              score or not. We would be building 4 ARIMA models with regressors ranging from 
#              current month to last quarter)

import sys

def auto_arima(timeseries, regressors=None, p=range(0, 2), d=range(0, 1), q=range(0, 2)):

    # Generate all different combinations of p, d and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    
    warnings.filterwarnings("ignore") # specify to ignore warning messages

    best_aic = np.inf
    best_pdq = None
    best_seasonal_pdq = None
    best_results = None

    for param in pdq:
        for param_seasonal in seasonal_pdq:

            try:
                temp_model = sm.tsa.statespace.SARIMAX(endog=timeseries,
                                                 exog=regressors,
                                                 order = param,
                                                 seasonal_order = param_seasonal,
                                                 enforce_stationarity=True,
                                                 enforce_invertibility=True)
                temp_results = temp_model.fit()

                # print("SARIMAX{}x{}12 - AIC:{}".format(param, param_seasonal, results.aic))
                if temp_results.aic < best_aic:
                    best_aic = temp_results.aic
                    best_pdq = param
                    best_seasonal_pdq = param_seasonal
                    best_results = temp_results
            except:
                #print("Unexpected error:", sys.exc_info()[0])
                continue
                
                

    print("Best SARIMAX{}x{}12 model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))
    print(best_results.summary().tables[0])
    print(best_results.summary().tables[1])

#ARIMA Model No Regressor
timeseries = lagged_mkt_exp_diff_sales_data["Diff_Sales"][3:47]
regressors = lagged_mkt_exp_diff_sales_data["Diff_Mkt_Exp"][3:47]
auto_arima(timeseries, regressors, p=range(0, 2), d=range(0, 1), q=range(0, 2))

#ARIMA Model with 1 month Lag Marekting Sales Data Regressor
timeseries = lagged_mkt_exp_diff_sales_data["Diff_Sales"][3:47]
regressors = lagged_mkt_exp_diff_sales_data[["Diff_Mkt_Exp", "Diff_Mkt_Exp_Lag_1"]][3:47]
auto_arima(timeseries, regressors, p=range(0, 2), d=range(0, 1), q=range(0, 2))

#ARIMA Model with 2 months Lag Marekting Sales Data Regressor
timeseries = lagged_mkt_exp_diff_sales_data["Diff_Sales"][3:47]
regressors = lagged_mkt_exp_diff_sales_data[["Diff_Mkt_Exp", "Diff_Mkt_Exp_Lag_1", "Diff_Mkt_Exp_Lag_2"]][3:47]
auto_arima(timeseries, regressors, p=range(0, 2), d=range(0, 1), q=range(0, 2))

#ARIMA Model with 3 months Lag Marekting Sales Data Regressor
timeseries = lagged_mkt_exp_diff_sales_data["Diff_Sales"][3:47]
regressors = lagged_mkt_exp_diff_sales_data[["Diff_Mkt_Exp", "Diff_Mkt_Exp_Lag_1", "Diff_Mkt_Exp_Lag_2", "Diff_Mkt_Exp_Lag_3"]][3:47]
auto_arima(timeseries, regressors, p=range(0, 2), d=range(0, 1), q=range(0, 2))

    