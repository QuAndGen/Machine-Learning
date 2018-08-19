#changing data type of selected columns
for c, dtype in zip(prop.columns, prop.dtypes):
    if dtype = np.float64:
        prop[c] = prop[c].astype(np.float32)


quantitative = [f for f in prop.columns if prop.dtypes[f] == np.float64]
prop[quantitative] = prop[quantitative].astype(np.float32) 

for c in prop.columns:
    prop[c]=prop[c].fillna(-1)
    if prop[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(prop[c].values))
        prop[c] = lbl.transform(list(prop[c].values))

print( "\nProcessing data for CatBoost ...")
for c in prop.columns:
    prop[c]=prop[c].fillna(-1)
    if prop[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(prop[c].values))
        prop[c] = lbl.transform(list(prop[c].values))
        
cfeatures = list(prop.select_dtypes(include = ['int64', 'int32', 'uint8', 'int8']).columns)
#merge example
df_train = train.merge(prop, how='left', on='parcelid')
#merage two tables and get selected columns
X_test = (sample.merge(prop, on='ParcelId', how='left')).loc[:,features]

#fill missing value
df_train.fillna(df_train.median(),inplace = True)

#mising value with random years in the data
randomyears=pd.Series(np.random.choice(prop['yearbuilt'].dropna().values,len(prop)))
prop['yearbuilt']=prop['yearbuilt'].fillna(randomyears).astype(int)
med_yr=prop['yearbuilt'].quantile(0.5)
prop['New']=prop['yearbuilt'].apply(lambda x: 1 if x > med_yr else 0).astype(np.int8)  # adding a new feature

    
#delete few columns from data
x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 
                         'propertycountylandusecode', 'fireplacecnt', 'fireplaceflag'], axis=1)

# drop out ouliers
train_df=train_df[ train_df.logerror > -0.4 ]
train_df=train_df[ train_df.logerror < 0.419 ]

train = train[train['logerror'] <  train['logerror'].quantile(0.9975)]  # exclude 0.5% of outliers
train = train[train['logerror'] >  train['logerror'].quantile(0.0025)]



#ensabmble modeling
lgb_weight = (1 - XGB_WEIGHT - BASELINE_WEIGHT - CAT_WEIGHT) / (1 - OLS_WEIGHT)
xgb_weight0 = XGB_WEIGHT / (1 - OLS_WEIGHT)
baseline_weight0 =  BASELINE_WEIGHT / (1 - OLS_WEIGHT)
cat_weight0 = CAT_WEIGHT / (1 - OLS_WEIGHT)
pred0 = xgb_weight0*xgb_pred + baseline_weight0*BASELINE_PRED + lgb_weight*p_test + cat_weight0*cat_preds


#play with date variable
def get_features(df):
    df["transactiondate"] = pd.to_datetime(df["transactiondate"])
    df["transactiondate_year"] = df["transactiondate"].dt.year
    df["transactiondate_month"] = df["transactiondate"].dt.month
    df['transactiondate'] = df['transactiondate'].dt.quarter
    df = df.fillna(-1.0)
    return df

train = get_features(train[col])
test['transactiondate'] = '2016-01-01' #should use the most common training date
test = get_features(test[col])


test_dates = ['2016-10-01','2016-11-01','2016-12-01','2017-10-01','2017-11-01','2017-12-01']
test_columns = ['201610','201611','201612','201710','201711','201712']



    
    
#rename a column in pandas
train17.rename(columns={'parcelid': 'ParcelId'},inplace=True)

#play with Geographical data points

prop['longitude']=prop['longitude'].fillna(prop['longitude'].median()) / 1e6   #  convert to float32 later
prop['latitude'].fillna(prop['latitude'].median()) / 1e6
prop['censustractandblock'].fillna(prop['censustractandblock'].median()) / 1e12
    
#replace latitudes and longitudes with 500 clusters  (similar to ZIP codes)
from sklearn.cluster import MiniBatchKMeans
coords = np.vstack(prop[['latitude', 'longitude']].values)
sample_ind = np.random.permutation(len(coords))[:1000000]
kmeans = MiniBatchKMeans(n_clusters=500, batch_size=100000).fit(coords[sample_ind])
prop['Cluster'] = kmeans.predict(prop[['latitude', 'longitude']])

#collect categorical variable and give one order of number
qualitative = [f for f in prop.columns if prop.dtypes[f] == object]
prop[qualitative] = prop[qualitative].fillna('Missing')
for c in qualitative:  prop[c] = LabelEncoder().fit(list(prop[c].values)).transform(list(prop[c].values)).astype(int)

#do lable encoding for only categorical variable which are actually in numbers like any id
#make sure numeric id is not more than 100, if 100 then find another way to do lable encoding
smallval = [f for f in prop.columns if np.abs(prop[f].max())<100]
prop[smallval] = prop[smallval].fillna('Missing')
for c in smallval:  prop[c] = LabelEncoder().fit(list(prop[c].values)).transform(list(prop[c].values)).astype(np.int8)

#take the mean by month and map it with the table column
month_err=(train.groupby('Month').aggregate({'logerror': lambda x: np.mean(x)})- train['logerror'].mean()).values
train['Meanerror']=train['Month'].apply(lambda x: month_err[x-1]).astype(np.float32)


