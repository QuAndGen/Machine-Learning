#changing data type of selected columns
for c, dtype in zip(prop.columns, prop.dtypes):
    if dtype = np.float64:
        prop[c] = prop[c].astype(np.float32)

#merge example
df_train = train.merge(prop, how='left', on='parcelid')

#fill missing value
df_train.fillna(df_train.median(),inplace = True)

#delete few columns from data
x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 
                         'propertycountylandusecode', 'fireplacecnt', 'fireplaceflag'], axis=1)

#convert pandas sereis into numpy array
y_train = df_train['logerror'].values



for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

#delete unused objects
del df_train; gc.collect()

#change pandas dataframe into numpy array with datatype float32
x_train = x_train.values.astype(np.float32, copy=False)

d_train = lgb.Dataset(x_train, label=y_train)

##### RUN LIGHTGBM

params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.0021 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l1'          # or 'mae'
params['sub_feature'] = 0.5      # feature_fraction -- OK, back to .5, but maybe later increase this
params['bagging_fraction'] = 0.85 # sub_row
params['bagging_freq'] = 40
params['num_leaves'] = 512        # num_leaf
params['min_data'] = 500         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
params['verbose'] = 0


clf = lgb.train(params, d_train, 430)

del d_train; gc.collect()
del x_train; gc.collect()


sample = pd.read_csv("../input/sample_submission.csv")
sample['parcelid'] = sample['ParcelId']

df_test = sample.merge(prop, on='parcelid', how='left')

del sample, prop; gc.collect()

x_test = df_test[train_columns]

del df_test; gc.collect()

for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)

x_test = x_test.values.astype(np.float32, copy=False)

p_test = clf.predict(x_test)

del x_test; gc.collect()

