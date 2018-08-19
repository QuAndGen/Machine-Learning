


##### RUN CATBOOST

train_pool = Pool(x_train, y_train) # cat_features=[0,2,5])
test_pool = Pool(x_test) #, cat_features=[0,2,5]) 

model = CatBoostRegressor(rsm=0.8, depth=5, learning_rate=0.037, eval_metric='MAE')
#train the model
model.fit(train_pool)

# make the prediction using the resulting model
cat_preds = model.predict(test_pool)


from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
X_train, X_eval, y_train, y_eval = train_test_split(X,y, test_size=0.15, random_state=1)
model = CatBoostRegressor(iterations=1000,learning_rate=0.002, depth=7, loss_function='MAE', 
                          eval_metric='MAE', random_seed=1)
model.fit(X_train, y_train, eval_set=(X_eval, y_eval), use_best_model=True, verbose=False, plot=True)
pred1 = model.predict(X_train)
pred2 = model.predict(X_eval)

FeatImp=pd.DataFrame(model.feature_importances_, index=features, columns=['Importance'])
FeatImp=FeatImp.sort_values('Importance')
FeatImp.plot(kind='barh', figsize=(8,14))
plt.show()




