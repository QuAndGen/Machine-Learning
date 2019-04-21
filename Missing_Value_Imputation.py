
class missing_imputation:
    def __init__(self,data,num_var=None,char_var=None):
        self.data=data
        print('Below packages are requird for missing imputation','\n')
        print('import pandas as pd','\n')
        print('from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute','\n')
        print('import matplotlib.pyplot as plt','\n')
        if num_var==None or len(num_var)=='':
            data_t=data.dtypes
            self.num_var=list(data_t[data_t!='object'].index.values)
        else:
            self.num_var=num_var
        
        if char_var==None or len(char_var)=='':
            data_t=data.dtypes
            self.char_var=list(data_t[data_t=='object'].index.values)        
        else:
            self.char_var=char_var
    def missing_plot(self):
        x=round(self.data.isnull().sum()*1.0/self.data.shape[0],4)*100
        x.sort_values(ascending=False,inplace=True)
        lbl= np.arange(len(x))
        fig, ax = plt.subplots(figsize=(5,10))
        plt.barh(lbl, x.values, align='center',color='lightblue', ecolor='black')
        ax.set_yticks(lbl)
        ax.set_yticklabels(x.index.values)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Missing Percentange')
        ax.set_ylabel('Variables')
        ax.set_title('Missing pattern')
        plt.show()
            
    def is_missing(self,x,mu):
        if np.isnan(x):
            return(mu)
        else:
            return(x)
        
        
    def mean_imput(self,x):
        #print(type(x))
        mu=x.mean()
        y=x.apply(lambda x:self.is_missing(x,mu))
        return(y)
        
        
        #df['num_legs'].sample(n=3, random_state=1)
    
    def drop_imput(self):
        before_shape=self.data.shape[0]
        self.data.dropna(inplace=True)
        after_shape=self.data.shape[0]
        diff=np.array(before_shape)-np.array(after_shape)
        print(str(diff),'rows were removed due to presence of missing value out of',str(before_shape))
        return(self.data)
    
    def char_var_miss_imput(self,data):
        from sklearn.preprocessing import Imputer
        imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
        imp.fit(data)
        y= imp.transform(data)
        return(y)
        
    def missing_imput(self,method):
        if method=='drop':
            output=self.drop_imput()
        elif method=='mean1':
            output_num=self.data[self.num_var].apply(self.mean_imput,axis=1)
        elif method=='mean':
            val=dict(self.data[self.num_var].apply(np.mean))
            output_num=self.data[self.num_var].fillna(value=val)
        elif method=='median':
            val=dict(self.data[self.num_var].apply(np.median))
            output_num=self.data[self.num_var].fillna(value=val)
        elif method=='mode':
            output_num=self.char_var_miss_imput(self.data[self.num_var])
        else:
            output=self.data
        
        
        if method in ['mean1','mean','median','mode']:
            output_char=self.char_var_miss_imput(self.data[self.char_var])
            output=pd.concat([output_char,output_num],axis=1)
        
        self.data=output
        return(output)







