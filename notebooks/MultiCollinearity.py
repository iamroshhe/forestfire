import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

df=pd.read_csv('cleaned_dataset.csv')

df['Classes'] = np.where(df['Classes'].str.contains('not fire'), 0, 1)        
df=df.drop(columns=['day','month','year'])

X = df.drop(columns=['FWI'])
y=df['FWI']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(X_train.shape,X_test.shape)

#Check for multicollinearity
plt.figure(figsize=(12,10))
corr=df.corr()
sns.heatmap(corr,annot=True)
plt.show()

def correlation(dataset,threshold):
    col_corr = set() 
    corr_matrix=dataset.corr() 
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

##threshold--Domain expertise
corr_features=correlation(X_train,0.85)

print(corr_features)

#Drop features when correlation is more than 0.85
X_train.drop(corr_features,axis=1,inplace=True)
X_test.drop(corr_features,axis=1,inplace=True)
print(X_train.shape,X_test.shape)

#Feature Scaling Or standardization
from sklearn.preprocessing import StandardScaler 
scaler=StandardScaler()

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
print(X_train_scaled[:5])

#Box Plots to understand Effect of Standard Scaler
plt.subplots(figsize=(15,5))
plt.subplot(1,2,1) ##1 row 2nd column 1 box
sns.boxplot(data=X_train)
plt.title('X_train Before Scaling')
plt.subplot(1,2,2)  ##1 row 2nd column 2 box
sns.boxplot(data=X_train_scaled)
plt.title('X_train After Scaling')
plt.show()

#LINEAR REGRESSION MODEL 
model=LinearRegression()
model.fit(X_train_scaled,y_train)
y_pred = model.predict(X_test_scaled)
mae=mean_absolute_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
print('Mean absolute error',mae)
print('R2 Score',score)

#LASSO REGRESSION
from sklearn.linear_model import Lasso
lasso=Lasso()
lasso.fit(X_train_scaled,y_train)
y_pred=lasso.predict(X_test_scaled)
mae=mean_absolute_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
print('Mean absolute error',mae)
print('R2 Score',score)

#RIDGE REGRESSION MODEL
from sklearn.linear_model import Ridge
ridge=Ridge()
ridge.fit(X_train_scaled,y_train)
y_pred=ridge.predict(X_test_scaled)
mae=mean_absolute_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
print('Mean absolute error',mae)
print('R2 Score',score)


#ELASTICNET REGRESSION MODEL
from sklearn.linear_model import ElasticNet
elastic=ElasticNet()
elastic.fit(X_train_scaled,y_train)
y_pred=elastic.predict(X_test_scaled)
mae=mean_absolute_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
print('Mean absolute error',mae)
print('R2 Score',score)

#ridge and linear regression have same accuracy but still ridge is better as it does not
#overfit
import pickle
pickle.dump(scaler,open('scaler.pkl','wb'))
pickle.dump(ridge,open('ridge.pkl','wb'))
