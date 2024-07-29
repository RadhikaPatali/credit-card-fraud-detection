import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv('./fraudTrain.csv')
test_df = pd.read_csv('./fraudTest.csv')
print("Train fraud Data size=",train_df.shape)
print("Test fraud Data size=",test_df.shape)

train_df.describe()  # General statistics

train_df.columns

test_df.tail(5)

train_df.info()

test_df.info()

fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(train_df.corr(),cmap="YlGnBu", annot=True)
plt.show()

train_df.loc[train_df['is_fraud'] == 1].sort_values('amt', ascending=False).head(2)

plt.figure(figsize=(9,7))
plt.title('Number of frauds by category')
sns.barplot(x="gender", y='is_fraud' ,data=train_df)

# Category
plt.figure(figsize=(16,8))
plt.title('Number of frauds by category')
sns.barplot(x="category", y='is_fraud' ,data=train_df)

print("Number of is_fraud data")
print(train_df['is_fraud'].value_counts())

from sklearn.utils import resample 
# .iloc[:,22] = is_fraud
df_minority = train_df[train_df.iloc[:,22].values==0]
df_majority = train_df[train_df.iloc[:,22].values==1] 
 
# Downsample majority class
df_minority_downsampled = resample(df_minority,
                                 n_samples=7506,
                                 random_state=42)
 
# Combine minority class with downsampled majority class
train_df_final = pd.concat([df_minority_downsampled, df_majority])
 
# final train data
train_df_final.info()


print("Number of is_fraud data",train_df_final['is_fraud'].value_counts())

train_df_final['trans_date_trans_time'] = pd.to_datetime(train_df_final['trans_date_trans_time'])
train_df_final['week_number'] = train_df_final['trans_date_trans_time'].dt.dayofweek
assert train_df_final['week_number'].max() == 6
train_df_final['month_number'] = train_df_final['trans_date_trans_time'].dt.month
assert train_df_final['month_number'].max() == 12
train_df_final['year'] = train_df_final['trans_date_trans_time'].dt.year
train_df_final.head()

test_df['trans_date_trans_time'] = pd.to_datetime(test_df['trans_date_trans_time'])
test_df['week_number'] = test_df['trans_date_trans_time'].dt.dayofweek
assert test_df['week_number'].max() == 6
test_df['month_number'] = test_df['trans_date_trans_time'].dt.month
assert test_df['month_number'].max() == 12
test_df['year'] = test_df['trans_date_trans_time'].dt.year
test_df.head()

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix

category_onehot = pd.get_dummies(train_df_final.category, prefix='category')
train_df_final = train_df_final.join(category_onehot)
train_df_final.head()

category_onehot_test_data = pd.get_dummies(test_df.category, prefix='category')
test_df = test_df.join(category_onehot_test_data)
test_df.head()

train_df_final['gender'] = train_df_final['gender'].replace(['F','M'],[0,1])
test_df['gender'] = test_df['gender'].replace(['F','M'],[0,1])
print('Gender of train dataset', train_df_final['gender'].value_counts())
print('Gender of test dataset', test_df['gender'].value_counts())

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
x_train = train_df_final['merchant']
train_df_final['merchant_number'] = label_encoder.fit_transform(x_train)
x_test = test_df['merchant']
test_df['merchant_number'] = label_encoder.fit_transform(x_test)
print('Merchant Number of train dataset',train_df_final['merchant_number'])
print('Merchant Number of test dataset',test_df['merchant_number'])

from datetime import date
def calculate_age(row):
    today = date.today()
    return today.year - row['dob'].year - ((today.month, today.day) < (row['dob'].month, row['dob'].day))

train_df_final['dob'] = pd.to_datetime(train_df_final['dob'])
train_df_final['age'] = train_df_final['dob']
train_df_final['age'] = train_df_final.apply (lambda row: calculate_age(row), axis=1)

test_df['dob'] = pd.to_datetime(test_df['dob'])
test_df['age'] = test_df['dob']
test_df['age'] = test_df.apply (lambda row: calculate_age(row), axis=1)

print('Age of train dataset', train_df_final['age'].head(3))
print('Age of test dataset', test_df['age'].head(3))

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

x_train = train_df_final['job']
train_df_final['job_number'] = label_encoder.fit_transform(x_train)
print(train_df_final['job_number'])
x_test = test_df['job']
test_df['job_number'] = label_encoder.fit_transform(x_test)
print(test_df['job_number'])

data_train = train_df_final[['amt','category_shopping_net','category_grocery_pos','category_home','category_misc_net',
                                         'category_kids_pets','category_health_fitness','gender','age','month_number',
                                         'category_food_dining','unix_time','category_personal_care','category_shopping_pos','is_fraud']]


data_test = test_df[['amt','category_shopping_net','category_grocery_pos','category_home','category_misc_net',
                                         'category_kids_pets','category_health_fitness','gender','age','month_number',
                                         'category_food_dining','unix_time','category_personal_care','category_shopping_pos','is_fraud']]

X_train = data_train[['amt','category_shopping_net','category_grocery_pos','category_home','category_misc_net',
                                         'category_kids_pets','category_health_fitness','gender','age','month_number',
                                         'category_food_dining','unix_time','category_personal_care','category_shopping_pos']]
y_train = data_train['is_fraud']


X_test = data_test[['amt','category_shopping_net','category_grocery_pos','category_home','category_misc_net',
                                         'category_kids_pets','category_health_fitness','gender','age','month_number',
                                         'category_food_dining','unix_time','category_personal_care','category_shopping_pos']]
y_test = data_test['is_fraud']


from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()
newValue = scaler.fit_transform(X_train)
X_train = pd.DataFrame(newValue, columns=X_train.columns)
X_train.head()

scaler = preprocessing.MinMaxScaler()
newValue = scaler.fit_transform(X_test)
X_test = pd.DataFrame(newValue, columns=X_test.columns)
X_test.head()

from sklearn.svm import SVC  
clf = SVC(kernel='linear') 
  
# fitting x samples and y classes 
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

cf=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cf/np.sum(cf), annot=True, 
            fmt='.2%', cmap='Blues')

print("Classification report")
print(classification_report(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Confusion matrix
cf=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cf/np.sum(cf), annot=True, 
            fmt='.2%', cmap='Blues')

print("Classification report")
print(classification_report(y_test, y_pred))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

#Predict
y_pred = model.predict(X_test)

# Confusion matrix
cf=confusion_matrix(y_test,y_pred)

X_test.shape

plt.figure(figsize=(10,8))
sns.heatmap(cf/np.sum(cf), annot=True, 
            fmt='.2%', cmap='Blues')

print("Classification report")
print(classification_report(y_test, y_pred))

from sklearn import metrics

y_pred_proba = model.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
roc_auc = metrics.auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc,
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()

#Dataset
#https://www.kaggle.com/datasets/kartik2112/fraud-detection

from xgboost import XGBClassifier

# fit model no training data
clf = XGBClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Confusion matrix
cf=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cf/np.sum(cf), annot=True, 
            fmt='.2%', cmap='Blues')

print("Classification report")
print(classification_report(y_test, y_pred))

import xgboost as xgb
from sklearn.metrics import accuracy_score


# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180,
        'seed': 0
    }

xgb.XGBClassifier()

def objective(space):
    clf=xgb.XGBClassifier(
                    n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
                    reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                    colsample_bytree=int(space['colsample_bytree']))
    
    evaluation = [( X_train, y_train), ( X_test, y_test)]
    
    clf.fit(X_train, y_train,
            eval_set=evaluation, eval_metric="auc",
            early_stopping_rounds=10,verbose=False)
    

    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred>0.5)
    print ("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK }

trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 20,
                        trials = trials)

print("The best hyperparameters are : ","\n")
print(best_hyperparams)