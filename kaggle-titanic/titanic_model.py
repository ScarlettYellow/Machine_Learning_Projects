# coding:utf-8
import pandas as pd 

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
print train.shape
print train.info()
print test.info()

# 人工选取用于预测的有效特征
selected_features=['Pclass','Sex','Age','Embarked','SibSp','Parch','Fare']
X_train=train[selected_features]
X_test=test[selected_features]
y_train=train['Survived']

# 需要补全Embarked特征的缺失值
print X_train['Embarked'].value_counts()
print X_test['Embarked'].value_counts()
# 对于embarked这种类别型特征，可使用出现频率最高的特征值来补充，以减少误差
X_test['Embarked'].fillna('S',inplace=True)
X_train['Embarked'].fillna('S',inplace=True)
# 对于Age这种数值型特征，可使用平均值或中位数来填充缺失值
X_train['Age'].fillna(X_train['Age'].mean(),inplace=True)
X_test['Age'].fillna(X_test['Age'].mean(),inplace=True)
X_test['Fare'].fillna(X_test['Fare'].mean(),inplace=True)
# 检查是否已补全
print X_train.info()
print X_test.info()

# 采用DictVectorizer进行特征向量化
from sklearn.feature_extraction import DictVectorizer
dict_vec=DictVectorizer(sparse=False)
X_train=dict_vec.fit_transform(X_train.to_dict(orient='record'))
dict_vec.feature_names_
X_test=dict_vec.transform(X_test.to_dict(orient='record'))

# 训练模型
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()

from xgboost import XGBClassifier
xgbc=XGBClassifier()

from sklearn.cross_validation import cross_val_score
# 使用5折交叉验证法在训练集上分别对默认配置的RandomForestClassifier和XGBClassifier进行性能评估，并获得平均分类准确性的得分
cross_val_score(rfc, X_train,y_train,cv=5).mean()
cross_val_score(xgbc,X_train,y_train,cv=5).mean()

# 使用模型进行预测
rfc.fit(X_train,y_train)
rfc_y_predict=rfc.predict(X_test)
rfc_submission=pd.DataFrame({'PassengerId': test['PassengerId'],'Survived': rfc_y_predict})
rfc_submission.to_csv('/Users/scarlett/repository/projects/kaggle-titanic/rfc_submission.csv',index=False)

xgbc.fit(X_train, y_train)
xgbc_y_predict=xgbc.predict(X_test)
xgbc_submission=pd.DataFrame({'PassengerId': test['PassengerId'],'Survived': rfc_y_predict})
xgbc_submission.to_csv('/Users/scarlett/repository/projects/kaggle-titanic/xgbc_submission.csv',index=False)

# 使用并行网络搜索的方式寻找更好的参数组合，以进一步提高XGBC的预测性能
from sklearn.grid_search import GridSearchCV
params={'max_depth':range(2,7),'n_estimators':range(100,1100,200),'learning_rate':[0.05,0.1,0.25,0.5,1.0]}
xgbc_best=XGBClassifier()
gs=GridSearchCV(xgbc_best, params,n_jobs=-1,cv=5,verbose=1)
gs.fit(X_train,y_train)

# 查验优化之后的XGBC的超参数配置和交叉验证的准确性
print gs.best_score_
print gs.best_params_

# 使用经优化超参数配置的XGBC对测试数据的预测结果存储在文件中
xgbc_best_y_predict=gs.predict(X_test)
xgbc_best_submission=pd.DataFrame({'PassengerId': test['PassengerId'],'Survived':xgbc_best_y_predict})
xgbc_best_submission.to_csv('/Users/scarlett/repository/projects/kaggle-titanic/xgbc_best_submission.csv',index=False)























