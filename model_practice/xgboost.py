# # coding:utf-8
# import pandas as pd 
# titanic=pd.read_csv('/Users/scarlett/repository/projects/titanic/titanic.csv')
# print (titanic.info())

# # 选取训练特征
# X=titanic[['pclass','age','sex']]
# y=titanic['survived']

# # 补全缺失值
# X['age'].fillna(X['age'].mean(),inplace=True)

# # 数据分割
# from sklearn.cross_validation import train_test_split
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)

# # 导入文本特征向量化的模块DictVectorizer
# from sklearn.feature_extraction import DictVectorizer
# vec=DictVectorizer(sparse=False)
# # 对原数据进行特征向量化处理
# X_train=vec.fit_transform(X_train.to_dict(orient='record'))
# X_test=vec.transform(X_test.to_dict(orient='record'))

# # 采用默认配置下的随机森林分类器对测试集进行预测
# from sklearn.ensemble import RandomForestClassifier
# rfc=RandomForestClassifier()
# rfc.fit(X_train,y_train)
# print ('rfc accuracy:',rfc.score(X_test,y_test))

# # 采用默认配置的xgboost模型对相同测试集进行预测
# from xgboost import XGBClassifier
# xgbc=XGBClassifier()
# xgbc.fit(X_train,y_train)
# print ('xgbc accuracy:',xgbc.score(X_test,y_test)) 


# 使用TensorFlow输出一句话
import tensorflow as tf 
import numpy as np 

# 初始化一个TensorFlow常量，使greeting作为一个计算模块
greeting=tf.constant("Hello Google!")
# 启动一个会话
sess=tf.Session()
# 使用会话执行greeting计算模块
result=sess.run(greeting)
# 输出会话执行结果
print result
# 关闭会话
sess.close()




# 使用TensorFlow完成一次线性函数的运算
# 声明matrix1为一个1*2的行向量，matrix2为一个2*1的列向量
matrix1=tf.constant([[3.,3.]])
matrix2=tf.constant([[2.],[2.]])
# Product将上面两个算子相乘，作为新算例
product=tf.matmul(matrix1, matrix2)
# 继续讲Product与一个标量2.0求和拼接，作为最终linear算例
linear=tf.add(product, tf.constant(2.0))
# 直接在会话中执行linear算例，相当于将上面所有单独的算例拼接成流程图来执行
with tf.Session() as sess:
	result=sess.run(linear)
	print result





# # 使用TensorFlow自定义一个线性分类器用于肿瘤预测
# import tensorflow as tf 
# import numpy as np 
# import pandas as pd 

# cancer=pd.read_csv('/Users/scarlett/repository/projects/breast_cancer/breast_cancer.csv')
# print cancer.shape
# print cancer.info()

# from sklearn.cross_validation import train_test_split
# X_train,X_test,y_train,y_test=cross_validation.train_test_split(cancer.data,cancer.target,test_size=0.25,random_state=33)

# # X_train=np.float32(train[['Clump Thickness','Cell Size']])


from sklearn import datasets,metrics,preprocessing,cross_validation
boston=datasets.load_boston()
X,y=boston.data,boston.target

X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.25,random_state=33)
scaler=preprocessing.StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

import skflow
tf_lr=skflow.TensorFlowLinearRegressor(steps=10000,learning_rate=0.01,batch_size=50)
tf_lr.fit(X_train, y_train)
tf_lr_y_predict=tf_lr.predict(X_test)

# 输出skflow中linearregressor模型的回归性能
print 'MAE:',metrics.mean_absolute_error(tf_lr_y_predict, y_test)
print 'MSE:',metrics.mean_squared_error(tf_lr_y_predict, y_test)
print 'R-squared:',metrics.r2_score(tf_lr_y_predict, y_test)

# 使用skflow的DNNRegressor，并注意其每个隐层特征数量的配置
tf_dnn_regressor=skflow.TensorFlowDNNRegressor(hidden_units=[100,40],
	steps=10000,learning_rate=0.01,batch_size=50)
tf_dnn_regressor.fit(X_train,y_train)
tf_dnn_regressor_y_predict=tf_dnn_regressor.predict(X_test)

# 输出skflow中DNNRegressor模型的回归性能
print 'MSE dnn:',metrics.mean_squared_error(tf_dnn_regressor_y_predict, y_test)
print 'MAE dnn:',metrics.mean_absolute_error(tf_dnn_regressor_y_predict, y_test)
print 'R-squared dnn:',metrics.r2_score(tf_dnn_regressor_y_predict, y_test)

# 使用sklearn中的RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor()
rfr.fit(X_train,y_train)
rfr_y_predict=rfr.predict(X_test)
print 'MSE rfr:',metrics.mean_squared_error(rfr_y_predict, y_test)
print 'MAE rfr:',metrics.mean_absolute_error(rfr_y_predict, y_test)
print 'R-squared rfr:',metrics.r2_score(rfr_y_predict, y_test)






















