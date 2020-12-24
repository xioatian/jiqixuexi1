import matplotlib
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore') #  忽略弹出的warnings信息

data = pd.read_csv('../datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv')
pd.set_option('display.max_columns', None)    # 显示所有列

print(data.isnull().any())

data['TotalCharges'] = data['TotalCharges'].apply(pd.to_numeric, errors='coerce')
data['TotalCharges'] = data['TotalCharges'].fillna(data['MonthlyCharges'])


# data.describe()


# # tenure特征
# bp_list = list(data['tenure'])
#
# plt.figure(figsize=(20,4)) # 建立图像
# plt.boxplot(bp_list, vert=False, flierprops = {"marker":"o","markerfacecolor":"steelblue"})
# plt.show() # 展示箱型图
#
# # # MonthlyCharges特征
# # bp_list = list(data['MonthlyCharges'])
# #
# # plt.figure(figsize=(20,4)) # 建立图像
# # plt.boxplot(bp_list, vert=False, flierprops = {"marker":"o","markerfacecolor":"steelblue"})
# # plt.show() # 展示箱型图
# #
# # # TotalCharges
# #
# # bp_list = list(data['TotalCharges'])
# #
# # plt.figure(figsize=(20,4)) # 建立图像
# # plt.boxplot(bp_list, vert=False, flierprops = {"marker":"o","markerfacecolor":"steelblue"})
# # plt.show() # 展示箱型图


# # 目标变量正负样本的分布
# p = data['Churn'].value_counts()
# plt.figure(figsize=(10, 6))  # 构建图像
# # 绘制饼图并调整字体大小
# patches, l_text, p_text = plt.pie(p, labels=['No', 'Yes'], autopct='%1.2f%%', explode=(0, 0.1))
# # l_text是饼图对着文字大小，p_text是饼图内文字大小
# for t in p_text:
#     t.set_size(15)
# for t in l_text:
#     t.set_size(15)
# plt.show()  # 展示图像





# ### 性别、是否老年人、是否有配偶、是否有家属等特征对客户流失的影响
# # # 电话业务
# FemaleDf = data[data['gender'] == 'Female']
# MaleDf = data[data['gender'] == 'Male']
# SeniorCitizen0=data[data['SeniorCitizen'] == 0]
# SeniorCitizen1=data[data['SeniorCitizen'] == 1]
# Partner0=data[data['Partner'] == 'No']
# Partner1=data[data['Partner'] == 'Yes']
# Dependents0=data[data['Dependents'] == 'No']
# Dependents1=data[data['Dependents'] == 'Yes']
#
#
# fig = plt.figure(figsize=(15,6)) # 建立图像
#
# ax1 = fig.add_subplot(241)
# p1 = FemaleDf['Churn'].value_counts()
# ax1.pie(p1,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
# ax1.set_title('Churn of (gender = Female)')
#
# ax2 = fig.add_subplot(242)
# p2 = MaleDf['Churn'].value_counts()
# ax2.pie(p2,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
# ax2.set_title('Churn of (gender = Male)')
#
# ax3 = fig.add_subplot(243)
# p3 = SeniorCitizen0['Churn'].value_counts()
# ax3.pie(p3,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
# ax3.set_title('Churn of (SeniorCitizen = 0)')
#
# ax4 = fig.add_subplot(244)
# p4 = SeniorCitizen1['Churn'].value_counts()
# ax4.pie(p4,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
# ax4.set_title('Churn of (SeniorCitizen = 1)')
#
# ax5 = fig.add_subplot(245)
# p5 = Partner0['Churn'].value_counts()
# ax5.pie(p5,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
# ax5.set_title('Churn of (Partner = No)')
#
# ax6 = fig.add_subplot(246)
# p6 = Partner1['Churn'].value_counts()
# ax6.pie(p6,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
# ax6.set_title('Churn of (Partner = Yes)')
#
# ax7 = fig.add_subplot(247)
# p7= Dependents0['Churn'].value_counts()
# ax7.pie(p7,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
# ax7.set_title('Churn of (Dependents = No)')
#
# ax7 = fig.add_subplot(248)
# p7 = Dependents1['Churn'].value_counts()
# ax7.pie(p7,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
# ax7.set_title('Churn of (Dependents = Yes)')
#
#
# plt.tight_layout(pad=0.5)    # 设置子图之间的间距
# plt.show() # 展示饼状图
#
#
# ### 观察流失率与入网月数的关系
# # 折线图
# groupDf = data[['tenure', 'Churn']]    # 只需要用到两列数据
# groupDf['Churn'] = groupDf['Churn'].map({'Yes': 1, 'No': 0})    # 将正负样本目标变量改为1和0方便计算
# pctDf = groupDf.groupby(['tenure']).sum() / groupDf.groupby(['tenure']).count()    # 计算不同入网月数对应的流失率
# pctDf = pctDf.reset_index()    # 将索引变成列
#
# plt.figure(figsize=(10, 5))
# plt.plot(pctDf['tenure'], pctDf['Churn'], label='Churn percentage')    # 绘制折线图
# plt.legend()    # 显示图例
# plt.show()
#
# # 电话业务
# posDf = data[data['PhoneService'] == 'Yes']
# negDf = data[data['PhoneService'] == 'No']
#
# fig = plt.figure(figsize=(10,4)) # 建立图像
#
# ax1 = fig.add_subplot(331)
# p1 = posDf['Churn'].value_counts()
# ax1.pie(p1,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
# ax1.set_title('Churn of (PhoneService = Yes)')
#
# ax2 = fig.add_subplot(332)
# p2 = negDf['Churn'].value_counts()
# ax2.pie(p2,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
# ax2.set_title('Churn of (PhoneService = No)')
#
# # 多线业务
# df1 = data[data['MultipleLines'] == 'Yes']
# df2 = data[data['MultipleLines'] == 'No']
# df3 = data[data['MultipleLines'] == 'No phone service']
#
# ax3 = fig.add_subplot(334)
# p3 = df1['Churn'].value_counts()
# ax3.pie(p3,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
# ax3.set_title('Churn of (MultipleLines = Yes)')
#
# ax4 = fig.add_subplot(335)
# p4 = df2['Churn'].value_counts()
# ax4.pie(p4,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
# ax4.set_title('Churn of (MultipleLines = No)')
#
# ax5 = fig.add_subplot(336)
# p5= df3['Churn'].value_counts()
# ax5.pie(p5,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
# ax5.set_title('Churn of (MultipleLines = No phone service)')
#
# # 互联网业务
# df4 = data[data['InternetService'] == 'No']
# df5 = data[data['InternetService'] == 'Fiber optic']
# df6 = data[data['InternetService'] == 'DSL']
#
# ax6 = fig.add_subplot(337)
# p6= df4['Churn'].value_counts()
# ax6.pie(p5,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
# ax6.set_title('Churn of (InternetService = No)')
#
# ax7 = fig.add_subplot(338)
# p7= df5['Churn'].value_counts()
# ax7.pie(p5,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
# ax7.set_title('Churn of (InternetService = Fiber optic)')
#
# ax8 = fig.add_subplot(339)
# p8= df6['Churn'].value_counts()
# ax8.pie(p6,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
# ax8.set_title('Churn of (InternetService = DSL)')
#
# plt.tight_layout(pad=0.5)    # 设置子图之间的间距
# plt.show() # 展示饼状图
#
# 与互联网相关的业务
# internetCols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
#
# for i in internetCols:
#     df1 = data[data[i] == 'Yes']
#     df2 = data[data[i] == 'No']
#     df3 = data[data[i] == 'No internet service']
#
#     fig = plt.figure(figsize=(10, 3))  # 建立图像
#
#     ax1 = fig.add_subplot(131)
#     p1 = df1['Churn'].value_counts()
#     ax1.pie(p1, labels=['No', 'Yes'], autopct='%1.2f%%', explode=(0, 0.1))  # 开通业务
#     ax1.set_title('Yes')
#
#     ax2 = fig.add_subplot(132)
#     p2 = df2['Churn'].value_counts()
#     ax2.pie(p2, labels=['No', 'Yes'], autopct='%1.2f%%', explode=(0, 0.1))  # 未开通业务
#     ax2.set_title('No')
#
#     ax3 = fig.add_subplot(133)
#     p3 = df3['Churn'].value_counts()
#     ax3.pie(p3, labels=['No', 'Yes'], autopct='%1.2f%%', explode=(0, 0.1))  # 未开通互联网业务
#     ax3.set_title('No internet service')
#
#     plt.tight_layout()  # 设置子图之间的间距
#     plt.show()  # 展示饼状图
#
#
#
# # 合约期限
# df1 = data[data['Contract'] == 'Month-to-month']
# df2 = data[data['Contract'] == 'One year']
# df3 = data[data['Contract'] == 'Two year']
#
# fig = plt.figure(figsize=(15,6)) # 建立图像
#
# ax1 = fig.add_subplot(341)
# p1 = df1['Churn'].value_counts()
# ax1.pie(p1,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
# ax1.set_title('Churn of (Contract = Month-to-month)')
#
# ax2 = fig.add_subplot(342)
# p2 = df2['Churn'].value_counts()
# ax2.pie(p2,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
# ax2.set_title('Churn of (Contract = One year)')
#
# ax3 = fig.add_subplot(343)
# p3 = df3['Churn'].value_counts()
# ax3.pie(p3,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
# ax3.set_title('Churn of (Contract = Two year)')
#
# # 是否采用电子结算
# df5 = data[data['PaperlessBilling'] == 'Yes']
# df6 = data[data['PaperlessBilling'] == 'No']
#
#
#
# ax5 = fig.add_subplot(345)
# p5 = df1['Churn'].value_counts()
# ax5.pie(p5,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
# ax5.set_title('Churn of (PaperlessBilling = Yes)')
#
# ax6 = fig.add_subplot(346)
# p6 = df2['Churn'].value_counts()
# ax6.pie(p6,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
# ax6.set_title('Churn of (PaperlessBilling = No)')
#
#
# # 付款方式
# df7 = data[data['PaymentMethod'] == 'Bank transfer (automatic)']    # 银行转账（自动）
# df8 = data[data['PaymentMethod'] == 'Credit card (automatic)']    # 信用卡（自动）
# df9 = data[data['PaymentMethod'] == 'Electronic check']    # 电子支票
# df10 = data[data['PaymentMethod'] == 'Mailed check']    # 邮寄支票
#
#
#
# ax7 = fig.add_subplot(349)
# p7 = df7['Churn'].value_counts()
# ax7.pie(p7,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
# ax7.set_title('Churn of (PaymentMethod = Bank transfer')
#
# ax8 = fig.add_subplot(3,4,10)
# p8 = df8['Churn'].value_counts()
# ax8.pie(p8,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
# ax8.set_title('Churn of (PaymentMethod = Credit card)')
#
# ax9 = fig.add_subplot(3,4,11)
# p9 = df9['Churn'].value_counts()
# ax9.pie(p9,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
# ax9.set_title('Churn of (PaymentMethod = Electronic check)')
#
# ax10 = fig.add_subplot(3,4,12)
# p10 = df10['Churn'].value_counts()
# ax10.pie(p10,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
# ax10.set_title('Churn of (PaymentMethod = Mailed check)')
#
# plt.tight_layout(pad=0.5)    # 设置子图之间的间距
# plt.show() # 展示饼状图
#
# # 每月费用核密度估计图
# plt.figure(figsize=(10, 5))    # 构建图像
#
# negDf = data[data['Churn'] == 'No']
# sns.distplot(negDf['MonthlyCharges'], hist=False, label= 'No')
# posDf = data[data['Churn'] == 'Yes']
# sns.distplot(posDf['MonthlyCharges'], hist=False, label= 'Yes')
#
# plt.show()    # 展示图像
#
# # 总费用核密度估计图
# plt.figure(figsize=(10, 5))    # 构建图像
#
# negDf = data[data['Churn'] == 'No']
# sns.distplot(negDf['TotalCharges'], hist=False, label= 'No')
# posDf = data[data['Churn'] == 'Yes']
# sns.distplot(posDf['TotalCharges'], hist=False, label= 'Yes')
#
# plt.show()    # 展示图像
# data.info()



data.loc[data['MultipleLines']=='No phone service', 'MultipleLines'] = 'No'
internetCols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for i in internetCols:
    data.loc[data[i]=='No internet service', i] = 'No'

# 用1代替'Yes’，0代替 'No'
encodeCols = list(data.columns[3: 17].drop(['tenure', 'PhoneService', 'InternetService', 'StreamingTV', 'StreamingMovies', 'Contract']))
for i in encodeCols:
    data[i] = data[i].map({'Yes': 1, 'No': 0})
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

data['Contract']=data['Contract'].map({'Month-to-month':0, 'One year': 1,'Two year': 2})
data['PaymentMethod']=data['PaymentMethod'].map({'Bank transfer (automatic)':1, 'Credit card (automatic)': 2,'Electronic check': 3,'Mailed check': 4})
data['InternetService']=data['InternetService'].map({'No':0, 'DSL': 1,'Fiber optic': 2})

data = data.drop(['customerID', 'gender', 'PhoneService', 'StreamingTV', 'StreamingMovies'], axis=1)

from sklearn.preprocessing import StandardScaler    # 导入标准化库


#数据归一化
#StandardScaler只能归一数值数据
internetCols = ['SeniorCitizen','Partner','Dependents','tenure','MultipleLines','InternetService','OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport','Contract', 'PaperlessBilling','PaymentMethod','MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
for i in internetCols:
    data[i]=scaler.fit_transform(data[i].values.reshape(-1,1))

print(data.info())

# print(np.isfinite(data).all())
# # False:不包含
# # True:包含
# print(np.isinf(data).all())

# 空值排查
nan_list = data.isnull().sum().tolist()#把每一列的空值个数加起来
print(nan_list)
print(sum(nan_list))

# 无穷值排查
inf_list = np.isinf(data).sum().tolist()#把每一列的无穷值个数加起来
print(inf_list)
print(sum(inf_list))


#划分数据集
from sklearn.model_selection import train_test_split
def train_test_val_split(df,ratio_train,ratio_test,ratio_val):
    train, middle = train_test_split(df,test_size=1-ratio_train)
    ratio=ratio_val/(1-ratio_train)
    test,validation =train_test_split(middle,test_size=ratio)
    return train,test,validation
train,test,val=train_test_val_split(data,0.6,0.3,0.1)

train_churnDf = train['Churn'].to_frame()    # 取出目标变量列
train_featureDf = train.drop(['Churn'], axis=1)    # 所有特征列
test_churnDf = test['Churn'].to_frame()    # 取出目标变量列
test_featureDf = test.drop(['Churn'], axis=1)    # 所有特征列

#模型训练之逻辑回归模型
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#导入模型
Lr = LogisticRegression(C=100.0,random_state = 1)

#训练模型
Lr.fit(train_featureDf,train_churnDf)

#预测
test_label = Lr.predict(test_featureDf)

#输出结果
acc = accuracy_score(test_churnDf, test_label)
print("逻辑回归模型准确率: {}".format(acc))



#模型训练之决策树模型
from sklearn import tree

# 导入决策树模型
clf = tree.DecisionTreeClassifier()

#训练模型
clf.fit(train_featureDf,train_churnDf)

#预测
test_label = clf.predict(test_featureDf)

#输出结果
acc = accuracy_score(test_churnDf, test_label)
print("决策树模型准确率: {}".format(acc))

#模型训练之决策树模型
from sklearn import tree

# 导入决策树模型
clf1 = tree.DecisionTreeClassifier(criterion = 'entropy',max_depth=10) #更换criterion参数

#训练模型
clf1.fit(train_featureDf,train_churnDf)

#预测
test_label = clf1.predict(test_featureDf)

#输出结果
acc = accuracy_score(test_churnDf, test_label)
print("决策树模型准确率: {}".format(acc))






