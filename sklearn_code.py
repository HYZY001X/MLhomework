import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression#线性回归
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB#朴素贝叶斯
from sklearn.svm import LinearSVC#支持向量机
from sklearn.metrics import accuracy_score
import scipy.io as scio

dataFile = r".\mnist-original.mat"
mnist = scio.loadmat(dataFile)
# 'mnist.data' is 70k x 784 array, each row represents the pixels from a 28x28=784 image
#data是784*70k阵列，70k列每列储存784个像素点信息，这784个像素点对应于28*28的数字图片
# make the value of pixels from [0, 255] to [0, 1] for further process
#归一化处理，将像素取值从uint8范围转化为double范围
X = mnist['data']/255.
# 'mnist.target' is 70k x 1 array, each row represents the target class of the corresponding image
#label是70k列阵列，储存预标记好的数字值
Y = mnist['label']
#数据转置
X=X.T
Y=Y.T
# split data to train and test (for faster calculation, just use 1/10 data)
#分割数据进行训练和测试（为了更快地计算，只需使用1/10数据）
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X[::10,:], Y[::10,:], test_size=1000)

#logistic regression
#逻辑回归分析
logistic_model = LogisticRegression(penalty='l2',dual=False,tol=0.0001,C=1.0,fit_intercept=True,intercept_scaling=1,class_weight=None,random_state=None,solver='warn',max_iter=100,multi_class='warn',verbose=0,warm_start=False,n_jobs=None,l1_ratio=None)
logistic_model.fit(X_train, Y_train.ravel())   # 逻辑回归建模
logistic_predicted_train = logistic_model.predict(X_train)
#calculate the train accuracy of logistic regression
logistic_train_accuracy = accuracy_score(Y_train.ravel(),logistic_predicted_train.ravel())
#calculate the test acuracy of logistic regression
logistic_predicted_test = logistic_model.predict(X_test)
logistic_test_accuracy = accuracy_score(Y_test.ravel(),logistic_predicted_test.ravel())

#naive bayes
#朴素贝叶斯分类器
nb_model = BernoulliNB(alpha=1.0,binarize=None,class_prior=None,fit_prior=True)   # 使用默认配置初始化朴素贝叶斯
nb_model.fit(X_train,Y_train.ravel())    # 利用训练数据对模型参数进行估计
nb_predicted_train = nb_model.predict(X_train)
#calculate the train accuracy of naive bayes
nb_train_accuracy = accuracy_score(Y_train.ravel(),nb_predicted_train.ravel())
#calculate the test acuracy of naive bayes
nb_predicted_test = nb_model.predict(X_test)
nb_test_accuracy = accuracy_score(Y_test.ravel(),nb_predicted_test.ravel())

#SVM
#支持向量机
svm1_model=LinearSVC(penalty='l2',loss='squared_hinge',dual=True,tol=0.0001,C=1.0,multi_class='ovr',fit_intercept=True,intercept_scaling=1,class_weight=None,verbose=0,random_state=None,max_iter=1000)
svm1_model.fit(X_train,Y_train.ravel())#利用训练数据对模型参数进行估计
svm1_predicted_train = svm1_model.predict(X_train)
#calculate the train accuracy of SVM
svm1_train_accuracy = accuracy_score(Y_train.ravel(),svm1_predicted_train.ravel())
#calculate the test acuracy of SVM
svm1_predicted_test = svm1_model.predict(X_test)
svm1_test_accuracy = accuracy_score(Y_test.ravel(),svm1_predicted_test.ravel())

#use SVM with another group of parameters
#使用SVM，但更换一组参数，以下是参数说明
#penalty : string, ‘l1’ or ‘l2’ (default=’l2’)
#   指定惩罚中使用的规范。 'l2'惩罚是SVC中使用的标准。 'l1'导致稀疏的coef_向量。
#loss : string, ‘hinge’ or ‘squared_hinge’ (default=’squared_hinge’)
#   指定损失函数。 “hinge”是标准的SVM损失（例如由SVC类使用），而“squared_hinge”是hinge损失的平方。
#dual : bool, (default=True)
#   选择算法以解决双优化或原始优化问题。 当n_samples> n_features时，首选dual = False。
#tol : float, optional (default=1e-4)
#   公差停止标准
#C : float, optional (default=1.0)
#   错误项的惩罚参数
#multi_class : string, ‘ovr’ or ‘crammer_singer’ (default=’ovr’)
#   如果y包含两个以上的类，则确定多类策略。 “ovr”训练n_classes one-vs-rest分类器，而“crammer_singer”优化所有类的联合目标。 虽然crammer_singer在理论上是有趣的，因为它是一致的，但它在实践中很少使用，因为它很少能够提高准确性并且计算成本更高。 如果选择“crammer_singer”，则将忽略选项loss，penalty和dual。
#fit_intercept : boolean, optional (default=True)
#   是否计算此模型的截距。 如果设置为false，则不会在计算中使用截距（即，预期数据已经居中）。
#intercept_scaling : float, optional (default=1)
#   当self.fit_intercept为True时，实例向量x变为[x，self.intercept_scaling]，即具有等于intercept_scaling的常量值的“合成”特征被附加到实例向量。 截距变为intercept_scaling *合成特征权重注意！ 合成特征权重与所有其他特征一样经受l1 / l2正则化。 为了减小正则化对合成特征权重（并因此对截距）的影响，必须增加intercept_scaling。
#class_weight : {dict, ‘balanced’}, optional
#   将类i的参数C设置为SVC的class_weight [i] * C. 如果没有给出，所有课程都应该有一个重量。 “平衡”模式使用y的值自动调整与输入数据中的类频率成反比的权重，如n_samples /（n_classes * np.bincount（y））
#verbose : int, (default=0)
#   启用详细输出。 请注意，此设置利用liblinear中的每进程运行时设置，如果启用，可能无法在多线程上下文中正常工作。
#random_state : int, RandomState instance or None, optional (default=None)
#   在随机数据混洗时使用的伪随机数生成器的种子。 如果是int，则random_state是随机数生成器使用的种子; 如果是RandomState实例，则random_state是随机数生成器; 如果为None，则随机数生成器是np.random使用的RandomState实例。
#max_iter : int, (default=1000)
#   要运行的最大迭代次数
svm2_model=LinearSVC(penalty='l2',loss='hinge',dual=True,tol=0.001,C=0.1,multi_class='ovr',fit_intercept=True,intercept_scaling=1,class_weight=None,verbose=0,random_state=None,max_iter=10000)
svm2_model.fit(X_train,Y_train.ravel())#利用训练数据对模型参数进行估
svm2_predicted_train = svm2_model.predict(X_train)
#calculate the train accuracy of SVM
svm2_train_accuracy = accuracy_score(Y_train.ravel(),svm2_predicted_train.ravel())
#calculate the test acuracy of SVM
svm2_predicted_test = svm2_model.predict(X_test)
svm2_test_accuracy = accuracy_score(Y_test.ravel(),svm2_predicted_test.ravel())

print('Training accuracy of logistic regression: %0.2f%%' % (logistic_train_accuracy*100))
print('Testing accuracy of logistic regression: %0.2f%%' % (logistic_test_accuracy*100))

print('Training accuracy of naive bayes: %0.2f%%' % (nb_train_accuracy*100))
print('Testing accuracy of naive bayes: %0.2f%%' % (nb_test_accuracy*100))

print('Training accuracy of SVM: %0.2f%%' % (svm1_train_accuracy*100))
print('Testing accuracy of SVM: %0.2f%%' % (svm1_test_accuracy*100))

print('Training accuracy of SVM with another parameters: %0.2f%%' % (svm2_train_accuracy*100))
print('Testing accuracy of SVM with anothor parameters: %0.2f%%' % (svm2_test_accuracy*100))