import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
df=pd.read_csv("C:/Users/Jeevan/Downloads/datamid1.csv")
feature_col_names=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabeticPedigreeFunction','Age']
predicted_class_names=['Outcome']
X=df[feature_col_names].values#thesearefactorsfortheprediction
y=df[predicted_class_names].values#thisiswhatwewanttopredict
#splittingthedatasetintotrainandtestdata
xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.33)
print('\nthetotalnumberofTrainingData:',ytrain.shape)
print('\nthetotalnumberofTestData:',ytest.shape)
#TrainingNaiveBayes(NB)classifierontrainingdata.
clf=GaussianNB().fit(xtrain,ytrain.ravel())
predicted=clf.predict(xtest)
predictTestData=clf.predict([[6,148,72,35,0,33.6,0.627,50]])
#printingConfusionmatrix,accuracy,PrecisionandRecall
print('\nConfusionmatrix')
print(metrics.confusion_matrix(ytest,predicted))
print('\nAccuracyoftheclassifieris',metrics.accuracy_score(ytest,predicted))
print('\nThevalueofPrecision',metrics.precision_score(ytest,predicted))
print('\nThevalueofRecall',metrics.recall_score(ytest,predicted))
print("PredictedValueforindividualTestData:",predictTestData)
