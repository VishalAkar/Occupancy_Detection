import numpy as np
import pandas as pd

# Function of Applying Decision Tree Classifier

def DTC (Param1_train,Param2_train,Param1_test,Param2_test,Param1_valid,Param2_valid):


    #Importing Decision Tree classifier
    from sklearn.tree import DecisionTreeClassifier as DC
    depth = list(np.arange(1,101))
    sample_leaf = list(np.arange(1,11))

    #hyper parameter tuning using grid search
    tuned_parameter = [{'max_depth':depth,'min_samples_leaf':sample_leaf}]
    from sklearn.model_selection import GridSearchCV as Gs
    classifier = Gs(DC(random_state = 0),tuned_parameter)

    #classifier training
    classifier.fit(Param1_train,Param2_train)
    predict1 = classifier.predict(Param1_test)

    # using confusion matrix
    from sklearn.metrics import confusion_matrix
    predict1 = predict1.transpose()
    cm = confusion_matrix(Param2_test,predict1)
    print(cm)
    accuracy = np.trace(cm)/float(np.sum(cm))
    print("The acuracy from confusion matrix",accuracy)
    best_param = classifier.best_params_

    print("The best accuracy at testing is: ",classifier.best_score_," at max depth:",best_param['max_depth']," and min sample leaves:",best_param['min_samples_leaf'])

    # validation
    from sklearn.cross_validation import cross_val_score
    score = cross_val_score(DC(max_depth = best_param['max_depth']), Param1_valid,Param2_valid, cv = 4 ,n_jobs = 1 )
    print("The accuracy score during validation is",score.mean())
    print("\n\n")


#Creating Function for Support Vector Machine
def SVM(Param1_train,Param2_train,Param1_test,Param2_test,Param1_valid,Param2_valid):


    #importing SVM from sklearn
    from sklearn.svm import SVC
    tuned_parameter = [{'kernel':['linear']},{'kernel':['rbf']}]

    #Hyperparameter tuning
    from sklearn.model_selection import GridSearchCV as Gs

    #classifier training
    classifier = Gs(SVC(random_state = 0), tuned_parameter)
    classifier.fit(Param1_train,Param2_train)
    predict1 = classifier.predict(Param1_test)

    from sklearn.metrics import confusion_matrix
    predict1 = predict1.transpose()
    cm = confusion_matrix(Param2_test,predict1)
    print(cm)
    accuracy = np.trace(cm)/float(np.sum(cm))
    print("The acuracy from confusion matrix",accuracy)
    best_param = classifier.best_params_

    best_param = classifier.best_params_

    print("The best accuracy at testing  is: ",classifier.best_score_," by Kernel : ",best_param['kernel'])

    #Validation
    from sklearn.cross_validation import cross_val_score
    score = cross_val_score(SVC(kernel = '%s'%best_param['kernel']), Param1_valid,Param2_valid, cv = 5 ,n_jobs = 1)
    print("The accuracy during validation is : ",score.mean())
    print('\n\n')
   

#Creating the function for Logistic Regression

def LR(Param1_train,Param2_train,Param1_test,Param2_test,Param1_valid,Param2_valid):


    #Importing Logistic regression classifier from sklearn

    c = list(np.arange(1,101))
    from sklearn.linear_model import LogisticRegression as LR
    tuned_parameters = [{'C':c , 'penalty':['l1']},{'C':c,'penalty':['l2']},{'C':c}]
        
    #Hyper tuning using Grid_search_CV
    from sklearn.model_selection import GridSearchCV as Gs
    classifier = Gs(LR(random_state = 0),tuned_parameters)

    #Training
    classifier.fit(Param1_train,Param2_train)
    predict1 = classifier.predict(Param1_test)

    from sklearn.metrics import confusion_matrix
    predict1 = predict1.transpose()
    cm = confusion_matrix(Param2_test,predict1)
    print(cm)
    accuracy = np.trace(cm)/float(np.sum(cm))
    print("The acuracy from confusion matrix",accuracy)
    best_param = classifier.best_params_

    best_param = classifier.best_params_

    print("The best accuracy at testing  is: ",classifier.best_score_," at C : ",best_param['C'],"at penalty: ",best_param['penalty'])

    #Validation
    from sklearn.cross_validation import cross_val_score
    score = cross_val_score(LR(random_state = 0, penalty = '%s'%best_param['penalty'] , C = best_param['C']) , Param1_valid,Param2_valid, cv = 5 ,n_jobs = 1)
    print("Accuracy at Validation is : ",score.mean())
#Importing Training data from the file 
data_training = pd.read_csv('datatraining.csv')

#Remoing unwanted columns from the data
data_training = data_training.drop(['S.no','date'], axis = 1)

#Splitting data from Training purpose
Param1_train = data_training.iloc[:,0:len(data_training.columns)-1]
Param2_train = data_training.iloc[:,len(data_training.columns)-1]


#Importing Test data test 1 from the file
data_test1 = pd.read_csv('datatest.csv')

#Removing garbage data
data_test1 = data_test1.drop(['S.no','date'], axis = 1)

#Ssplitting data for testing purpose
Param1_test = data_test1.iloc[:,0:len(data_test1.columns)-1]
Param2_test = data_test1.iloc[:,len(data_test1.columns)-1]

#Importing Validation data from the file
data_validation = pd.read_csv('datatest2.csv')

#Removing Garbage data
data_validation = data_validation.drop(['S.no','date'], axis = 1)

#Splitting data for validation purpose
Param1_valid = data_validation.iloc[:,0:len(data_validation.columns)-1]
Param2_valid = data_validation.iloc[:,len(data_validation.columns)-1]

#Calling Decision tree classifier Function


print("Using Decision Tree Classifier")
DTC(Param1_train,Param2_train,Param1_test,Param2_test,Param1_valid,Param2_valid)
print("Using Simple Vector Machine")
SVM(Param1_train,Param2_train,Param1_test,Param2_test,Param1_valid,Param2_valid)

print("Using Logistic Regression")
LR(Param1_train,Param2_train,Param1_test,Param2_test,Param1_valid,Param2_valid) 

