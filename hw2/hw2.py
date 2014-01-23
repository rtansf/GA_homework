from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn import datasets

def load_iris_data() :
   iris = datasets.load_iris()
   return (iris.data, iris.target, iris.target_names) 


def cross_validate(XX, yy, classifier, k_fold, hyperparam) :
   k_fold_indices = KFold(len(XX), n_folds=k_fold, indices=True)

   k_score_total = 0
   for train_slice, test_slice in k_fold_indices :
       model = classifier(XX[[train_slice]],
                          yy[[train_slice]], hyperparam)
      
       k_score = model.score(XX[[test_slice]],
                             yy[[test_slice]])

       k_score_total += k_score
   
   return k_score_total * 1.0 / k_fold

def cross_validate_2(XX, yy, classifier, k_fold, hyperparam) :
   k_fold_indices = KFold(len(XX), n_folds=k_fold, indices=True)

   k_score_total = 0
   for train_slice, test_slice in k_fold_indices :
       model = classifier(XX[[train_slice]],
                          yy[[train_slice]], hyperparam)
      
       k_score = model.score(XX[[test_slice]],
                             yy[[test_slice]])

       k_score_total += k_score
   
   return k_score_total * 1.0 / k_fold

def knn(X_train, y_train, hyperparam) :

  # method returns a KNN object with methods:
  #   predict(X_classify)
  #   score (X_test, y_test)
  clf = KNeighborsClassifier(n_neighbors=3)
  clf.fit (X_train, y_train)
  return clf

def nb(X_train, y_train, hyperparam) :
   gnb = GaussianNB()
   clf = gnb.fit(X_train, y_train)
   return clf


def lr(X_train, y_train, hyperparam):
    # funtion returns an LR object
    #  useful methods of this object for this exercise:                                                                                                                    
    #   fit(X_train, y_train) --> fit the model using a training set                                                                                                       
    #   predict(X_classify) --> to predict a result using the trained model                                                                                                
    #   score(X_test, y_test) --> to score the model using a test set
    
    clf = LinearRegression()
    clf.fit(X_train, y_train)

    return clf

def logit(X_train, y_train, hyperparam) :
    clf = LogisticRegression(C=hyperparam)
    clf.fit(X_train, y_train)
    
    return clf

