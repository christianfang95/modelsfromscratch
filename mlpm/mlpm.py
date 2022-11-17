#Import required packages
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# simulate dataset
def simulate_data(classes = 3):
  X, y = make_classification(n_samples = 1000, 
                             n_features = 10, 
                             n_informative = 3, 
                             n_redundant = 7, 
                             n_classes = classes, 
                             random_state = 1)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
  return(X_train, X_test, y_train, y_test)

#Create MLPM

def mlpm(X_train, X_test, y_train, y_test):
  """This function fits multinomial linear probability models on the test data, 
  and gets predictions for the training data
  """
  lpm = OneVsRestClassifier(LinearRegression()).fit(X_train, y_train)
  y_pred = lpm.predict(X_test)
  lpm_report = classification_report(y_test, y_pred, output_dict = True)
  lpm_report = pd.DataFrame(lpm_report).transpose()
  return lpm_report

#Create multinomial logistic regression

def multinom(X_train, X_test, y_train, y_test):
  """This function fits multinomial logistic regression on the test data, 
  and gets predictions for the training data
  """
  multinom = OneVsRestClassifier(LogisticRegression(multi_class = "multinomial")).fit(X_train, y_train)
  y_pred = multinom.predict(X_test)
  multinom_report = classification_report(y_test, y_pred, output_dict = True)
  multinom_report = pd.DataFrame(multinom_report).transpose()
  return multinom_report

#Create KNN Classifier

def knn(X_train, X_test, y_train, y_test, classes):
  """This function fits KNN on the test data, 
  and gets predictions for the training data
  """
  knn = KNeighborsClassifier(n_neighbors = classes)
  knn.fit(X_train, y_train)
  y_pred = knn.predict(X_test)
  knn_report = classification_report(y_test, y_pred, output_dict = True)
  knn_report = pd.DataFrame(knn_report).transpose()
  return knn_report

#Define plot function
def plot(lpm, log, knn):
  """This function plots the F1 scores per class and averaged for all three models"""
  fig, ax = plt.subplots(nrows=1, ncols=1)
  fig.suptitle('F1 scores of multinomial regressions and KNN')
  
  #Set line style and line width
  ls = "-"
  lw = 2.5

  #Add lines for the 3 models
  plt.plot(knn.index[:-2], knn['f1-score'][0:4], marker = 'o', color = 'g',
              linewidth=lw, linestyle=ls)
  plt.plot(log.index[:-2], log['f1-score'][0:4], marker = 'o', color = 'b',
              linewidth=lw, linestyle=ls)
  plt.plot(lpm.index[:-2], lpm['f1-score'][0:4], marker = 'o', color = 'r', 
              linewidth=lw, linestyle=ls)

  
  #Set axes title, label, and legend
  ax.set_ylabel('F1 score')
  ax.set_xlabel('Class')
  ax.legend(('KNN', 'multinomial \nlogistic regression', 'MLPM'))

  #Plot formatting
  plt.xticks(['0', '1', '2', 'accuracy'], ['Britney', 'Taylor', 'Cher', 'mean'])
  plt.ylim([0, 1])
  plt.show()

#Define main function
def main():
  """This function calculates the three models and plots the results"""
  classes = 3
  X_train, X_test, y_train, y_test = simulate_data(classes = classes)
  mod1 = mlpm(X_train, X_test, y_train, y_test)
  mod2 = multinom(X_train, X_test, y_train, y_test)
  mod3 = knn(X_train, X_test, y_train, y_test, classes = classes)
  plot(mod1, mod2, mod3)
  plt.show()

#Run everything
main()  