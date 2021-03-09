import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
plt.style.use('ggplot')



# diretorio do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')

# listando os arquivos .csv
#for i in files_names:
 #   print(i)

# Escolhendo o arquivo .csv
files_names = [i for i in os.listdir(DATA_DIR) if i == 'pima-data.csv']

for i in files_names:
    df = pd.read_csv(os.path.join( DATA_DIR, i ))


# convertendo target onde 1 = diabetes(paciente) e 0 = não-diabetes(controle)
map_data = {True : 1, False : 0}
df['diabetes'] = df['diabetes'].map(map_data)

vl_paciente = len(df.loc[df['diabetes'] == 1])
vl_controle = len(df.loc[df['diabetes'] == 0])

# Creates a list of training time, predicting time and accuracies
train_time_NB, predict_time_NB, accuracy_NB = [], [], []
train_time_DT, predict_time_DT, accuracy_DT = [], [],[]
train_time_SVM, predict_time_SVM, accuracy_SVM = [],[],[]



# **************************** Funções ***************************************

def information():
    print('\n**************** Informações sobre o Dataset **********************************\n')
    print('Diretórios: \n')
    print('Projeto: ', BASE_DIR)
    print('Dataset: ', DATA_DIR)
    print('\nSamples and Features:\n')
    print(df.head())
    print('\nO conjunto de dados possui %d linhas e %d colunas para: ' %(len(df[:]), len(df.columns)))
    print('     %d pacientes, que correspondem a %.2f%% do conjunto de dados' %(vl_paciente, vl_paciente / (vl_paciente + vl_controle)*100))
    print('     %d controles, que correspondem a %.2f%% do conjunto de dados' %(vl_controle, vl_controle / (vl_controle + vl_paciente)*100))
    print('\nValores faltantes: ', df.isnull().values.any())
    print('\nValor de correlação: ',df.corr())

def formatarTimer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return seconds



# divisão entre as características e as classes a serem preditas
def split_model():
    print('\n**************** Resultados **********************************\n')
    dt_feature = df.iloc[:,:-1]
    dt_target = df.iloc[:,-1]

    rould = 0.10
    epochs = 1
    # Iterates over a set of runnings
    for i in range(9):
        X_train, X_test, y_train, y_test = train_test_split(dt_feature, dt_target, test_size=0.3, random_state=i)
        

        # Naive Bayes
        start = time.time()
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        train_time_NB.append(formatarTimer(start, time.time()))
        start = time.time()
        gnb_predictions = gnb.predict(X_test)
        predict_time_NB.append(formatarTimer(start, time.time()))
        acc_nb = gnb.score(X_test, y_test)
        
        
        
        # Decision Tree
        start = time.time()
        dt = DecisionTreeClassifier(random_state=i, criterion='entropy', max_depth=3)
        dt.fit(X_train, y_train)
        train_time_DT.append(formatarTimer(start, time.time()))
        start = time.time()
        dt_predictions = dt.predict(X_test)
        predict_time_DT.append([formatarTimer(start, time.time())])
        acc_dt = dt.score(X_test, y_test)
        
        
        # SVM-RBF
        start = time.time()
        param_grid = {'C':[1, 5, 10, 50, 100, 500, 1000, 5000], 'gamma':[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
        rbf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
        rbf = rbf.fit(X_train, y_train)
        #rbf = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo')
        #rbf.fit(X_train, Y_train)
        train_time_SVM.append(formatarTimer(start, time.time()))
        start = time.time()
        rbf_pred = rbf.predict(X_test)
        predict_time_SVM.append(formatarTimer(start, time.time()))
        acc_rbf = rbf.score(X_test, y_test)
        print('Best estimator found by grid search: ')
        print(rbf.best_estimator_)
        print('Best Score: ',rbf.best_score_)
        
        # Accuracy
        accuracy_NB.append(acc_nb)
        accuracy_DT.append(acc_dt)
        accuracy_SVM.append(acc_rbf)
        
         
        print('\nResultados Nayve Bayes:\nAcc_NB: ',acc_nb,'\nepoch: ',epochs,'\ntempo de treinamento: ',train_time_NB)
        print('\nResultados Decision Tree:\nAcc_DT: ',acc_dt,'\nepoch: ',epochs,'\ntempo de treinamento: ',train_time_DT)
        print('\nResultados Suport Vector Machine:\nAcc_SVM: ',acc_rbf,'\nepoch: ',epochs,'\ntempo de treinamento: ',train_time_SVM)    
        print('\n\n')
        rould+=0.10
        epochs+=1


    print('Acuracia média Nayve Bayes de %.2f%% com tempo médio de %.5f segundos para treinar.' \
        %(np.mean(accuracy_NB)*100, np.mean(train_time_NB)))
    print('Acuracia média Decision Tree de %.2f%% com tempo médio de %.5f segundos para treinar.' \
        %(np.mean(accuracy_DT)*100, np.mean(train_time_DT)))
    print('Acuracia média Suport Vector Machine de %.2f%% com tempo médio de %.5f segundos para treinar.' \
        %(np.mean(accuracy_SVM)*100, np.mean(train_time_SVM)))
    

# Feature selection
def feat_select():
    model = LogisticRegression(max_iter=2000)
    from sklearn.feature_selection import RFE
    rfe = RFE(model, 4)
    fit = rfe.fit(dt_feature, dt_target)
    # Mostrando o número de features:
    print ("Número de features = %d"%(fit.n_features_))  


# **************************** Plots ***************************************
def correlation():
    print('Matriz de correlação: \n')
    plot_corr(df)
    plt.show()

def plot_corr(df, size=10):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()




#plot_corr(df, size=10)
information()
#correlation()
#split_model()
#feat_select()