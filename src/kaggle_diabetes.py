import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
plt.style.use('ggplot')


# ********************************* Preparando a leitura do Dataset ******************
# diretorio do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')

# Escolhendo o arquivo .csv
files_names = [i for i in os.listdir(DATA_DIR) if i == 'pima-data.csv']

for i in files_names:
    df = pd.read_csv(os.path.join( DATA_DIR, i ))

# Apresentando o dataset
print('\n**************** Informações sobre o Dataset **********************************\n')
print('Diretórios: \n')
print('Projeto: ', BASE_DIR)
print('Dataset: ', DATA_DIR)
print('Cabeçalho do Dataframe: \n',df.head())

# convertendo target onde True = 1(paciente) e False = 0 (não-diabetes(controle))
map_data = {True : 1, False : 0}
df['diabetes'] = df['diabetes'].map(map_data)
print('\nAlteração de valore categóricos dos targets: \n',df.head())


# ******************************** Tratando os dados ***********************************
# Quantidade de valores por classe
vl_paciente = len(df.loc[df['diabetes'] == 1])
vl_controle = len(df.loc[df['diabetes'] == 0])

# Quem são as amostras por classes
samples0 = np.where(df.diabetes == 0)
samples1 = np.where(df.diabetes == 1)


# Creates a list of predicting time and accuracies
accuracy_LR = []
accuracy_PC = []
accuracy_RF = []
accuracy_NB = []
accuracy_DT = []
accuracy_SVM = []

# Verificando os colunas com valores = 0
print('\n********* Tratando valores iguais a 0 *******************')
print('\nColunas com valores = 0:\n',(df==0).sum())
print('\nQuantidade de valores = 0 Pacientes: ', vl_paciente)
print('\nQuantidade de valores = 1 Controle: ', vl_controle)
print('\nQuantidade de Outliers de Insulina:', len(df[df['insulin'] > 200]))
print('\nQuantidade de Outliers da Idade:', len(df[df['age'] > 67]))


# divisão entre as características e as classes a serem classificadas
dt_feature = df.iloc[:,:-1]
dt_target = df.iloc[:,-1]

# atribuindo o valor da média para os valores = 0
dt_feature = dt_feature.mask(dt_feature == 0).fillna(dt_feature.mean())
print('\nCaracterísticas com valores = 0 alteradas para valores de média:\n',dt_feature.head())


# **************************** Funções ***************************************

def information():
    print('\nO conjunto de dados possui %d linhas e %d colunas para: ' %(len(df[:]), len(df.columns)))
    print('     %d pacientes, que correspondem a %.2f%% do conjunto de dados' %(vl_paciente, vl_paciente / (vl_paciente + vl_controle)*100))
    print('     %d controles, que correspondem a %.2f%% do conjunto de dados' %(vl_controle, vl_controle / (vl_controle + vl_paciente)*100))
    print('\nValores faltantes: ', df.isnull().values.any())
    #print('\nValor de correlação: ',df.corr())
    #print('Valor: ',df.mask(df==0).fillna(df.mean()))


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

def plot_hist():
    plt.hist(df.iloc[:,-1], color='b',width=.1)
    plt.xlabel('Classe das amostras')
    plt.ylabel('Quantidade de amostras')
    plt.title('Histograma de Classes')
    plt.show()

def bolxplot():
    #ax = sns.boxplot(x=df.diabetes, y=df.age, data=df)
    green_diamond = dict(markerfacecolor='g', marker='D')
    ax = sns.boxplot(data=dt_feature['age'], orient="v", palette="Set2", flierprops=green_diamond)
    plt.show()


 # ******************************** funções de preparação do Modelo ********************   

def formatarTimer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return seconds

def split_model():
    print('\n**************** Resultados **********************************\n')
    rould = 0.10
    epochs = 1
    # Iterates over a set of runnings
    for i in range(1):
        X_train, X_test, y_train, y_test = train_test_split(dt_feature, dt_target, test_size=0.3, random_state=i)
        print('Divisão do conjunto de dados:\n')
        print('X_train: %d\ny_train: %d\nX_test: %d\ny_test: %d\n' %(len(X_train), len(y_train), len(X_test), len(y_test)))


        # Regressão Logistica
        lr = LogisticRegression(random_state=i).fit(X_train, y_train)
        lr_predictions = lr.predict(X_test)
        acc_lr = lr.score(X_test, y_test)
        
        
        # Perceptron
        percep = Perceptron(random_state=i)
        percep.fit(X_train, y_train)
        percep.predictions = percep.predict(X_test)
        acc_percep = percep.score(X_test, y_test)


        # Randon Forest
        rf = RandomForestClassifier(random_state=i)      # Create random forest object
        rf.fit(X_train, y_train)
        rf_predictions = rf.predict(X_test)
        acc_rf = rf.score(X_test, y_test)


        # Naive Bayes
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        gnb_predictions = gnb.predict(X_test)
        acc_nb = gnb.score(X_test, y_test)
            
            
            
        # Decision Tree
        dt = DecisionTreeClassifier(random_state=i, criterion='entropy', max_depth=3)
        dt.fit(X_train, y_train)
        dt_predictions = dt.predict(X_test)
        acc_dt = dt.score(X_test, y_test)
            

        # SVM
        clf = SVC(kernel='linear')
        clf = clf.fit(X_train, y_train)
        clf_pred = clf.predict(X_test)
        acc_clf = clf.score(X_test, y_test)

        # SVM-RBF
        #param_grid = {'C':[1, 5, 10, 50, 100, 500, 1000, 5000], 'gamma':[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
        #rbf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
        #rbf = rbf.fit(X_train, y_train)
        #rbf_pred = rbf.predict(X_test)
        #acc_rbf = rbf.score(X_test, y_test)
        #print('Best estimator found by grid search: ')
        #print(rbf.best_estimator_)
        #print('Best Score: ',rbf.best_score_)
            
        # Accuracy
        accuracy_LR.append(acc_lr)
        accuracy_PC.append(acc_percep)
        accuracy_RF.append(acc_rf)
        accuracy_NB.append(acc_nb)
        accuracy_DT.append(acc_dt)
        #accuracy_SVM.append(acc_rbf)
        accuracy_SVM.append(acc_clf)



        print('\nResultados Regressão Linear:\nAcc_LR: ',acc_lr,'\nepoch: ',epochs)
        print(metrics.confusion_matrix(y_test, lr_predictions) )
        print("\nClassification Report:\n ", metrics.classification_report(y_test, lr_predictions))

        print('\nResultados Perceptron:\nAcc_PC: ',acc_percep,'\nepoch: ',epochs)
        print(metrics.confusion_matrix(y_test, percep.predictions) )
        print("\nClassification Report:\n ", metrics.classification_report(y_test, percep.predictions))

        print('\nResultados Random Forest:\nAcc_RF: %.4f\nepoch: %d\n' %(acc_rf,epochs))
        print(metrics.confusion_matrix(y_test, rf_predictions) )
        print("\nClassification Report:\n ", metrics.classification_report(y_test, rf_predictions))

        print('\nResultados Nayve Bayes:\nAcc_NB: ',acc_nb,'\nepoch: ',epochs)
        print(metrics.confusion_matrix(y_test, gnb_predictions) )
        print("\nClassification Report:\n ", metrics.classification_report(y_test, gnb_predictions))

        print('\nResultados Decision Tree:\nAcc_DT: ',acc_dt,'\nepoch: ',epochs)
        print(metrics.confusion_matrix(y_test, dt_predictions) )
        print("\nClassification Report:\n ", metrics.classification_report(y_test, dt_predictions))

        '''
        print('\nResultados Suport Vector Machine:\nAcc_SVM: ',acc_rbf,'\nepoch: ',epochs)   
        print(metrics.confusion_matrix(y_test, rbf_pred) )
        print("\nClassification Report:\n ", metrics.classification_report(y_test, rbf_pred)) 
        print('\n\n')
        '''

        print('\nResultados Suport Vector Machine:\nAcc_SVM: ',acc_clf,'\nepoch: ',epochs)   
        print(metrics.confusion_matrix(y_test, clf_pred) )
        print("\nClassification Report:\n ", metrics.classification_report(y_test, clf_pred)) 
        print('\n\n')

        
        rould+=0.10
        epochs+=1

        # matriz de confusão
        #cf_matrix = confusion_matrix(y_test, rbf_pred)
        #sns.heatmap(cf_matrix, annot=True, fmt='', cmap='Blues')
        #plt.show()

    print('Acuracia média Regressão Linear de %.2f%%.' %(np.mean(accuracy_LR)*100))
    print('Acuracia média Perceptron de %.2f%%.' %(np.mean(accuracy_PC)*100))
    print('Acuracia média Random Forest de %.2f%%.' %(np.mean(accuracy_RF)*100))
    print('Acuracia média Nayve Bayes de %.2f%%.' %(np.mean(accuracy_NB)*100))
    print('Acuracia média Decision Tree de %.2f%%.' %(np.mean(accuracy_DT)*100))
    print('Acuracia média Suport Vector Machine de %.2f%%.' %(np.mean(accuracy_SVM)*100))
    



#bolxplot()
#plot_corr(df, size=10)
information()
#correlation()
split_model()
