import pandas as pd
import numpy as numpy
import os
import matplotlib.pyplot as plt
import seaborn as sns



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


def information():
    print('\n**************** Informações sobre o Dataset **********************************\n')
    print('Diretórios: \n')
    print('Projeto: ', BASE_DIR)
    print('Dataset: ', DATA_DIR)
    print('\nSamples and Eeatures:\n')
    print(df.head())
    print('Conjunto de dados com %d linhas e %d colunas.' %(len(df[:]), len(df.columns)))
    print('\nDiagnóstico por classe:\n', df['diabetes'].value_counts())
    print('Valores faltantes: ', df.isnull().values.any())

# convertendo target    
map_data = {True : 1, False : 0}
df['diabetes'] = df['diabetes'].map(map_data)
#print(df.head())


def plot_corr(df, size=10):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)



def correlation():
    print('Valor de correlação: \n')
    df.corr()

    print('Matriz de correlação: \n')
    plot_corr(df)
    plt.show()


information()
correlation()