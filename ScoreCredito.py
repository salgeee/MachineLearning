# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split # Utilizado para separar dados de treino e teste
from sklearn.preprocessing import StandardScaler # Utilizado para fazer a normalização dos dados
from sklearn.preprocessing import MinMaxScaler # Utilizado para fazer a normalização dos dados
from sklearn.preprocessing import LabelEncoder # Utilizado para fazer o OneHotEncoding
from sklearn.linear_model import LinearRegression # Algoritmo de Regressão Linear
from sklearn.metrics import r2_score # Utilizado para medir a acuracia do modelo preditivo


#Comando para exibir todas colunas do arquivo
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# %%
#IMPORTAÇÃO DOS DADOS, ANÁLISE EXPLORATORIA E TRATAMENTO DOS DADOS

# %%
#Carregar arquivo e armaneza-lo na váriavel
df_dados = pd.read_excel("C:\\Users\\User\\Documents\\GitHub\\DataTestsBlips\\ScoreCreditoML\\dados_credito.xlsx")
# %%
# Verifica a quantidade de linhas e colunas
df_dados.shape
# %%
# Comando utilizado para verificar as linhas inicias (5)
df_dados.head()
# %%
#Comando para verificar as ultimas (5) linhas
df_dados.tail()
# %%
# Comando para verificar o tipo das colunas (string, obj, int)
df_dados.info()
# %%
# Excluir a variavel(coluna) codigo_cliente para podermos agrupar os valores e indetificar se há algum valor discrepante
df_dados.drop('CODIGO_CLIENTE', axis=1, inplace=True)
# %%
# No agrupamento podemos verificar que tem um dado "SEM DADOS"
df_dados.groupby(['ULTIMO_SALARIO']).size()
# %%
# Existem duas formas para tratamento de dados nesse caso
#Excluindo o registro ou substituir para um valor médio
#df_dados.drop(df_dados.loc[df_dados['VALOR']=='SEM VALOR'].index, inplace=True) **Para exclusão**

df_dados.loc[df_dados['ULTIMO_SALARIO'] == 'SEM DADOS']
# %%
#Aqui vamos substituir sem dados para um valor nulo
df_dados.replace('SEM DADOS', np.nan, inplace=True)
# %%
#Aqui vamos transformar para float
df_dados['ULTIMO_SALARIO'] = df_dados['ULTIMO_SALARIO'].astype(np.float64)
# %%
# Vamos verificar se tem algum valor NAN (Not Available, Nulo)
df_dados.isnull().sum()
# %%
# Aqui vamos atualiazar o valor do ultimo salario de acordo com a mediana do modelo
df_dados['ULTIMO_SALARIO'] = df_dados['ULTIMO_SALARIO'].fillna((df_dados['ULTIMO_SALARIO'].median()))
# %%
# Vamos confirmar se tem algum valor NAN (Not Available, Nulo)
df_dados.isnull().sum()
# %%
#Vamos avaliar novamente os tipos
df_dados.info()
# %%
## Comando para trazer medidas estatisticas como: mediana, média, 25%, 50%, 75%, max, min
df_dados.describe()
# %%

## Agora iremos avaliar os outliers númericos


#lista para carregar float e int
variaveis_numericas = []
for i in df_dados.columns[0:16].tolist():
  if df_dados.dtypes[i] == 'int64' or df_dados.dtypes[i] == 'float64':
    print(i, ':', df_dados.dtypes[i])
    variaveis_numericas.append(i)
# %%
## Chamar a lista para verificar outliers no boxplot
variaveis_numericas
# %%
# Comando para exibir grafico de todas as colunas de uma vez para facilitar analise

#Definir tamanho da tela para ser exibido

plt.rcParams["figure.figsize"] = [15.00, 12.00]
plt.rcParams["figure.autolayout"] = True

# Definir linhas e colunas
f, axes = plt.subplots(2,5) # 2 linhas e 5 colunas

linha = 0
coluna = 0

for i in variaveis_numericas:
  sns.boxplot(data = df_dados, y=i,ax=axes[linha][coluna])
  coluna +=1
  if coluna == 5:
    linha += 1
    coluna = 0

plt.show()
# %%
# Agora é possível analisar que temos possíveis outliers nas variáveis QT_FILHOS, QT_IMOVEIS, VALOR_TABELA-CARROS E OUTRA_RENDA_VALOR
#Verificar os outliers para avaliar como trata-lós

df_dados.loc[df_dados['QT_FILHOS'] > 4]
# %%
# Como podemos analisar é improvável que alguem tenha 38 filhos, como são apenas 2 registros vamos excluir

df_dados.drop(df_dados.loc[df_dados['QT_FILHOS']> 4].index, inplace=True)
# %%
# Verificando o restante
df_dados.groupby(['OUTRA_RENDA_VALOR']).size()
# %%
# Nenhuma alteração necessária
df_dados.groupby(['QT_IMOVEIS']).size()

# %%
# Nenhuma alteração necessária
df_dados.groupby(['VALOR_TABELA_CARROS']).size()

# %%
#Vamos fazer um histograma para avaliar distribuição de dados
#Podemos avaliar que estão bem dispersos


#Tamanho da tela
plt.rcParams["figure.figsize"] = [15.00, 12.00]
plt.rcParams["figure.autolayout"] = True

f, axes = plt.subplots(4,3) # 4linhas e 3 colunas

linha = 0
coluna = 0

for i in variaveis_numericas:
  sns.histplot(data = df_dados, x=i,ax=axes[linha][coluna])
  coluna +=1
  if coluna == 3:
    linha += 1
    coluna = 0
    
plt.show()

# %%
#Gráfico de dispersão
sns.lmplot(x = "VL_IMOVEIS", y = "SCORE", data = df_dados);
# %%
sns.lmplot(x = "ULTIMO_SALARIO", y = "SCORE", data = df_dados);
# %%
sns.lmplot(x = "TEMPO_ULTIMO_EMPREGO_MESES", y = "SCORE", data = df_dados);
# %%

#Criar novo campo de faixa etaria

print('Menor Idade: ', df_dados['IDADE'].min())
print('Maior Idade: ', df_dados['IDADE'].max())
# %%
# Nova variavel para faixa etaria
idade_bins = [0, 30, 40, 50, 60]
idade_categoria = ["Até 30", "31 a 40", "41 a 50", "Maior que 50"]

df_dados["FAIXA_ETARIA"] = pd.cut(df_dados["IDADE"], idade_bins, labels=idade_categoria)

df_dados["FAIXA_ETARIA"].value_counts()
# %%
#Agrupar por faixa etaria o score
df_dados.groupby(["FAIXA_ETARIA"]).mean(["SCORE"])
# %%

#Vamos criar uma lista de variaveis categoricas

variaveis_categoricas = []
for i in df_dados.columns[0:48].tolist():
  if df_dados.dtypes[i] == 'object' or df_dados.dtypes[i] == 'category':
    print(i, ':', df_dados.dtypes[i])
    variaveis_categoricas.append(i)
    
# %%
# Vamos criar um gráfico de colunas de todas as variaveis categoricas

# Tamanho da tela para exibição dos gráficos
plt.rcParams["figure.figsize"] = [15.00, 22.00]
plt.rcParams["figure.autolayout"] = True

f, axes = plt.subplots(4,2) # 3 linhas e 2 colunas

linha = 0
coluna = 0
for i in variaveis_categoricas:
  sns.countplot(data = df_dados, x=i,ax=axes[linha][coluna])
  
  coluna += 1
  if coluna == 2:
    linha += 1
    coluna = 0
    
plt.show()
# %%

# cria o encoder (as strings viram números por exmeplo: Antes o UF era MG, agora ficara UF = 2(Apenas um exemplo))

lb = LabelEncoder()
#Aplica o encoder nas variaveis strings
df_dados['FAIXA_ETARIA'] = lb.fit_transform(df_dados['FAIXA_ETARIA'])
df_dados['OUTRA_RENDA'] = lb.fit_transform(df_dados['OUTRA_RENDA'])
df_dados['TRABALHANDO_ATUALMENTE'] = lb.fit_transform(df_dados['TRABALHANDO_ATUALMENTE'])
df_dados['ESTADO_CIVIL'] = lb.fit_transform(df_dados['ESTADO_CIVIL'])
df_dados['CASA_PROPRIA'] = lb.fit_transform(df_dados['CASA_PROPRIA'])
df_dados['ESCOLARIDADE'] = lb.fit_transform(df_dados['ESCOLARIDADE'])
df_dados['UF'] = lb.fit_transform(df_dados['UF'])

# Remove valores missing eventualmente gerados
df_dados.dropna(inplace = True)
# %%
df_dados.head(200)
# %%
df_dados.info()
# %%
#Heatmap
plt.rcParams["figure.figsize"] = (18,8)
ax = sns.heatmap(df_dados.corr(), annot=True)
# %%
# Decidimos qual é o nosso target (SCORE)
target = df_dados.iloc[:,15:16]

# %%
# Separando var preditoras
preditoras = df_dados.copy() 
del preditoras['SCORE'] #Exlcuindo a target porquê já foi armazenada
preditoras.head()
# %%
# Divimos os dados treino e teste
# test-size = 30% do nosso dados
#random_state é pra garantir a randomização da função sempre produza a mesma divisão dos dados etnre treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(preditoras,target,test_size = 0.3, random_state = 40)
# %%
#PADRONIZAÇÃO DE DADOS
sc = MinMaxScaler()
X_treino_normalizados = sc.fit_transform(X_treino)
X_teste_normalizados = sc.fit_transform(X_teste)
# %%

# Criar e avaliar o nosso modelo preditivo


# %%
# Treinar modelo
modelo = LinearRegression(fit_intercept = True)

modelo = modelo.fit(X_treino_normalizados, y_treino)
# %%
r2_score(y_teste, modelo.fit(X_treino_normalizados, y_treino).predict(X_teste_normalizados))
# %%

#Testando previsão de dados

UF = 2
IDADE = 42 
ESCOLARIDADE = 1
ESTADO_CIVIL = 2
QT_FILHOS = 1
CASA_PROPRIA = 1
QT_IMOVEIS = 1
VL_IMOVEIS = 300000
OUTRA_RENDA = 1
OUTRA_RENDA_VALOR = 2000 
TEMPO_ULTIMO_EMPREGO_MESES = 18 
TRABALHANDO_ATUALMENTE = 1
ULTIMO_SALARIO = 5400.0
QT_CARROS = 4
VALOR_TABELA_CARROS = 70000
FAIXA_ETARIA = 3

novos_dados = [UF, IDADE, ESCOLARIDADE, ESTADO_CIVIL, QT_FILHOS,CASA_PROPRIA,QT_IMOVEIS,VL_IMOVEIS,OUTRA_RENDA,
               OUTRA_RENDA_VALOR,TEMPO_ULTIMO_EMPREGO_MESES,TRABALHANDO_ATUALMENTE,ULTIMO_SALARIO,QT_CARROS,
               VALOR_TABELA_CARROS, FAIXA_ETARIA]


# Reshape
X = np.array(novos_dados).reshape(1, -1)
X = sc.transform(X)

# Previsão
print("Score de crédito previsto para esse cliente:", modelo.predict(X))
# %%
