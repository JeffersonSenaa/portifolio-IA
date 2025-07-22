# Portfólio 6 - Aprendizado de Máquina

## Introdução

O **Aprendizado de Máquina** (*Machine Learning* ou simplesmente ML) representa um pilar fundamental da **Inteligência Artificial (IA)**, onde concentra-se no desenvolvimento de algoritmos e técnicas que visam capacitar sistemas computacionais a extrair conhecimento e aprimorar seu desempenho a partir da **experiência em dados**. A partir disso, em vez de serem explicitamente programados para cada tarefa individual, essa metodologia tem se provado revolucionária, impulsionando avanços significativos em domínios diversos como a **classificação e reconhecimento de padrões** (e.g., imagens, voz), **sistemas de recomendação personalizados**, **detecção de anomalias** (e.g., fraudes financeiras), **diagnósticos médicos assistidos por IA**, **análise preditiva** em finanças e mercado, e o controle autônomo em **robótica**.

Para aprofundarmos neste tema, este portfólio tem como propósito explorar os conceitos basicos, as taxonomias principais e os algoritmos mais influentes dentro do campo do Aprendizado de Máquina. Buscamos não apenas elucidar os **fundamentos teóricos**, mas também diferenciar as **categorias primárias de aprendizado** (supervisionado, não supervisionado e por reforço) e demonstrar suas aplicações práticas através de implementações em **Python**, utilizando bibliotecas como **Scikit-learn** e **NumPy**. A compreensão desses elementos é crucial para o desenvolvimento e a implementação de soluções inteligentes no cenário tecnológico atual.

-----

## Fundamentos do Aprendizado de Máquina

### O que é Aprendizado de Máquina?

**Aprendizado de Máquina** é um subcampo da ciência da computação e da estatística que se dedica ao projeto e à construção de sistemas que podem aprender a partir da análise de dados. O cerne do ML é permitir que sistemas automatizados **melhorem iterativamente seu desempenho em uma tarefa específica** (*T*) com base na **experiência** (*E*) e considerando uma **métrica de desempenho** (*P*), sem que sejam explicitamente programados para cada cenário. Sendo assim, a essência não está em fornecer regras fixas, mas em capacitar o sistema a **identificar e generalizar padrões implícitos** a partir de grandes volumes de dados (Big Data), aplicando essas generalizações a novos e não vistos exemplos. Isso implica que o sistema pode adaptar o seu comportamento e previsões à medida que mais dados se tornarem disponíveis, ou que novos cenários sejam encontrados, refletindo uma capacidade de **autoaprendizagem**.

### Tipos de Aprendizado

A categorização dos algoritmos de Aprendizado de Máquina é fundamental no entendimento de suas aplicações e metodologias:

1.  **Aprendizado Supervisionado**
    Nesta modalidade, o modelo busca aprender a partir de um **conjunto de dados rotulado**, onde cada amostra de entrada (*feature vector*) associada a uma **saída desejada ou "rótulo"**. O objetivo é que o algoritmo consiga aprender o mapeamento da entrada para a saída, minimizando a diferença entre a previsão do modelo e o rótulo verdadeiro.

      * **Dados:** `(X, y)` onde `X` são as *features* de entrada e `y` são os rótulos de saída correspondentes.
      * **Finalidade:** Previsão de uma variável-alvo.
      * **Problemas Comuns:**
          * **Classificação:** A variável de saída é categórica (discreta). Exemplo: Classificação de e-mails como *spam* ou *não spam*, diagnóstico de doenças (doente/saudável), reconhecimento de objetos contidos em imagens.
          * **Regressão:** A variável de saída é contínua. Exemplo: Previsão de preços de imóveis com base nas características como número de quartos e localização, estimativa da temperatura ambiente, previsão de vendas futuras.
      * **Algoritmos Notáveis:**
          * **Regressão Linear:** Para previsões de valores contínuos, assume uma relação linear entre variáveis.
          * **Regressão Logística:** Apesar do nome, é um algoritmo de classificação binária que modela a probabilidade de uma instância pertencer a uma classe.
          * **Árvores de Decisão:** Modelos hierárquicos que particionam o espaço de entrada com base em regras de decisão.
          * **Random Forest:** Um *ensemble* de árvores de decisão que votam na saída, reduzindo overfitting e aumentando a robustez.
          * **Support Vector Machines (SVM):** Encontra um hiperplano ótimo que maximiza a margem de separação entre classes em problemas de classificação.

2.  **Aprendizado Não Supervisionado**
    Ao contrário do aprendizado supervisionado, os dados neste paradigma **não possuem rótulos pré-definidos**. O propósito deste é que o algoritmo identifique **padrões intrínsecos, estruturas ocultas ou relações latentes** nos dados, sem qualquer orientação externa.

      * **Dados:** Apenas `X` (vetores de *features* de entrada).
      * **Finalidade:** Descoberta de estruturas e padrões nos dados.
      * **Problemas Comuns:**
          * **Agrupamento (Clustering):** Organiza os dados em grupos (clusters) onde os elementos dentro de um grupo são mais similares entre si do que com elementos de outros grupos. Exemplo: Segmentação de clientes para estratégias de marketing, agrupamento de documentos por tópico.
          * **Redução de Dimensionalidade:** Simplifica a representação dos dados, projetando-os em um espaço de menor dimensão, mantendo a maior parte da variância. Útil para visualização e pré-processamento de dados. Exemplo: Compressão de dados, redução de ruído.
          * **Associação:** Descobrir as regras de associação entre variáveis em grandes bancos de dados. Exemplo: A Análise de cesta de compras (clientes que compram item A também compram item B).
      * **Algoritmos Notáveis:**
          * **K-Means:** Algoritmo iterativo de agrupamento que particiona `n` observações em `k` clusters.
          * **PCA (Análise de Componentes Principais):** Técnica que visa a redução de dimensionalidade que transforma as variáveis originais em um novo conjunto de variáveis não correlacionadas, chamadas componentes principais.
          * **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** Algoritmo de agrupamento, baseado em densidade capaz de descobrir clusters de formas arbitrárias e identificar ruído.

3.  **Aprendizado por Reforço**
    Neste tipo de aprendizado, um **agente de software** busca aprender a tomar decisões sequenciais em um **ambiente dinâmico** a fim de **maximizar uma medida de recompensa** cumulativa. O agente visa interagir com o ambiente, executando ações e recebendo **sinais de recompensa ou punição** como feedback, sem um conjunto de dados predefinido.

      * **Componentes Chave:** Agente, Ambiente, Estado, Ação, Recompensa, Política.
      * **Finalidade:** Aprender uma política ótima que mapeie estados a ações para maximizar a recompensa cumulativa.
      * **Exemplos:** Agentes de jogos (AlphaGo, xadrez), robótica (aprendizado de locomoção ou manipulação), sistemas de recomendação adaptativos que aprendem com o feedback do usuário.
      * **Algoritmos Notáveis:** Q-Learning, SARSA, Deep Q-Networks (DQN), PPO (Proximal Policy Optimization).

-----

## Principais Algoritmos de Aprendizado de Máquina

### 1\. Regressão Linear

A **Regressão Linear** é um dos algoritmos supervisionados mais básicos e fundamentais em problemas de **regressão**, ou seja, ele **prevê valores contínuos** baseados em uma ou mais variáveis independentes (features). Tem como objetivo encontrar a melhor reta (ou hiperplano em múltiplas dimensões) que se ajuste aos dados, minimizando a soma dos quadrados dos resíduos (diferença entre os valores observados e os valores previstos).

#### **Exemplo em Python:**

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt 

np.random.seed(0) 
X = 2 * np.random.rand(100, 1) 
y = 4 + 3 * X + np.random.randn(100, 1) 

# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instanciando e treinando o modelo de Regressão Linear
model = LinearRegression()
model.fit(X_train, y_train)

print(f"Coeficiente Angular (Beta_1): {model.coef_[0][0]:.2f}")
print(f"Intercepto (Beta_0): {model.intercept_[0]:.2f}")

# Previsões no conjunto de teste
y_pred = model.predict(X_test)

# Visualização da Regressão Linear
plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, label='Dados de Treino', alpha=0.6)
plt.scatter(X_test, y_test, label='Dados de Teste', alpha=0.8, color='orange')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Linha de Regressão')
plt.xlabel("Tamanho da Casa (m²)")
plt.ylabel("Preço (R$)")
plt.title("Regressão Linear: Preço vs. Tamanho da Casa")
plt.legend()
plt.grid(True)
plt.show()

# Avaliação do modelo (R-squared)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f"R-squared no conjunto de teste: {r2:.2f}")
```

**Análise do Exemplo:** O coeficiente indica que, para cada unidade adicional no "tamanho da casa", o "preço" tende a aumentar em aproximadamente 3.06 unidades. O intercepto seria o preço base se o tamanho da casa fosse zero. O R-squared ($R^2$) é uma métrica que indica o quão bem o modelo de regressão se ajusta aos dados, com valores próximos a 1 indicando um ajuste excelente.

-----

### 2\. Regressão Logística

A **Regressão Logística**, é um algoritmo de **classificação supervisionado** utilizado em problemas de **classificação binária**, onde a variável de saída é dicotômica (e.g., sim/não, 0/1, verdadeiro/falso). Embora seu nome seja "regressão", ela modela a **probabilidade** de uma instância pertencer a uma determinada classe e utiliza uma função logística (ou sigmoide) para mapear a saída linear para um valor entre 0 e 1.

#### **Exemplo em Python:**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X = iris.data
y = (iris.target != 0).astype(int) 

# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Instanciando e treinando o modelo de Regressão Logística
# max_iter é aumentado para garantir convergência em datasets maiores
model = LogisticRegression(solver='liblinear', random_state=42, max_iter=200)
model.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliando a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia no conjunto de teste: {accuracy:.4f}")

# Relatório de Classificação Detalhado
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=['Iris-Setosa', 'Não Iris-Setosa']))
```

**Análise do Exemplo:** A acurácia mostra a proporção de previsões corretas. O `classification_report` fornece métricas detalhadas como **Precisão (Precision)**, **Recall (Sensibilidade)** e **F1-Score** para cada classe, que são cruciais para avaliar o desempenho em problemas de classificação, especialmente quando há desbalanceamento de classes.

-----

### 3\. K-Nearest Neighbors (KNN)

O algoritmo **K-Nearest Neighbors (KNN)**, é um dos algoritmos de **classificação e regressão supervisionados** mais simples e intuitivos. É um método **não paramétrico** e **lazy learning** (aprendizagem preguiçosa), o que significa que ele não constrói um modelo explícito durante a fase de treinamento, mas sim armazena todo o conjunto de dados de treinamento. As previsões são feitas "na hora" da consulta, baseando-se na similaridade com as amostras mais próximas.


#### **Exemplo em Python:**

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = KNeighborsClassifier(n_neighbors=5) 
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Avaliando a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia no conjunto de teste (k=5): {accuracy:.4f}")

# Demonstração de previsão para uma nova amostra
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]]) # Características de uma Íris Setosa
predicted_class = model.predict(new_sample)
print(f"Previsão para a nova amostra: {iris.target_names[predicted_class[0]]}")
```

**Análise do Exemplo:** O KNN é simples de entender e implementar, mas sua performance pode ser sensível à escala das features e ao valor de `k`. A escolha de `k` e da métrica de distância é crucial para o desempenho.

-----

### 4\. K-Means Clustering

O **K-Means Clustering**, é um dos algoritmos de **agrupamento (clustering) não supervisionados** mais populares e utilizados. Tem como objetivo particionar `n` observações em `k` clusters, onde cada observação pertence ao cluster cujo centro (o *centroid*) é o mais próximo.

#### **Exemplo em Python:**

```python
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt 

np.random.seed(42)
X = np.concatenate([
    np.random.normal(loc=[0, 0], scale=1, size=(50, 2)), 
    np.random.normal(loc=[5, 5], scale=1, size=(50, 2)), 
    np.random.normal(loc=[-5, 5], scale=1, size=(50, 2)) 
])


# init='k-means++' é uma estratégia para escolher centroids iniciais que acelera a convergência

kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
kmeans.fit(X)

print("Rótulos atribuídos aos pontos (clusters):")
print(kmeans.labels_)
print("\nCentroides finais dos clusters:")
print(kmeans.cluster_centers_)

# Visualização dos clusters
plt.figure(figsize=(10, 7))
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', s=50, alpha=0.7, label='Pontos de Dados')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker='X', s=200, color='red', edgecolor='black', label='Centroides')
plt.title('K-Means Clustering Exemplo')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
```

**Análise do Exemplo:** O K-Means é eficiente em grandes datasets e fácil de interpretar. No entanto, ele requer que o número de clusters `k` seja especificado de antemão e é sensível à inicialização dos centroids e a clusters de formatos não esféricos ou densidades muito diferentes.

-----

### 5\. Árvores de Decisão

**Árvores de Decisão** são algoritmos de **aprendizado supervisionado** versáteis, utilizados tanto para **classificação** quanto para **regressão**. Eles funcionam construindo um modelo que se assemelha a uma estrutura de árvore, onde cada nó interno representa um teste em um atributo (feature), cada ramo representa o resultado do teste, e cada nó folha representa uma decisão ou um valor de previsão.


#### **Exemplo em Python:**

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# max_depth limita a profundidade da árvore para evitar overfitting e tornar a visualização mais clara
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Avaliando a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia no conjunto de teste: {accuracy:.4f}")

# Visualizando a Árvore de Decisão
plt.figure(figsize=(15, 10))
plot_tree(model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, rounded=True)
plt.title("Árvore de Decisão para Classificação de Íris")
plt.show()
```

**Análise do Exemplo:** A visualização da árvore de decisão mostra claramente quais são as regras de decisão aprendidas pelo modelo. Árvores de decisão são fáceis de interpretar, mas podem ser instáveis (pequenas mudanças nos dados podem levar a árvores muito diferentes) e tendem a overfitting.

-----

### 6\. Random Forest

**Random Forest** é um algoritmo de **aprendizado por conjunto (*ensemble learning*)** que utiliza múltiplas **Árvores de Decisão** para melhorar a precisão e a estabilidade. Ele constrói uma "floresta" de árvores de decisão durante o treinamento e, para classificação, a saída é a moda das classes previstas pelas árvores individuais (votação); para regressão, é a média das previsões das árvores.

#### **Exemplo em Python:**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X, y = iris.data, iris.target

# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Instanciando e treinando o modelo Random Forest

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Avaliando a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia no conjunto de teste: {accuracy:.4f}")

# Relatório de Classificação Detalhado
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

**Análise do Exemplo:** Random Forest, é geralmente um dos melhores algoritmos para a maioria dos problemas tabulares. Ele oferece alta precisão e boa resistência ao overfitting. A principal desvantagem é a menor interpretabilidade em comparação com uma única árvore de decisão.

-----

### 7\. Support Vector Machine (SVM)

**Support Vector Machines (SVMs)** são algoritmos de **aprendizado supervisionado** poderosos, utilizados para **classificação e regressão**. Para problemas de classificação, o SVM busca encontrar o **hiperplano ideal** que melhor separa as classes no espaço de features, maximizando a margem (distância) entre o hiperplano e os pontos de dados mais próximos de cada classe, conhecidos como **vetores de suporte**.

#### **Exemplo em Python:**

```python
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# kernel='rbf' (Radial Basis Function) é um kernel popular para dados não linearmente separáveis
# C é o parâmetro de regularização
model = svm.SVC(kernel='rbf', C=1.0, random_state=42)
model.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliando a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia no conjunto de teste: {accuracy:.4f}")

# Relatório de Classificação Detalhado
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

**Análise do Exemplo:** SVMs são particularmente eficazes em espaços de alta dimensão e quando o número de features é maior do que o número de amostras. A escolha do kernel e a otimização dos hiperparâmetros (C, gamma para RBF) são cruciais para o desempenho.

-----

## Avaliação de Modelos

A avaliação criteriosa de modelos de Aprendizado de Máquina é um passo **indispensável** para assegurarmos sua **robustez, generalização e eficácia** em cenários do mundo real. Não basta apenas treinar um modelo; é preciso quantificar quão bem ele se comporta em dados não vistos.

  * **Divisão de Dados:** A prática padrão, é dividir o dataset em conjuntos de **treinamento**, **validação** (para ajuste de hiperparâmetros) e **teste** (para avaliação final do desempenho generalizado). A validação cruzada (*cross-validation*), como o K-fold cross-validation, é uma técnica robusta para estimar o desempenho do modelo de forma mais confiável, reduzindo a dependência de uma única divisão.

### **Métricas de Avaliação para Classificação:**

Para problemas de classificação, diversas métricas fornecem diferentes perspectivas sobre o desempenho do modelo:

1.  **Acurácia (Accuracy):**

      * **Definição:** A proporção de previsões corretas (acertos) com relação ao total de previsões.
      * **Uso:** É uma métrica intuitiva, mas pode ser enganosa em datasets com **classes desbalanceadas**. Se 95% dos dados pertencem à classe majoritária, um modelo que sempre prevê a classe majoritária terá 95% de acurácia, mas seria inútil.

2.  **Matriz de Confusão:**

      * **Definição:** Uma tabela que sumariza o desempenho de um algoritmo de classificação, mostrando o número de previsões corretas e incorretas em cada classe.
      * **Componentes (para classificação binária):**
          * **Verdadeiro Positivo (VP):** Casos positivos corretamente previstos como positivos.
          * **Verdadeiro Negativo (VN):** Casos negativos corretamente previstos como negativos.
          * **Falso Positivo (FP - Erro Tipo I):** Casos negativos incorretamente previstos como positivos.
          * **Falso Negativo (FN - Erro Tipo II):** Casos positivos incorretamente previstos como negativos.
      * **Uso:** É a base para o cálculo de outras métricas e fornece uma visão granular dos tipos de erros que o modelo está cometendo.

3.  **Precisão (Precision):**

      * **Definição:** A proporção de verdadeiros positivos em relação a todos os resultados classificados como positivos pelo modelo. Responde: "Dos que o modelo previu como positivos, quantos eram realmente positivos?"
      * **Uso:** Importante quando o custo de um Falso Positivo é alto (e.g., diagnóstico de doença grave onde um FP pode levar a tratamentos desnecessários e ansiedade).

4.  **Recall (Sensibilidade ou Taxa de Verdadeiro Positivo):**

      * **Definição:** A proporção de verdadeiros positivos em relação a todos os casos que eram realmente positivos. Responde: "Dos que eram realmente positivos, quantos o modelo conseguiu identificar?"
      * **Uso:** Crítico quando o custo de um Falso Negativo é alto (e.g., detecção de fraude, diagnóstico de câncer onde um FN significa que uma condição séria não será tratada).

5.  **F1-Score:**

      * **Definição:** A média harmônica da Precisão e do Recall. É uma métrica de equilíbrio que tenta representar tanto a precisão quanto o recall em um único número.
      * **Uso:** Útil quando se busca um balanço entre Precisão e Recall, especialmente em datasets com classes desbalanceadas.

#### **Exemplo de Uso com `classification_report` (Scikit-learn):**

```python
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Re-treinando um modelo simples para ter y_true e y_pred
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) 
model = LogisticRegression(max_iter=500, solver='liblinear', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("--- Matriz de Confusão ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Relatório de Classificação ---")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

**Análise do Exemplo:** A **matriz de confusão**, mostra os counts exatos de VPs, VNs, FPs e FNs para cada classe. O **relatório de classificação** sumariza as métricas (precisão, recall, f1-score) para cada classe, além da acurácia geral e o suporte (número de instâncias reais da classe no conjunto de teste).

-----

### **Métricas Adicionais (Regressão e Outros):**

  * **Erro Médio Absoluto (MAE - Mean Absolute Error):** Média dos valores absolutos dos erros (diferença entre previsão e valor real).
  * **Erro Quadrático Médio (MSE - Mean Squared Error):** Média dos quadrados dos erros. Penaliza erros maiores mais severamente.
  * **Raiz do Erro Quadrático Médio (RMSE - Root Mean Squared Error):** Raiz quadrada do MSE, na mesma unidade da variável alvo.
  * **R-quadrado ($R^2$ ou Coeficiente de Determinação):** Explicar a proporção da variância na variável dependente que é previsível a partir das variáveis independentes. Varia de 0 a 1, sendo 1 um ajuste perfeito.
  * **Curva ROC e AUC (Area Under the Curve):** Para problemas de classificação binária, a Curva ROC (Receiver Operating Characteristic) plota a Taxa de Verdadeiro Positivo (Recall) versus a Taxa de Falso Positivo em vários limiares de classificação. AUC é a área sob essa curva; um AUC próximo de 1 indica um modelo excelente.

-----

## Conclusão

O **Aprendizado de Máquina** não é apenas uma área da Inteligência Artificial, é uma **disciplina transformadora** que busca redefinir a interação entre dados, algoritmos e tomada de decisão. Este portfólio ofereceu uma visão panorâmica dos seus fundamentos, para categorizar em **aprendizado supervisionado, não supervisionado e por reforço**, cada um com suas peculiaridades e casos de uso ideais.

Exploramos algoritmos clássicos e robustos como **Regressão Linear e Logística**, **K-Nearest Neighbors (KNN)**, **K-Means Clustering**, **Árvores de Decisão**, **Random Forest** e **Support Vector Machines (SVM)**, ilustrando suas aplicações práticas com exemplos em Python e a biblioteca Scikit-learn. Mais importante, enfatizamos a **crucialidade da avaliação de modelos** por meio de métricas como **acurácia, precisão, recall, F1-score e a matriz de confusão**, que são essenciais para validar a performance e a confiabilidade de qualquer solução de ML em cenários reais.

Obter o Domínio desses conceitos e algoritmos clássicos constitui uma base sólida para qualquer profissional que almeje atuar com IA aplicada. Embora novas técnicas, como as redes neurais profundas, continuem a surgir e a expandir as fronteiras da IA, a compreensão profunda desses fundamentos permite não só interpretar resultados complexos, mas também ajustar modelos, depurar problemas e construir soluções eficazes e eticamente responsáveis. O futuro do Aprendizado de Máquina é dinâmico e promissor, e a capacidade de aplicar esses conhecimentos de forma crítica e criativa será cada vez mais valiosa.

-----

## **Referências**

  * RUSSELL, Stuart J.; NORVIG, Peter. *Artificial Intelligence: A Modern Approach*. 4. ed. Pearson, 2020.
  * LUGER, George F. *Artificial Intelligence: Structures and Strategies for Complex Problem Solving*. 6. ed. Pearson, 2009.
  * HASTIE, Trevor; TIBSHIRANI, Robert; FRIEDMAN, Jerome. *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. 2. ed. Springer, 2009.
  * BISHOP, Christopher M. *Pattern Recognition and Machine Learning*. Springer, 2006.
  * GOODFELLOW, Ian; BENGIO, Yoshua; COURVILLE, Aaron. *Deep Learning*. MIT Press, 2016.
  * MUELLER, Andreas C.; GUIDA, Sarah. *Introduction to Machine Learning with Python: A Guide for Data Scientists*. O'Reilly Media, 2016.
  * ALPAYDIN, Ethem. *Introduction to Machine Learning*. 4. ed. MIT Press, 2020.