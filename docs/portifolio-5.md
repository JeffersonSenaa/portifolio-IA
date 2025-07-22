# Portfólio 5: Raciocínio Probabilístico e Estimação de Estados

## Introdução

Esta parte do portfólio busca explorar os conceitos fundamentais em Inteligência Artificial que lidam com a **incerteza**, realizam **inferência** e estimam **estados em ambientes dinâmicos**. Será abordado a **quantificação da incerteza** e o poder das **Redes Bayesianas**, passando pelo **raciocínio probabilístico ao longo do tempo**, até os **Filtros de Kalman** para estimação ótima de estados.

## 1\. Quantificando Incertezas e Redes Bayesianas

Os Agentes de IA no mundo real, por sua vez, precisam **lidar com a incerteza**. Esse fator, ocorre devido a fatores como a observabilidade parcial do ambiente, o não determinismo das ações ou presença de adversários. Uma abordagem inicial, sendo vista em agentes lógicos, é envolver o acompanhamento de todos os estados de mundo possíveis, mas essa estratégia tende a possuir desvantagens significativas em problemas complexos:

  * O agente precisaria considerar e armazenar **todas as explicações possíveis** para as suas observações, mesmo as observações mais improváveis.
  * Um **plano de contingência** que consiga lidar com todas as eventualidades e possa crescer arbitrariamente e incluir contingências muito improváveis.
  * Em alguns casos, não há um plano que garanta o alcance da meta, exigindo que o agente compare os méritos de planos não garantidos.

Para ilustrarmos, imagine um **agente de um modelo de Smart Taxi** que precisa levar um cliente ao aeroporto para um voo às 21h, exigindo que a chegada seja às 20h. O tempo de viagem poderá variar (30 min em trânsito médio, 45 min em pesado, 20 min em leve) dependendo do horário e condições climáticas (chuva, por exemplo, sempre significa trânsito pesado). Além disso, há uma chance de acidentes na rota, aumentando o tempo de viagem. Qual seria o melhor plano para chegar com segurança? A incerteza nesse caso é clara.

Outro exemplo é o **diagnóstico odontológico**. Uma regra lógica simples como "Dor de dente -\> Cárie" é incorreta, pois nem toda dor de dente é ou será cárie. Para torná-la correta, seria necessário listar todas as qualificações (exceções e condições) para que uma cárie cause dor de dente, o que é impraticável. Tentar usar essa lógica para domínios como diagnóstico médico é difícil por três razões:

  * **Complexidade**: É um trabalho árduo listar o conjunto completo de antecedentes ou consequentes que garanta uma regra sem exceções.
  * **Ignorância teórica**: A ciência médica, por exemplo, não possui uma teoria completa para seu domínio.
  * **Ignorância prática**: Mesmo com todas as regras, ainda pode haver incerteza sobre um paciente específico se nem todos os testes necessários foram ou puderam ser executados.

A solução para lidar com essa incerteza, nesses casos, é a **Teoria da Probabilidade**. Enquanto um agente lógico lida com sentenças como verdadeiro, falso ou desconhecido, um **agente probabilístico** pode ter um **grau numérico de crença** entre 0 (certamente falso) e 1 (certamente verdadeiro). Dessa forma, a teoria da probabilidade fornece uma maneira de **resumir a incerteza** que vem da falta de tempo e ignorância, resolvendo o problema da qualificação. Por exemplo, podemos acreditar que há 80% de chance (probabilidade de 0,8) de um paciente com dor de dente ter cárie.

Para a tomada de decisão sob incerteza, a **Teoria da Utilidade** pode ser combinada com as probabilidades. A utilidade representa as preferências do agente, onde cada estado (ou sequência de estados) possui um grau de utilidade, e o agente preferirá estados com maior utilidade. A **Teoria da Decisão** visa afirmar que um agente é racional se escolhe a ação que produz a **maior utilidade esperada** (MEU - Maximum Expected Utility), calculada pela média de todos os possíveis resultados de uma ação.

### Notação Básica de Probabilidade

  * **Espaço Amostral (Omega) e Mundos Possíveis (omega)**: O conjunto de todos os estados de mundos possíveis é -> o espaço amostral, e omega representa um elemento dele.
  * **Axiomas Básicos**: Todo estado de mundo possível tem uma probabilidade entre 0 e 1, e a probabilidade do conjunto de todos os estados de mundo possíveis é 1.
  * **Eventos/Proposições**: Conjuntos de mundos possíveis (por exemplo, a probabilidade de dois dados somarem 11). A probabilidade de uma proposição é a soma das probabilidades dos mundos em que ela é válida.
  * **Probabilidades Incondicionais (a priori)**: P(Total = 11) ou P(Duplos).
  * **Probabilidades Condicionais (a posteriori)**: P(a|b) - a probabilidade de 'a' dado 'b'. Definida como P(a E b) / P(b).
  * **Regra do Produto**: P(a E b) = P(a|b)P(b).
  * **Variáveis Aleatórias**: Representadas por letras maiúsculas (e.g., Dado1, Tempo) e seus valores por letras minúsculas (e.g., Dado1 = 5). Podem ser Booleanas (verdadeiro/falso), discretas (Idade = {juvenil, adolescente, adulto}) ou contínuas.
  * **Distribuição de Probabilidade**: P(X) em negrito indica um vetor de números, definindo a distribuição para a variável aleatória. P(X|Y) fornece os valores de P(X=xi | Y=yj).
  * **Função Densidade de Probabilidade (PDF)**: Usada para variáveis aleatórias contínuas.
  * **Distribuição Conjunta de Probabilidade**: P(Tempo, Cárie). Um modelo probabilístico é completamente determinado pela **distribuição de probabilidade conjunta completa** de todas as variáveis aleatórias.
  * **Marginalização (Somatório)**: Para calcular a probabilidade de uma proposição, somam-se as probabilidades dos mundos em que ela é verdadeira. Este processo remove variáveis da equação. Ex: P(Cárie) = P(Cárie, DorDeDente, Pego) + P(Cárie, ¬DorDeDente, Pego) + ....
  * **Regra de Condicionamento**: P(Y) = Soma\_z P(Y|z)P(z).
  * **Normalização**: Usamos alpha como uma constante para normalizar probabilidades, garantindo que somem 1.

### Independência e a Regra de Bayes

  * **Independência**: Duas proposições 'a' e 'b' são independentes se P(a E b) = P(a)P(b), ou equivalentemente, P(a|b) = P(a). A independência entre variáveis pode **reduzir dramaticamente a quantidade de informação** para especificar a distribuição conjunta completa e a complexidade da inferência.
  * **Regra de Bayes**: Derivada da regra do produto, é P(a|b) = P(b|a)P(a) / P(b). Esta equação é a **base da maioria dos sistemas modernos de IA** para inferência probabilística. É frequentemente usada quando temos evidência do **efeito** de uma causa desconhecida e desejamos determinar essa **causa** (P(causa|efeito) a partir de P(efeito|causa)). Por exemplo, um médico sabe P(sintomas|doença) e deseja P(doença|sintomas).
      * **Exemplo da Meningite**: Se P(torcicolo|meningite) = 0,7, P(meningite) = 1/50.000, e P(torcicolo) = 0,01. A probabilidade de um paciente com torcicolo ter meningite é P(meningite|torcicolo) = (0,7 \* 1/50.000) / 0,01 = 0,0014. Isso mostra como uma causa rara (meningite) pode ser inferida mesmo com um sintoma comum (torcicolo).
  * **Combinando Evidências**: Quando temos múltiplas evidências, como um paciente com dor de dente e uma sonda presa (indicando cárie), a complexidade aumenta. A chave é a **Independência Condicional**: Duas variáveis X e Y são condicionalmente independentes dada uma terceira variável Z se P(X, Y|Z) = P(X|Z)P(Y|Z). No exemplo odontológico, a dor de dente e a sonda presa são independentes, **dada a presença ou ausência de cárie**. Isso simplifica bastante a modelagem.
  * **Modelo de Bayes Ingênuo**: Um padrão comum onde uma única causa influencia diretamente uma série de efeitos que são condicionalmente independentes dada a causa. A distribuição conjunta completa pode ser escrita como P(Causa, Efeitos) = P(Causa) \* Produto P(Efeito\_i | Causa). Embora seja uma suposição simplificadora ("ingênua"), esses sistemas **funcionam muito bem na prática**, mesmo quando as condições de independência condicional não são estritamente verdadeiras.

### Redes Bayesianas

Uma **Rede Bayesiana** é um **grafo direcionado acíclico (DAG)** onde cada nó corresponde a uma variável aleatória (discreta ou contínua). Ligações por setas conectam pares de nós, onde X é um "genitor" de Y se há uma seta de X para Y. Cada nó X\_i tem uma **informação de probabilidade associada** theta(X\_i|Parents(X\_i)) que quantifica o efeito dos genitores no nó.

**A Semântica das Redes Bayesianas**: A rede Bayesiana define cada entrada na **distribuição conjunta** P(x\_1,...,x\_n) como o **produto das probabilidades condicionais locais**:

P(x\_1,...,x\_n) = Produto P(x\_i|parents(x\_i))

Isso significa que cada parâmetro da rede tem um significado preciso em termos de apenas um pequeno conjunto de variáveis, o que é crucial para a robustez e facilidade de especificação dos modelos. Sendo assim, uma rede Bayesiana pode ser usada para **responder a qualquer consulta** sobre o domínio, somando todos os valores de probabilidade conjunta relevantes.

**Método para Construir Redes Bayesianas**:

1.  **Nós**: Determine o conjunto de variáveis e ordene-as {X1,...,Xn}. A rede será mais compacta se as **causas precederem os efeitos**.
2.  **Links**: Para cada X\_i, escolha um conjunto mínimo de genitores de X1,...,Xi-1 que influenciam diretamente X\_i. Adicione um link do genitor para X\_i.
3.  **CPTs**: Escreva a **tabela de probabilidade condicional**, P(X\_i|Parents(X\_i)).

Este método, garante que a rede é **acíclica** e não contém valores de probabilidade redundantes.

**Independência Condicional em Redes Bayesianas**:

  * **Propriedade Não-Descendentes**: Cada variável é condicionalmente independente de seus não-descendentes, dados seus genitores.
  * **Cobertor de Markov**: Uma variável é condicionalmente independente de todos os outros nós da rede, dados seus **genitores, filhos e genitores dos filhos**.

**Representação Eficiente de Distribuições Condicionais**:

  * **Nós Determinísticos**: É 0 valor especificado exatamente pelos genitores, sem incerteza.
  * **Independência de Contexto Específico (CSI)**: Uma variável é condicionalmente independente de alguns de seus genitores dados certos valores de outros.
  * **OR-Ruídoso**: Permite incerteza sobre a capacidade de cada genitor de causar o filho ser verdadeiro.

**Redes Bayesianas com Variáveis Contínuas**:

  * **Discretização**: Dividir valores em intervalos fixos.
  * **Distribuições Probabilísticas**: Definir variáveis contínuas usando funções de densidade de probabilidade padrão (e.g., Gaussiana).
  * **Não Paramétrica**: Definir a distribuição condicional implicitamente com uma coleção de instâncias.

Redes Bayesianas que possuem variáveis discretas e contínuas são chamadas de **Redes Bayesianas Híbridas**.

**Exemplo de Aplicação**:

Aqui, será apresentado uma estrutura conceitual de como uma Rede Bayesiana pode ser implementada em código, focando na representação dos nós e na lógica de cálculo de probabilidade conjunta.

```python

class Node:
    def __init__(self, name, parents=None, cpts=None):
        self.name = name
        self.parents = parents if parents is not None else []
        self.cpts = cpts # Conditional Probability Tables (tabelas de probabilidade condicional)

class BayesianNetwork:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node):
        self.nodes[node.name] = node

    def calculate_joint_probability(self, assignment):
        # assignment: dicionário de {variable_name: value}
        # Multiplica as probabilidades condicionais de cada nó dado seus pais,
        # conforme a semântica da Rede Bayesiana. P(x1,...,xn) = Produto P(xi|parents(xi))
        joint_prob = 1.0
        for node_name, node in self.nodes.items():
            node_value = assignment.get(node_name)
            if node_value is None:
                raise ValueError(f"Missing value for node {node_name} in assignment")
            
            parent_values = {p_name: assignment.get(p_name) for p_name in node.parents}
            conditional_prob = self._get_conditional_probability(node, node_value, parent_values)
            joint_prob *= conditional_prob
        return joint_prob

    def _get_conditional_probability(self, node, node_value, parent_values):
        # Lógica para obter P(node_value | parent_values) da CPT do nó.
        # No fim das contas, CPTs são geralmente representadas como dicionários aninhados ou arrays.

        # Exemplo hipotético para JohnCalls (assumindo uma CPT simplificada):
        if node.name == "JohnCalls":
            alarm_state = parent_values.get("Alarm")
            if alarm_state is True:
                # Se o alarme está ligado, 90% de chance de John ligar (True), 10% de não ligar (False)
                return node.cpts['true_alarm'].get(node_value, 0.0) 
            elif alarm_state is False:
                # Se o alarme está desligado, 5% de chance de John ligar (falso positivo), 95% de não ligar
                return node.cpts['false_alarm'].get(node_value, 0.0) 
        
        # Para nós sem pais (raiz), a CPT seria apenas sua probabilidade a priori
        if not node.parents:
            return node.cpts.get(node_value, 0.0)
        
        return 0.0 # Retorna 0.0 se a CPT não for encontrada ou implementada para o nó
    
    def inference_by_enumeration(self, query_var, evidence):
        # Cálculo de P(query_var | evidence).
        # Este método envolve somar sobre todas as combinações possíveis de valores das variáveis não observadas.
        # Por exemplo: P(Burglary | JohnCalls=true, MaryCalls=true). As variáveis ocultas seriam Earthquake e Alarm.
        # A complexidade é alta para grandes redes, tornando-a computacionalmente cara.
        
        print(f"A inferência por enumeração para {query_var} dada {evidence} seria complexa e computacionalmente cara para grandes redes.")
        print("Envolveria somar sobre todas as combinações das variáveis ocultas (não observadas), multiplicando as probabilidades condicionais locais.")
        return "Conceptual Inference Result (Resultados reais exigem implementação completa)"

# Exemplo de Definição da Rede Bayesiana (Problema do Alarme):
# Nodes: Burglary (B), Earthquake (E), Alarm (A), JohnCalls (J), MaryCalls (M)
# Links: B -> A, E -> A, A -> J, A -> M

# Definindo as CPTs para o exemplo do alarme (valores hipotéticos/ilustrativos para o portfólio):
# Probabilidade a priori de Roubo e Terremoto
cpt_burglary = {True: 0.001, False: 0.999}
cpt_earthquake = {True: 0.002, False: 0.998}

# CPT do Alarme dependendo de Roubo e Terremoto
# P(Alarm | Burglary, Earthquake)
# Formato: {(Burglary_val, Earthquake_val): {Alarm_True: prob, Alarm_False: prob}}
cpt_alarm = {
    (True, True): {True: 0.94, False: 0.06},    # P(A|B,E)
    (True, False): {True: 0.95, False: 0.05},   # P(A|B,~E)
    (False, True): {True: 0.29, False: 0.71},   # P(A|~B,E)
    (False, False): {True: 0.001, False: 0.999} # P(A|~B,~E)
}

# CPTs para JohnCalls e MaryCalls dependendo do Alarme
# P(JohnCalls | Alarm)
cpt_johncalls = {
    'true_alarm': {True: 0.90, False: 0.10},  # P(J|A)
    'false_alarm': {True: 0.05, False: 0.95} # P(J|~A) (falso positivo, por exemplo)
}

# P(MaryCalls | Alarm)
cpt_marycalls = {
    'true_alarm': {True: 0.70, False: 0.30},  # P(M|A)
    'false_alarm': {True: 0.01, False: 0.99} # P(M|~A) (falso positivo, por exemplo)
}

# Criando os nós da rede
node_burglary = Node("Burglary", cpts=cpt_burglary)
node_earthquake = Node("Earthquake", cpts=cpt_earthquake)
node_alarm = Node("Alarm", parents=["Burglary", "Earthquake"], cpts=cpt_alarm)
node_johncalls = Node("JohnCalls", parents=["Alarm"], cpts=cpt_johncalls)
node_marycalls = Node("MaryCalls", parents=["Alarm"], cpts=cpt_marycalls)

# Construindo a rede
alarm_network = BayesianNetwork()
alarm_network.add_node(node_burglary)
alarm_network.add_node(node_earthquake)
alarm_network.add_node(node_alarm)
alarm_network.add_node(node_johncalls)
alarm_network.add_node(node_marycalls)

# Exemplo de uso (conceitual):
# Suponha que John e Mary ligaram. Qual a probabilidade de ter ocorrido um roubo?
# assignment_example = {
#     "Burglary": True, "Earthquake": False, "Alarm": True, "JohnCalls": True, "MaryCalls": True
# }
# joint_prob = alarm_network.calculate_joint_probability(assignment_example)
# print(f"\nProbabilidade conjunta do cenário: {joint_prob:.6f}")

# Esse é um Exemplo de inferência (apenas conceitual, pois a implementação completa é mais complexa)
# alarm_network.inference_by_enumeration(query_var="Burglary", 
#                                        evidence={"JohnCalls": True, "MaryCalls": True})
```

Para inferência em redes bayesianas, especialmente em grandes redes, a inferência exata (como por enumeração) pode ser **intratável**. Por exemplo, uma rede como a do seguro de carro precisaria de milhões de operações aritméticas para uma consulta. Métodos de **inferência aproximada**, também chamados de **algoritmos de Monte Carlo**, são usados para lidar com isso. Eles são capazes de gerar eventos aleatórios com base nas probabilidades da rede Bayesiana e contar as respostas. Com amostras suficientes, podemos nos aproximar da verdadeira distribuição de probabilidade.

  * **Amostragem Direta**: Gera amostras de uma distribuição de probabilidade conhecida, amostrando cada variável em ordem topológica.
  * **Cadeia de Markov Monte Carlo (MCMC)**: Gera uma amostra fazendo uma alteração aleatória na amostra anterior. Exemplos incluem Gibbs Sampling e Metropolis-Hastings.

-----

## 2\. Raciocínio Probabilístico ao Longo do Tempo

Enquanto problemas **estáticos** tendem a assumir que o estado do mundo não muda significativamente (como em um diagnóstico de carro), problemas **dinâmicos** envolvem estados que **mudam com o tempo** (como monitorar um paciente com diabetes ou rastrear a posição de um veículo). Para modelar isso, consideramos **problemas de tempo discreto**, onde as amostras temporais são enumeradas (0, 1, 2, ...).

  * **Variáveis de Estado (Xt)**: Variáveis não observáveis no tempo t (ex: a verdadeira posição de um carro).
  * **Variáveis de Evidência (Et)**: Variáveis observáveis no tempo t (ex: leituras do GPS do carro).

Dois componentes-chave são necessários:

  * **Modelo de Transição**: Especifica como o mundo evolui, ou seja, a distribuição de probabilidade das variáveis de estado mais recentes, dados os valores anteriores, P(Xt|X0:t-1). Para resolver o problema de o histórico crescer infinitamente, fazemos uma **suposição de Markov**: o estado atual depende apenas de um número fixo finito de estados anteriores. A forma mais simples é um **processo de Markov de 1ª ordem**, onde P(Xt|Xt-1).
  * **Modelo de Sensor (ou Observação)**: Especifica como as variáveis de evidência recebem seus valores, P(Et|X0:t, E1:t-1). A **suposição de sensor de Markov** simplifica isso para P(Et|Xt).

Além desses modelos, precisamos da **distribuição de probabilidade a priori no tempo 0, P(X0)**. Com isso, a distribuição conjunta completa sobre todas as variáveis é dada por:

P(X0:t, E1:t) = P(X0) \* Produto [P(Xi|Xi-1) \* P(Ei|Xi)]

Temos então como um exemplo clássico o **"Mundo do Guarda-Chuva"**, que é um processo de Markov de primeira ordem onde a probabilidade de chuva depende da chuva no dia anterior, e a observação (guarda-chuva) depende do estado da chuva.

### Inferência em Modelos Temporais

Quatro tarefas básicas de inferência são importantes:

  * **Filtragem (ou Estimação de Estados)**: Calcular a distribuição a posteriori do estado mais recente, dado todas as evidências até o momento: **P(Xt|e1:t)**. Um algoritmo de filtragem útil mantém o estado atual e o atualiza, em vez de voltar a todo o histórico de percepções. O cálculo envolve projetar a distribuição de estados de t para t+1 e depois atualizá-la com a nova evidência.
      * **Equação Recursiva para Filtragem**: P(Xt+1|e1:t+1) = alpha P(et+1|Xt+1) Soma P(Xt+1|Xt) P(Xt|e1:t). (Conhecida como "Forward Message")
  * **Predição**: Calcular a distribuição a posteriori de um estado futuro, dada todas as evidências até o instante de tempo atual: **P(Xt+k |e1:t)** para k \> 0.
  * **Suavização (Smoothing)**: Computar a distribuição a posteriori de um estado passado, dada todas as evidências até o instante de tempo atual: **P(Xk|e1:t)** para 0 \<= k \< t. Isso pode ser feito recursivamente, dividindo a evidência em duas partes. (Conhecida como "Backward Message")
  * **Explicação Mais Provável**: Dada uma sequência de observações, encontrar a sequência de estados que mais provavelmente gerou essas observações: **argmax x1:t P(x1:t |e1:t)**. Isso é resolvido pelo **Algoritmo de Viterbi**, que encontra o caminho mais provável através dos estados ao longo do tempo.

### Modelo Oculto de Markov (HMM)

Um **HMM** é um modelo probabilístico temporal onde o estado do processo é descrito por uma **única variável aleatória discreta**. O "Mundo do Guarda-Chuva" é um exemplo de HMM, pois possui uma única variável de estado ("chuva"). Embora os HMMs exijam um estado discreto, não há restrição para as variáveis de evidência, pois elas são sempre observadas.

**Algoritmos Matriciais Simplificados para HMMs**:

Para uma única variável de estado discreta Xt com 'S' valores possíveis:

  * O **modelo de transição P(Xt|Xt-1)** torna-se uma **matriz S x S T**, onde T\_i,j é a probabilidade de transição do estado i para o estado j.
  * O **modelo de sensor P(Et|Xt)** torna-se uma **matriz de observação S x S Ot** para cada passo de tempo. A i-ésima entrada diagonal de Ot é P(et|Xt=i) e as outras entradas são 0.

As equações recursivas de filtragem e suavização podem ser expressas de forma concisa usando essas matrizes.

**Exemplo de Aplicação**:

O exemplo abaixo demonstra a estrutura de um HMM para o "Mundo do Guarda-Chuva", representando suas probabilidades de transição e observação como matrizes e dicionários.

```python
# Conceito de Estrutura de Código para Modelos Ocultos de Markov (HMM)
import numpy as np

class HMM:
    def __init__(self, states, observations_map, transition_matrix, initial_state_prob):
        self.states = states 
        self.observations_map = observations_map # Mapeia evidências para distribuições de probabilidade P(E|X)
        self.T = np.array(transition_matrix) 
        self.pi = np.array(initial_state_prob) 

    def _create_observation_matrix(self, observed_evidence):
        # Ot é uma matriz diagonal onde O_t[i,i] = P(observed_evidence | Xt=states[i])
        S = len(self.states)
        O_t = np.diag([self.observations_map[state][observed_evidence] for state in self.states])
        return O_t

    def forward_algorithm(self, evidences_sequence):
        # Implementa o algoritmo de filtragem (forward algorithm)
        # alpha_t = alpha * Ot * T.T * alpha_t-1
        
        alpha = self.pi 
        filtered_states = []

        for evidence in evidences_sequence:
            O_t = self._create_observation_matrix(evidence)
            
            # Etapa de Predição: projected_alpha = T.T @ alpha (P(Xt+1|e1:t) não normalizado)
            predicted_alpha = np.dot(self.T.T, alpha)
            
            # Etapa de Atualização: alpha = O_t @ predicted_alpha (P(Xt+1|e1:t+1) não normalizado)
            updated_alpha_unnormalized = np.dot(O_t, predicted_alpha)
            
            # Normalização
            alpha = updated_alpha_unnormalized / np.sum(updated_alpha_unnormalized)
            filtered_states.append(alpha)
            
        return filtered_states

    def backward_algorithm(self, evidences_sequence):
        # Implementa o algoritmo de suavização (backward algorithm)
        # b_k:t = T * O_k+1 * b_k+1:t
        
        S = len(self.states)
        b = np.ones(S) # Inicialização P(e_t+1:t | Xt) = 1 (para o último passo)
        
        backward_messages = [b]
        # Itera de trás para frente na sequência de evidências (excluindo a última)
        for t in range(len(evidences_sequence) - 1, 0, -1):
            O_t_plus_1 = self._create_observation_matrix(evidences_sequence[t])
            # Calcula b_k (P(e_k+1:t | X_k))
            b = np.dot(self.T, np.dot(O_t_plus_1, b))
            backward_messages.insert(0, b)
        
        return backward_messages

    def smooth_states(self, evidences_sequence):
        # Combina forward e backward para suavização
        # P(Xk|e1:t) = alpha * f_1:k * b_k+1:t (elemento a elemento)
        filtered_messages = [self.pi] + self.forward_algorithm(evidences_sequence) # Inclui P(X0) no início
        backward_messages = self.backward_algorithm(evidences_sequence)
        
        smoothed_states = []
        for k in range(len(evidences_sequence)): # Itera sobre t=0 até T-1 (para evidences_sequence)
            smoothed_unnormalized = filtered_messages[k] * backward_messages[k]
            smoothed_states.append(smoothed_unnormalized / np.sum(smoothed_unnormalized))
        return smoothed_states

    def viterbi_algorithm(self, evidences_sequence):
        # Implementa o algoritmo de Viterbi para encontrar a sequência de estados mais provável
        # m_1:t+1(x_t+1) = P(e_t+1|x_t+1) * max_{x_t} (P(x_t+1|x_t) * m_1:t(x_t))
        
        S = len(self.states)
        T_steps = len(evidences_sequence)
        
        dp_table = np.zeros((S, T_steps)) # Armazena a probabilidade do caminho mais provável até o estado t
        path_table = np.zeros((S, T_steps), dtype=int) # Armazena os backpointers para reconstruir o caminho
        
        # Inicialização (t=0)
        O0 = self._create_observation_matrix(evidences_sequence[0])
        dp_table[:, 0] = self.pi * np.diag(O0)
        
        # Iteração para t=1 até T_steps-1
        for t in range(1, T_steps):
            O_t = self._create_observation_matrix(evidences_sequence[t])
            for j in range(S): # Para cada estado atual j
                possible_prev_probs = np.array([dp_table[i, t-1] * self.T[i, j] for i in range(S)])
                
                max_prob = np.max(possible_prev_probs)
                max_idx = np.argmax(possible_prev_probs)
                
                dp_table[j, t] = np.diag(O_t)[j] * max_prob
                path_table[j, t] = max_idx
                
        # Reconstrução do caminho mais provável
        most_likely_path = [np.argmax(dp_table[:, T_steps-1])]
        for t in range(T_steps-1, 0, -1):
            most_likely_path.insert(0, path_table[most_likely_path[0], t])
            
        return [self.states[idx] for idx in most_likely_path]

# Exemplo de uso conceitual para o Mundo do Guarda-Chuva:
# Estados: 'Rain', 'NoRain'
# Observações: 'Umbrella', 'NoUmbrella'

# Probabilidades de Transição (T): P(Xt|Xt-1)
# T[i,j] = P(State_j | State_i)
# Ex: T = [[P(Rain|Rain), P(NoRain|Rain)], [P(Rain|NoRain), P(NoRain|NoRain)]]
transition_matrix_ex = [[0.7, 0.3], # Se estava chovendo, 70% de chance de chover no próximo dia
                        [0.3, 0.7]] # Se não estava chovendo, 30% de chance de chover no próximo dia

# Probabilidades Iniciais (pi): P(X0)
# Ex: pi = [P(Rain), P(NoRain)]
initial_state_prob_ex = [0.5, 0.5] # 50% de chance de chover no dia 0

# Probabilidades de Observação (observations_map): P(E|X)
observations_map_ex = {
    'Rain': {'Umbrella': 0.9, 'NoUmbrella': 0.1}, # Se está chovendo, 90% de chance de levar guarda-chuva
    'NoRain': {'Umbrella': 0.2, 'NoUmbrella': 0.8} # Se não está chovendo, 20% de chance de levar guarda-chuva (por precaução)
}

hmm_model = HMM(
    states=['Rain', 'NoRain'],
    observations_map=observations_map_ex,
    transition_matrix=transition_matrix_ex,
    initial_state_prob=initial_state_prob_ex
)

# Exemplo de sequência de evidências (observações de guarda-chuva ao longo de 3 dias)
evidences = ['Umbrella', 'Umbrella', 'NoUmbrella']

```

-----

## 3\. Filtros de Kalman

Até o momento, foi abordado os modelos onde as variáveis de estado eram discretas. Mas e se o estado do mundo é contínuo, como a posição e velocidade de um carro, ou a temperatura de uma sala? Para esses cenários, os **Filtros de Kalman** são uma ferramenta poderosa para a **estimação ótima de estados em sistemas dinâmicos lineares com ruído Gaussiano**.

Um Filtro de Kalman é um algoritmo recursivo que permite estimar o estado de um processo com base em uma série de medições ruidosas. Ele é amplamente utilizado em navegação, controle de robôs, sistemas de rastreamento e muitas outras aplicações de engenharia e IA.

A intuição por trás do Filtro de Kalman é que ele combina duas fontes de informação:

1.  **Modelo de Previsão (Modelo de Transição de Estado)**: Como o estado do sistema se espera que evolua ao longo do tempo. Este modelo é baseado nas leis da física ou em um entendimento do comportamento do sistema, e também inclui o ruído do processo (incerteza em como o sistema se move).
2.  **Modelo de Medição (Modelo de Sensor)**: Como as medições observadas se relacionam com o estado verdadeiro do sistema. Este modelo também considera o ruído da medição (incerteza nas leituras do sensor).

O filtro opera em duas fases principais de forma recursiva:

  * **Fase de Predição**: O filtro estima o estado atual e a incerteza associada com base na estimativa do estado anterior e no modelo de transição do sistema. Isso gera uma "previsão a priori" ou "predição".
  * **Fase de Atualização**: Quando uma nova medição se torna disponível, o filtro combina essa medição com a previsão a priori para refinar a estimativa do estado. A medição ruidosa é ponderada pela sua incerteza e pela incerteza da previsão, resultando em uma "estimativa a posteriori" ou "correção".

A beleza do Filtro de Kalman reside em sua capacidade de fornecer uma estimativa ótima (no sentido de minimizar o erro quadrático médio) para sistemas lineares Gaussianos, mesmo quando as medições são imprecisas e o sistema é afetado por ruído. A incerteza do estado é representada por uma **matriz de covariância**, que é atualizada em cada passo do tempo, refletindo como a confiança na estimativa do estado muda.

### Equações do Filtro de Kalman

Vamos representar o estado do sistema no tempo `t` como um vetor `x_t` e a medição como um vetor `z_t`.

**Fase de Predição:**

1.  **Estado Previsto (a priori)**: `x̂_t|t-1 = F_t * x̂_t-1|t-1 + B_t * u_t`
      * `x̂_t|t-1`: Estimativa do estado no tempo `t` baseada nas informações até `t-1`.
      * `F_t`: Matriz de transição de estado.
      * `x̂_t-1|t-1`: Estimativa do estado anterior (corrigida).
      * `B_t`: Matriz de controle (opcional, para entradas de controle `u_t`).
      * `u_t`: Vetor de controle.
2.  **Covariância Prevista (a priori)**: `P_t|t-1 = F_t * P_t-1|t-1 * F_t.T + Q_t`
      * `P_t|t-1`: Matriz de covariância do erro da previsão.
      * `P_t-1|t-1`: Matriz de covariância do erro da estimativa anterior (corrigida).
      * `Q_t`: Matriz de covariância do ruído do processo (reflete incertezas no modelo).

**Fase de Atualização:**

1.  **Ganho de Kalman**: `K_t = P_t|t-1 * H_t.T * (H_t * P_t|t-1 * H_t.T + R_t)^-1`
      * `K_t`: Ganho de Kalman, pondera o quanto a medição influencia a correção da estimativa.
      * `H_t`: Matriz de observação (transforma o estado em espaço de medição).
      * `R_t`: Matriz de covariância do ruído da medição (reflete incertezas do sensor).
2.  **Estado Corrigido (a posteriori)**: `x̂_t|t = x̂_t|t-1 + K_t * (z_t - H_t * x̂_t|t-1)`
      * `x̂_t|t`: Estimativa final do estado no tempo `t` após incorporar a medição.
      * `z_t`: Vetor de medições no tempo `t`.
3.  **Covariância Corrigida (a posteriori)**: `P_t|t = (I - K_t * H_t) * P_t|t-1`
      * `P_t|t`: Matriz de covariância do erro da estimativa corrigida.
      * `I`: Matriz identidade.

### Exemplo de Aplicação

Considere o rastreamento de um objeto se movendo em uma dimensão (por exemplo, um carro em uma estrada reta). O estado `x` pode ser definido como `[posição, velocidade]`.

```python
# Conceito de Estrutura de Código para Filtro de Kalman
import numpy as np

class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, F, H, Q, R, B=None):
        self.x_hat = initial_state         
        self.P = initial_covariance        # P_t-1|t-1 (Covariância inicial do erro)
        self.F = F                         
        self.H = H                         
        self.Q = Q                         
        self.R = R                         
        self.B = B                         

    def predict(self, u=None):
        # 1. Estado Previsto: x̂_t|t-1 = F_t * x̂_t-1|t-1 + B_t * u_t
        if self.B is not None and u is not None:
            self.x_hat_prior = np.dot(self.F, self.x_hat) + np.dot(self.B, u)
        else:
            self.x_hat_prior = np.dot(self.F, self.x_hat)
        
        # 2. Covariância Prevista: P_t|t-1 = F_t * P_t-1|t-1 * F_t.T + Q_t
        self.P_prior = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        return self.x_hat_prior, self.P_prior

    def update(self, z):
        # 3. Ganho de Kalman: K_t = P_t|t-1 * H_t.T * (H_t * P_t|t-1 * H_t.T + R_t)^-1
        # Inovação ou Resíduo: y = z_t - H_t * x̂_t|t-1
        y = z - np.dot(self.H, self.x_hat_prior)
        
        # Covariância da Inovação: S = H_t * P_t|t-1 * H_t.T + R_t
        S = np.dot(np.dot(self.H, self.P_prior), self.H.T) + self.R
        
        K = np.dot(np.dot(self.P_prior, self.H.T), np.linalg.inv(S))
        
        # 4. Estado Corrigido: x̂_t|t = x̂_t|t-1 + K_t * (z_t - H_t * x̂_t|t-1)
        self.x_hat = self.x_hat_prior + np.dot(K, y)
        
        # 5. Covariância Corrigida: P_t|t = (I - K_t * H_t) * P_t|t-1
        I = np.eye(self.P.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P_prior)
        
        return self.x_hat, self.P

# --- Exemplo de Uso de um Filtro de Kalman para Rastreamento de Posição/Velocidade ---

# Definindo o intervalo de tempo (dt) e o ruído
dt = 1.0 # Segundo
process_noise_std = 0.1 # Desvio padrão do ruído do processo
measurement_noise_std = 0.5 # Desvio padrão do ruído da medição

# Estado inicial: [posição, velocidade]
initial_state = np.array([0.0, 0.0]) # Posição 0, velocidade 0
# Covariância inicial: Muita incerteza no início
initial_covariance = np.array([[1.0, 0.0],
                               [0.0, 1.0]])

# Matriz de Transição de Estado (F):
# Nova posição = Posição anterior + Velocidade * dt
F_matrix = np.array([[1.0, dt],
                     [0.0, 1.0]])

# Matriz de Observação (H):
H_matrix = np.array([[1.0, 0.0]])

# Matriz de Covariância do Ruído do Processo (Q):
Q_matrix = np.array([[0.25 * dt**4, 0.5 * dt**3],
                     [0.5 * dt**3, dt**2]]) * process_noise_std**2

# Matriz de Covariância do Ruído da Medição (R):
# Reflete a incerteza das leituras do sensor (GPS ruidoso).
R_matrix = np.array([[measurement_noise_std**2]])

# Criando o filtro de Kalman
kf = KalmanFilter(initial_state, initial_covariance, F_matrix, H_matrix, Q_matrix, R_matrix)

true_positions = []
measured_positions = []
estimated_positions = []

true_pos = 0.0
true_vel = 1.0 # Velocidade constante de 1 m/s

num_steps = 20
for i in range(num_steps):
    # Simula o estado real (com algum ruído de processo real)
    process_noise = np.random.normal(0, process_noise_std, 2) # [ruído_pos, ruído_vel]
    true_pos = true_pos + true_vel * dt + process_noise[0]
    true_vel = true_vel + process_noise[1] # Velocidade pode variar ligeiramente

    true_state = np.array([true_pos, true_vel])
    true_positions.append(true_state[0])

    # Simula a medição ruidosa (apenas posição)
    measurement_noise = np.random.normal(0, measurement_noise_std)
    measured_pos = true_pos + measurement_noise
    measured_positions.append(measured_pos)

    # Aplica o Filtro de Kalman
    kf.predict()
    estimated_state, _ = kf.update(np.array([measured_pos]))
    estimated_positions.append(estimated_state[0])

print("\n--- Exemplo de Rastreamento com Filtro de Kalman ---")
print(f"Número de passos simulados: {num_steps}")
print(f"Primeiras 5 Posições Reais: {true_positions[:5]}")
print(f"Primeiras 5 Posições Medidas: {measured_positions[:5]}")
print(f"Primeiras 5 Posições Estimadas: {estimated_positions[:5]}")
print("\n(Em um portfólio real, gráficos mostrando a redução do ruído seriam ideais)")

```

### Conclusão

Este portfólio buscou explorar a essência da IA na gestão da incerteza, sendo uma peça fundamental para sistemas que operam no mundo real. Discutimos como a Teoria da Probabilidade e a Regra de Bayes oferecem uma base robusta para quantificar crenças e realizar inferências, superando as limitações da lógica booleana. As Redes Bayesianas, apresentadas como modelos gráficos são eficientes para representar e raciocinar sobre dependências complexas.

Também aprofundamos no raciocínio probabilístico temporal, com Modelos Ocultos de Markov (HMMs), essenciais para filtrar e prever estados em sequências de tempo. Finalmente, abordamos os Filtros de Kalman, destacando sua otimização na estimação de estados contínuos em sistemas dinâmicos lineares. Em síntese, obter domínio sobre esses conceitos é crucial para construir IA robustas que operam com eficácia e confiança em ambientes ambíguos, permitindo que as máquinas aprendam e ajam inteligentemente apesar da informação imperfeita.

---

## **Referências**

* RUSSELL, Stuart J.; NORVIG, Peter. *Inteligência Artificial*. 3. ed. Rio de Janeiro: Elsevier, 2013. (Título original: *Artificial Intelligence: A Modern Approach*)
* LUGER, George F. *Inteligência Artificial: estruturas e estratégias para a resolução de problemas complexos*. 6. ed. Pearson, 2009.
* NILSSON, Nils J. *Artificial Intelligence: A New Synthesis*. Morgan Kaufmann, 1998.
* THRUN, Sebastian; BURGARD, Wolfram; FOX, Dieter. *Probabilistic Robotics*. MIT Press, 2005.
* DEAN, Thomas L.; KANAZAWA, Keiji. *A Model for Reasoning About Persistence and Causation*. Computational Intelligence, v. 5, n. 3, p. 142-150, 1989.

---