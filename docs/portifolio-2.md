# Portifolio 2

## Algoritmo de busca cega;

## Random Search

```python
import random
def random_search(goal, domain):
attempts = 0
while True:
guess = random.choice(domain)
attempts += 1
if guess == goal:
return guess, attempts
```

## O que são Algoritmos de Busca Cega?

**Algoritmos de busca cega** (ou **busca não informada**) são algoritmos que **não utilizam nenhuma informação adicional** sobre o problema, como distância ou estimativas até o objetivo. Eles exploram o espaço de busca de forma sistemática e **completa**, sem levar em consideração a posição da meta, sendo então “cegos” quanto ao melhor caminho.

## Origem

- Surgiram nos **anos 1950-60**, envolto aos primeiros trabalhos em **Inteligência Artificial simbólica**.
- Aplicados em **resolução automática de problemas**, como quebra-cabeças, jogos simples e sistemas de inferência lógica.
- São os **primeiros tipos de algoritmos de busca utilizados em IA**, tendo raízes na teoria de grafos e na teoria dos autômatos.

## Como Funcionam?

Eles expandem os **nós** do espaço de estados de acordo com uma **regra fixa**, como:

- Em ordem de chegada (fila – busca em largura).
- Em ordem de profundidade (pilha – busca em profundidade).
- Com menor custo acumulado (busca de custo uniforme).

Esses algoritmos, **não sabem se estão se aproximando da meta**. Apenas tentam todos os caminhos possíveis dentro do contexto.

## Contextos de Aplicação

Embora menos eficientes que os algoritmos informados, os algoritmos de busca cega ainda são úteis quando:

- **Não há informação heurística confiável disponível.**
- O **espaço de busca é pequeno.**
- É necessário garantir que a **solução seja ótima ou completa**, mesmo sem pistas.

Exemplos de aplicações:

- Soluções para **problemas clássicos** de IA como o problema das torres de Hanói, quebra-cabeças 8-puzzle.
- **Sistemas educacionais**, para ensinar lógica de busca.
- **Diagnóstico automatizado** onde a heurística não é clara.
- **Planejamento de ações simples** em jogos e agentes autônomos.

## Importância

- São a **base conceitual** para algoritmos mais avançados.
- **Garantem completude** (encontram a solução, se houver uma).
- Podem ser usados em **qualquer tipo de problema**, mesmo sem conhecimento do domínio.
- São úteis para **comparação e benchmarking** de novos algoritmos.

### 1. **Busca Exaustiva (Brute-force Search)**

- Verifica **todas as combinações possíveis** até encontrar a solução.
- Muito lento, mas **simples e confiável**.
- Usado quando o espaço de busca é pequeno ou para validar soluções.

```
def brute_force_search(problem):
    for solution in problem.all_possible_solutions():
        if problem.is_goal(solution):
            return solution
    return None
#Aplicacao: Tentar todas as senhas possíveis de um cadeado.
```

### 2. **Backtracking**

- Técnica de busca **recursiva** que **volta atrás** quando encontra um beco sem saída.
- Muito usada em **problemas de combinação e permutação**, como Sudoku e quebra-cabeças.

```python
def solve(problem, state):
    if problem.is_goal(state):
        return state
    for move in problem.legal_moves(state):
        next_state = problem.apply_move(state, move)
        result = solve(problem, next_state)
        if result:
            return result
    return None
#Aplicacao: Resolver o Sudoku ou labirintos.
```

### 3. **Iterative Deepening Search (IDS)**

- Combinação de profundidade com garantia de completude.
- Executa várias buscas em profundidade com profundidade limitada que vai aumentando.
- Usa **pouca memória** encontrando soluções ótimas.

## Aplicação: Sudoku com Backtracking

Imagine que você está resolvendo um **Sudoku**. Você não sabe qual número colocar, então tenta todos de 1 a 9 em uma célula vazia. Caso dê erro, você volta atrás e tenta outro número — esse é o comportamento típico da **busca cega com backtracking**.

---

## Algoritmo de busca informada;

### Algoritmo de Simulated Annealing

```python
import math
import random

def simulated_annealing(objective_function, domain, temp_initial, cooling_rate):
    current = random.uniform(*domain)
    temp = temp_initial

    while temp > 1:
        neighbor = current + random.uniform(-1, 1)
        neighbor = max(min(neighbor, domain[1]), domain[0])
        cost_diff = objective_function(neighbor) - objective_function(current)

        if cost_diff < 0 or math.exp(-cost_diff / temp) > random.random():
            current = neighbor

        temp *= cooling_rate

    return current

# Exemplo: minimizar a função f(x) = x^2
result = simulated_annealing(lambda x: x**2, domain=(-10, 10), temp_initial=100, cooling_rate=0.95)
print(f"Solução encontrada: {result}, f(x) = {result**2}")
#**Aplicação:** Otimizar de forma contínua as funções complexas com mínimos locais.

```

## O que são Algoritmos de Busca Informada?

**Algoritmos de busca informada**, são algoritmos que utilizam **informações adicionais (heurísticas)** sobre o problema para a tomada de decisões mais inteligentes durante a busca por soluções. Tendem a ser mais eficientes do que algoritmos de **busca cega** (como BFS ou DFS) porque conseguem **priorizar caminhos promissores** e assim evitar explorar caminhos irrelevantes.

## Origem

- Derivações de pesquisas em **ciência da computação** e **IA clássica** nas décadas de 1960 e 1970.
- Fortemente influenciados pelo trabalho de nomes como **Alan Turing**, **John McCarthy** e **Marvin Minsky**.
- Cresceram com os primeiros sistemas de **planejamento automático** e **resolução de problemas** em ambientes como jogos e sistemas especialistas.

## Como Funcionam?

Esses algoritmos usam uma **função heurística** `h(n)` que estima o custo restante de um estado `n` até a solução. Essa função:

- **Não precisa ser perfeita**, apenas uma boa estimativa.
- Ajuda a **escolher qual nó explorar primeiro** com base em qual parece mais promissor.

## Contextos de Aplicação

- **Jogos de estratégia e tabuleiro** (ex.: xadrez, damas, sudoku)
- **Planejamento de rotas** (ex.: GPS, logística)
- **Robótica** (ex.: navegação de robôs autônomos)
- **Diagnóstico automático e sistemas especialistas**
- **Soluções em tempo real** para IA em videogames
- **Assistentes de voz e NLP** (em fases de planejamento de diálogo)

## Importância

- **Eficiência:** Exploram **menos nós** que as buscas cegas.
- **Escalabilidade:** Lidam melhor com **espaços de busca grandes**.
- **Adaptabilidade:** A heurística pode ser ajustada para diferentes domínios.
- São a **base para algoritmos como A*, IDA*, Beam Search** e etc.

## Exemplos de Algoritmos de Busca Informada

### 1. **Beam Search**

- Explora apenas os `k` melhores caminhos em cada nível da árvore de busca, limitando o número de nós expandidos.
- Usado em **NLP** (ex: tradução automática), pois é mais eficiente do que expandir todos os caminhos possíveis.

```python
import heapq

def beam_search(start, goal, neighbors, heuristic, beam_width):
    beam = [(heuristic(start, goal), start)]
    while beam:
        new_beam = []
        for _, node in beam:
            if node == goal:
                return node
            for neighbor in neighbors(node):
                cost = heuristic(neighbor, goal)
                heapq.heappush(new_beam, (cost, neighbor))
        beam = heapq.nsmallest(beam_width, new_beam)
    return None
```

### 2. **Iterative Deepening A* (IDA*)**

- Combina a **busca em profundidade** com **heurísticas**.
- Utiliza **limites crescentes** baseados na função `f(n) = g(n) + h(n)`.

### 3. **Weighted A***

- Variante do A*, mas multiplica a heurística por um fator `w`:
    - `f(n) = g(n) + w * h(n)`
- Com `w > 1`, torna-se mais **rápido**, mas **menos preciso**.
- Usado quando o tempo é mais importante que a perfeição (ex: jogos ou robôs em tempo real).

## Exemplo de Aplicação: Planejamento de Rota com Beam Search

Imagine um robô de limpeza que decida o caminho mais eficiente até um ponto para recarregar. O mapa da casa é representado como um grafo. O algoritmo usa a **distância euclidiana** como heurística decidindo quais caminhos explorar primeiro, evitando paredes e obstáculos.

## Algoritmo de busca em ambientes complexos (Gradient Descent ou Hill Climbing)

### Ambiente Complexo — Algoritmo de Gradiente Descendente

```python
def gradient_descent(derivative_func, start, learning_rate, n_iterations):
    x = start
    for _ in range(n_iterations):
        grad = derivative_func(x)
        x = x - learning_rate * grad
    return x

# Exemplo: minimizar f(x) = x^2 → f'(x) = 2x
minimum = gradient_descent(lambda x: 2*x, start=10.0, learning_rate=0.1, n_iterations=100)
print(f"Mínimo encontrado: x = {minimum}, f(x) = {minimum**2}")

```

## O que são Algoritmos de Busca em Ambientes Complexos?

São algoritmos usados quando o espaço de busca é **muito grande, contínuo ou dinâmico**, o que torna inviável testar todas as soluções possíveis. Eles **buscam melhorar uma solução progressivamente**, explorando os arredores de uma solução atual com base em algum critério de melhoria.

## Principais Tipos

### 1. **Gradient Descent (Descida do Gradiente)**

- **Tipo:** Algoritmo de otimização contínua.
- **Utilização:** A função objetivo é **diferenciável** (ou seja, podemos calcular sua derivada).
- **Ideia:** Dado um ponto inicial, move-se **na direção do gradiente negativo** da função, que é a direção de maior "declínio" da função.

### Como funciona:

1. Escolhe-se uma solução inicial (ex: peso de um modelo).
2. Calcula-se o gradiente (a inclinação) da função no ponto atual.
3. Move-se na direção oposta ao gradiente.
4. Repete-se o processo até atingir um **mínimo local ou global**.

### Fórmula básica:   x = x - α * ∇f(x)

### **2. Busca Local (Hill Climbing, Simulated Annealing, etc.)**

- **Tipo:** Heurísticas de otimização baseadas em **melhorar uma solução atual**.
- **Diferente do Gradient Descent:** Não depende de derivadas, funciona com funções **não contínuas**.
- **Exemplo relevante:** **Simulated Annealing**, que permite aceitar piores soluções com alguma probabilidade para escapar de mínimos locais.

## Origem e Contexto Histórico

- O **Gradient Descent** surgiu no contexto do **cálculo multivariado** e foi adaptado para problemas de IA a partir da década de 1950.
- Tornou-se extremamente popular na **aprendizagem de máquina** com o surgimento das **redes neurais**.
- Métodos de busca local têm origem em técnicas de otimização clássica, mas evoluíram para lidar com problemas **não determinísticos e estocásticos** em IA.

## Aplicações na Inteligência Artificial

### Gradient Descent:

- **Aprendizado de máquina:** Treinamento de **redes neurais profundas**, **regressão logística**, **SVM**, etc.
- **Redução de erro:** Minimiza uma **função de custo**, como erro quadrático médio (MSE).
- **Modelos probabilísticos:** Ajuste de parâmetros em modelos como **Naive Bayes** ou **HMM**.

### Outros algoritmos de busca local:

- **Otimização de hiperparâmetros** (ex: número de neurônios, taxa de aprendizado).
- **Planejamento de ações** em IA (com Simulated Annealing ou algoritmos estocásticos).
- **IA em jogos e robótica**, onde o ambiente muda e não há tempo para uma busca exaustiva.

## Exemplos de Algoritmos

### 1. **Gradient Descent**

```python
# Minimizar a função f(x) = (x - 5)**2
def f(x):
    return (x - 5)**2

def grad_f(x):
    return 2 * (x - 5)

x = 0  # ponto inicial
lr = 0.1  # taxa de aprendizado

for _ in range(100):
    x -= lr * grad_f(x)

print(f"Mínimo encontrado em x = {x}")
#Converge para x = 5, que é o mínimo global da função.
```

### 2. **Simulated Annealing** (busca local com perturbações)

```python
import math
import random

def objective(x):
    return (x - 3)**2 + 2

x = random.uniform(-10, 10)
T = 1.0
T_min = 0.0001
alpha = 0.9

while T > T_min:
    new_x = x + random.uniform(-1, 1)
    delta = objective(new_x) - objective(x)
    if delta < 0 or random.random() < math.exp(-delta / T):
        x = new_x
    T *= alpha

print(f"Melhor solução encontrada: x = {x}")

```

## Importância dos Algoritmos em Ambientes Complexos

- São **fundamentais** para resolver problemas onde:
    - O espaço de busca é **enorme ou contínuo**.
    - A solução exata não é conhecida.
    - É necessário **tempo real** ou **adaptação constante** (ex: IA em jogos, carros autônomos).
- Estão na **base de muitos avanços modernos** em deep learning e inteligência artificial.

| Conceito | Gradient Descent | Busca Local |
| --- | --- | --- |
| Precisa de derivada? | Sim | Não |
| Tipo de função | Contínua | Arbitrária |
| Exemplo | Treinamento de rede neural | Planejamento em IA |
| Vantagem | Convergência rápida | Versátil para ambientes estocásticos |

---

# Algoritmo genético

```python
import random

def fitness(x):
    return -1 * (x - 3)**2 + 9  # Máximo em x = 3

def genetic_algorithm(population, generations, mutation_rate=0.1):
    for _ in range(generations):
        population.sort(key=fitness, reverse=True)
        next_gen = population[:2]  # elitismo

        while len(next_gen) < len(population):
            p1, p2 = random.sample(population[:4], 2)
            child = (p1 + p2) / 2
            if random.random() < mutation_rate:
                child += random.uniform(-1, 1)
            next_gen.append(child)

        population = next_gen

    best = max(population, key=fitness)
    return best

# Exemplo: encontrar o máximo da função - (x-3)^2 + 9
initial_pop = [random.uniform(0, 6) for _ in range(6)]
solution = genetic_algorithm(initial_pop, generations=20)
print(f"Solução genética: x = {solution}, f(x) = {fitness(solution)}")

```

## **O que são Algoritmos Genéticos?**

**Algoritmos Genéticos (AGs)** são técnicas de **otimização e busca** inspiradas nos processos de **seleção natural** e **evolução biológica** descritos por Charles Darwin. Eles pertencem à categoria de **algoritmos evolutivos**, que usam conceitos como **população**, **mutação**, **reprodução** e **seleção** para resolver problemas complexos.

## **Origem dos Algoritmos Genéticos**

- Foram propostos por **John Holland** na década de **1970**, na Universidade de Michigan.
- Holland criou uma estrutura matemática para entender os mecanismos da evolução natural e adaptá-los à resolução de problemas computacionais.
- Seu livro de 1975, *"Adaptation in Natural and Artificial Systems"*, é a base teórica dos AGs.

## **Como Funcionam os Algoritmos Genéticos?**

1. **Inicialização:** Uma população inicial de soluções (chamadas de indivíduos ou cromossomos) é gerada aleatoriamente.
2. **Avaliação (Fitness):** Cada indivíduo é avaliado por uma função de aptidão (fitness function) que mede quão boa é a solução.
3. **Seleção:** Indivíduos mais aptos têm mais chances de serem escolhidos para gerar descendentes.
4. **Crossover (Recombinação):** Partes dos cromossomos dos pais são combinadas para criar novos indivíduos.
5. **Mutação:** Pequenas alterações aleatórias são feitas nos descendentes para manter diversidade genética.
6. **Substituição:** Uma nova geração substitui (total ou parcialmente) a antiga.
7. **Iteração:** O processo se repete por várias gerações até que uma condição de parada seja satisfeita (ex: número máximo de gerações, solução suficientemente boa, etc).

## **Contextos de Aplicação dos Algoritmos Genéticos**

AGs são usados em **problemas de otimização complexos**, onde não há uma solução exata viável ou onde o espaço de busca é muito grande:

### Exemplos de aplicações:

- **Otimização de rotas:** problema do caixeiro viajante (TSP), roteamento de entregas.
- **Agendamento:** alocação de tarefas em fábricas ou escalas de funcionários.
- **Design de redes neurais:** otimização de hiperparâmetros de redes profundas.
- **Engenharia:** design de circuitos eletrônicos, estruturas mecânicas.
- **Bioinformática:** alinhamento de sequências genéticas, previsão de estrutura de proteínas.
- **Criptografia:** quebra de cifras por tentativa evolutiva.
- **Jogos e IA:** criar NPCs adaptativos ou estratégias vencedoras.

## **Por que Algoritmos Genéticos são Importantes?**

- **Robustez:** Funcionam bem mesmo com funções de custo não lineares, descontínuas ou com múltiplos ótimos locais.
- **Paralelismo natural:** Por usar populações, podem explorar várias soluções ao mesmo tempo.
- **Generalidade:** Podem ser aplicados a muitos tipos diferentes de problemas.
- **Não precisam de derivadas:** Diferente do gradiente descendente, AGs não exigem funções diferenciáveis.

## **Exemplos de Algoritmos Genéticos e Variantes**

1. **Algoritmo Genético Clássico (GA):**
    - Baseado nos operadores básicos: seleção, crossover e mutação.
2. **Algoritmo Genético com Elitismo:**
    - Garante que os melhores indivíduos de uma geração sobrevivem à próxima.
3. **Algoritmos Genéticos Multiobjetivo (ex: NSGA-II):**
    - Resolvem problemas com múltiplos critérios de otimização.
4. **Algoritmos Evolutivos Diferenciais:**
    - Variante que usa diferenças entre soluções para gerar novas.
5. **Algoritmos Genéticos com Codificação Real:**
    - Ao invés de bits, usam números reais (mais comum em otimização contínua).


## **Referências**

* RUSSELL, Stuart J.; NORVIG, Peter. *Inteligência Artificial*. 3. ed. Rio de Janeiro: Elsevier, 2013.
  (Título original: *Artificial Intelligence: A Modern Approach*)
* LUGER, George F. *Inteligência Artificial: estruturas e estratégias para a resolução de problemas complexos*. 6. ed. Pearson, 2009.
* GOLDBERG, David E. *Algoritmos Genéticos em Busca, Otimização e Aprendizado de Máquina*. Pearson Education, 1989.
* HOLLAND, John H. *Adaptation in Natural and Artificial Systems*. MIT Press, 1992.
* MITCHELL, Melanie. *An Introduction to Genetic Algorithms*. MIT Press, 1996.
* GOODFELLOW, Ian; BENGIO, Yoshua; COURVILLE, Aaron. *Deep Learning*. MIT Press, 2016. Disponível em: [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)
* HASTIE, Trevor; TIBSHIRANI, Robert; FRIEDMAN, Jerome. *The Elements of Statistical Learning*. Springer, 2009.
* OpenAI Blog – AI concepts and research: [https://openai.com/research](https://openai.com/research)
* Towards Data Science – Medium Publication: [https://towardsdatascience.com/](https://towardsdatascience.com/)
* Geeks for Geeks – AI/ML tutorials: [https://www.geeksforgeeks.org/fundamentals-of-artificial-intelligence/](https://www.geeksforgeeks.org/fundamentals-of-artificial-intelligence/)
* Stanford University – AI Course Materials: [https://cs221.stanford.edu/](https://cs221.stanford.edu/)

