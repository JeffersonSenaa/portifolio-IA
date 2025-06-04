# Portfólio 3 - Constraint Satisfaction Problems - CSP

## Introdução aos CSPs

Problemas de Satisfação de Restrições (CSPs), trata-se de uma classe de problemas onde o objetivo é encontrar valores para um conjunto de variáveis que satisfa um conjunto de restrições. Formalmente, um CSP consiste em:

- Um conjunto de variáveis X = {X₁, X₂, ..., Xₙ}
- Um conjunto de domínios D = {D₁, D₂, ..., Dₙ}, onde cada Dᵢ é o conjunto de valores possíveis para Xᵢ
- Um conjunto de restrições C = {C₁, C₂, ..., Cₘ} que especificam combinações válidas de valores

## 1.1 Exemplo Prático: Solucionador de Sudoku Simples usando CSP

Neste exemplo, implemento um solucionador de Sudoku utilizando técnicas de CSP.

- Variáveis: 81 células do tabuleiro (9x9)
- Domínios: Números de 1 a 9 para cada célula
- Restrições: 
  - Cada linha contém números únicos de 1 a 9
  - Cada coluna contém números únicos de 1 a 9
  - Cada subgrade 3x3 contém números únicos de 1 a 9

```python
class SudokuCSP:
    def __init__(self, board):
        self.board = board
        self.variables = [(i, j) for i in range(9) for j in range(9)]
        self.domains = {var: set(range(1, 10)) for var in self.variables}
        self.constraints = self._get_constraints()
        
    def _get_constraints(self):
        constraints = []
        # Restrições de linha
        for i in range(9):
            for j in range(9):
                for k in range(j + 1, 9):
                    constraints.append(((i, j), (i, k)))
        
        # Restrições de coluna
        for j in range(9):
            for i in range(9):
                for k in range(i + 1, 9):
                    constraints.append(((i, j), (k, j)))
        
        # Restrições de subgrade 3x3
        for block_i in range(3):
            for block_j in range(3):
                cells = [(3*block_i + i, 3*block_j + j) 
                        for i in range(3) for j in range(3)]
                for i, cell1 in enumerate(cells):
                    for cell2 in cells[i+1:]:
                        constraints.append((cell1, cell2))
        
        return constraints

    def is_consistent(self, var, value, assignment):
        for neighbor in self.get_neighbors(var):
            if neighbor in assignment and assignment[neighbor] == value:
                return False
        return True

    def get_neighbors(self, var):
        neighbors = set()
        for constraint in self.constraints:
            if var in constraint:
                neighbors.add(constraint[1] if constraint[0] == var else constraint[0])
        return neighbors
```

## Estratégias de Resolução de CSPs

### 1. Backtracking Search

Técnica fundamental para resolver CSPs. Consiste em tentar atribuir valores às variáveis uma por uma, voltando atrás quando encontra um conflito.

```python
def backtracking_search(csp):
    def backtrack(assignment):
        if len(assignment) == len(csp.variables):
            return assignment
        
        var = select_unassigned_variable(csp, assignment)
        for value in order_domain_values(csp, var, assignment):
            if csp.is_consistent(var, value, assignment):
                assignment[var] = value
                result = backtrack(assignment)
                if result is not None:
                    return result
                del assignment[var]
        return None
    
    return backtrack({})
```

### 2. Forward Checking

Técnica de propagação de restrições que mantém a consistência dos domínios das variáveis não atribuídas.

```python
def forward_checking(csp, var, value, assignment, domains):
    for neighbor in csp.get_neighbors(var):
        if neighbor not in assignment:
            if value in domains[neighbor]:
                domains[neighbor].remove(value)
                if not domains[neighbor]:
                    return False
    return True
```

## Estratégias de Seleção de Variáveis

### 1. Minimum Remaining Values (MRV)

Seleciona a variável com o menor número de valores possíveis no domínio.

```python
def select_unassigned_variable_mrv(csp, assignment):
    unassigned = [var for var in csp.variables if var not in assignment]
    return min(unassigned, key=lambda var: len(csp.domains[var]))
```

### 2. Degree Heuristic

Seleciona a variável que está envolvida no maior número de restrições com outras variáveis não atribuídas.

```python
def select_unassigned_variable_degree(csp, assignment):
    unassigned = [var for var in csp.variables if var not in assignment]
    return max(unassigned, key=lambda var: len(csp.get_neighbors(var)))
```

## Estratégias de Ordenação de Valores

### 1. Least Constraining Value (LCV)

Ordena os valores de forma a minimizar o impacto nas outras variáveis.

```python
def order_domain_values_lcv(csp, var, assignment):
    def count_conflicts(value):
        conflicts = 0
        for neighbor in csp.get_neighbors(var):
            if neighbor not in assignment and value in csp.domains[neighbor]:
                conflicts += 1
        return conflicts
    
    return sorted(csp.domains[var], key=count_conflicts)
```

## Contribuições e Melhorias

### 1. Algoritmo de Consistência de Arco (AC-3)

Implementação do algoritmo AC-3 para manter a consistência de arco:

```python
def ac3(csp):
    queue = csp.constraints.copy()
    while queue:
        (xi, xj) = queue.pop()
        if revise(csp, xi, xj):
            if not csp.domains[xi]:
                return False
            for xk in csp.get_neighbors(xi):
                if xk != xj:
                    queue.append((xk, xi))
    return True

def revise(csp, xi, xj):
    revised = False
    for x in csp.domains[xi].copy():
        if not any(csp.is_consistent(xi, x, {xj: y}) for y in csp.domains[xj]):
            csp.domains[xi].remove(x)
            revised = True
    return revised
```

### 2. Algoritmo de Busca Local

Implementação de um algoritmo de busca local para CSPs:

```python
def min_conflicts(csp, max_steps=1000):
    assignment = {var: random.choice(list(csp.domains[var])) 
                 for var in csp.variables}
    
    for _ in range(max_steps):
        if is_solution(csp, assignment):
            return assignment
        
        var = select_conflicted_variable(csp, assignment)
        value = min_conflicts_value(csp, var, assignment)
        assignment[var] = value
    
    return None

def select_conflicted_variable(csp, assignment):
    conflicted = []
    for var in csp.variables:
        if not csp.is_consistent(var, assignment[var], assignment):
            conflicted.append(var)
    return random.choice(conflicted)

def min_conflicts_value(csp, var, assignment):
    return min(csp.domains[var], 
              key=lambda value: count_conflicts(csp, var, value, assignment))
```

## Análise de Desempenho

Nas diferentes estratégias utilizadas, realizamos testes com vários tabuleiros de Sudoku:

1. **Backtracking com MRV + LCV**: 
   - Mais eficiente para problemas pequenos
   - Garante solução ótima
   - Pode ser lento para problemas grandes

2. **Forward Checking**:
   - Reduz significativamente o número de backtrackings
   - Melhor para problemas com muitas restrições
   - Custo adicional de manutenção dos domínios

3. **AC-3**:
   - Muito eficiente para problemas com restrições binárias
   - Pode resolver alguns problemas sem backtracking
   - Custo computacional maior

4. **Min-Conflicts**:
   - Excelente para problemas grandes
   - Não garante solução ótima
   - Muito rápido em média

## Demais Exemplos Práticos

## 2. Problema das N-Rainhas (N-Queens)
O objetivo deste problema é posicionar N rainhas em um tabuleiro NxN de forma que nenhuma rainha possa atacar outra.

```python
class NQueensCSP:
    def __init__(self, n):
        self.n = n
        self.variables = list(range(n))  # Cada variável representa uma coluna
        self.domains = {var: list(range(n)) for var in self.variables}
        self.constraints = self._get_constraints()
    
    def _get_constraints(self):
        constraints = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                constraints.append((i, j))
        return constraints
    
    def is_consistent(self, var, value, assignment):
        for col, row in assignment.items():
            if col == var:
                continue
            # Verifica ataques na mesma linha
            if row == value:
                return False
            # Verifica ataques nas diagonais
            if abs(col - var) == abs(row - value):
                return False
        return True
```

## 3. Problema de Coloração de Mapas (Map Coloring)
Um problema onde o objetivo é colorir um mapa com um número mínimo de cores, para garantir que regiões adjacentes tenham cores diferentes.

```python
class MapColoringCSP:
    def __init__(self, regions, adjacencies, colors):
        self.variables = regions
        self.domains = {var: set(colors) for var in self.variables}
        self.constraints = self._get_constraints(adjacencies)
    
    def _get_constraints(self, adjacencies):
        constraints = []
        for region1, neighbors in adjacencies.items():
            for region2 in neighbors:
                if (region2, region1) not in constraints:
                    constraints.append((region1, region2))
        return constraints
    
    def is_consistent(self, var, value, assignment):
        for neighbor in self.get_neighbors(var):
            if neighbor in assignment and assignment[neighbor] == value:
                return False
        return True
```

## 4. Problema de Horários (Scheduling)
Um problema comum em escolas e universidades para criar horários de aulas, evitando conflitos de professores, salas e turmas.

```python
class SchedulingCSP:
    def __init__(self, classes, teachers, rooms, time_slots):
        self.variables = classes
        self.domains = {
            var: [(room, time) for room in rooms for time in time_slots]
            for var in self.variables
        }
        self.teacher_assignments = {class_: teacher for class_, teacher in teachers}
        self.constraints = self._get_constraints()
    
    def _get_constraints(self):
        constraints = []
        # Restrições de professor
        for class1 in self.variables:
            for class2 in self.variables:
                if class1 < class2:
                    if self.teacher_assignments[class1] == self.teacher_assignments[class2]:
                        constraints.append((class1, class2))
        return constraints
    
    def is_consistent(self, var, value, assignment):
        room, time = value
        # Verifica conflito de sala
        for class_, (r, t) in assignment.items():
            if r == room and t == time:
                return False
        # Verifica conflito de professor
        teacher = self.teacher_assignments[var]
        for class_, (r, t) in assignment.items():
            if t == time and self.teacher_assignments[class_] == teacher:
                return False
        return True
```

### 5. Problema de Roteamento de Veículos (Vehicle Routing)
Um problema de otimização onde o objetivo é encontrar rotas ótimas para uma frota de veículos atendendo a um conjunto de clientes.

```python
class VehicleRoutingCSP:
    def __init__(self, customers, vehicles, capacity_constraints):
        self.variables = customers
        self.domains = {
            var: [(v, pos) for v in vehicles for pos in range(len(customers))]
            for var in self.variables
        }
        self.capacity_constraints = capacity_constraints
        self.constraints = self._get_constraints()
    
    def _get_constraints(self):
        constraints = []
        # Restrições de capacidade
        for v in self.vehicles:
            customer_assignments = [c for c in self.variables 
                                 if self.domains[c][0] == v]
            if sum(self.capacity_constraints[c] for c in customer_assignments) > v.capacity:
                constraints.append(customer_assignments)
        return constraints
    
    def is_consistent(self, var, value, assignment):
        vehicle, position = value
        # Verifica capacidade
        current_load = sum(self.capacity_constraints[c] 
                         for c, (v, _) in assignment.items() 
                         if v == vehicle)
        if current_load + self.capacity_constraints[var] > vehicle.capacity:
            return False
        # Verifica posição
        for c, (v, pos) in assignment.items():
            if v == vehicle and pos == position:
                return False
        return True
```

### 6. Problema de Montagem de Produtos (Assembly Line Balancing)
Um problema de otimização onde o objetivo é distribuir tarefas de montagem entre estações de trabalho, para minimizar o tempo total de ciclo.

```python
class AssemblyLineCSP:
    def __init__(self, tasks, stations, task_times, precedences):
        self.variables = tasks
        self.domains = {var: list(range(stations)) for var in self.variables}
        self.task_times = task_times
        self.precedences = precedences
        self.constraints = self._get_constraints()
    
    def _get_constraints(self):
        constraints = []
        # Restrições de precedência
        for task1, task2 in self.precedences:
            constraints.append((task1, task2))
        return constraints
    
    def is_consistent(self, var, value, assignment):
        # Verifica precedência
        for task1, task2 in self.precedences:
            if task1 == var and task2 in assignment:
                if assignment[task2] <= value:
                    return False
            elif task2 == var and task1 in assignment:
                if assignment[task1] >= value:
                    return False
        # Verifica tempo de ciclo
        station_time = sum(self.task_times[t] 
                         for t, s in assignment.items() 
                         if s == value)
        if station_time + self.task_times[var] > self.cycle_time:
            return False
        return True
```
## Conclusão 

Cada um desses projetos demonstra diferentes aspectos dos CSPs:

1. **Solucionador de Sudoku**: Demonstra restrições complexas em ambiente multivariavel, como exclusividade de valores em linhas, colunas e blocos 3x3
2. **N-Rainhas**: Demonstra restrições binárias simples e simétricas, em que nenhuma rainha pode atacar outra na mesma linha, coluna ou diagonal.
3. **Coloração de Mapas**: Mostra como modelar problemas com restrições de adjacência, onde regiões vizinhas não podem ter a mesma cor.
4. **Scheduling**: Representação de problemas com restrições múltiplas e heterogêneas, como conflitos de horário, recursos limitados (salas, professores) e preferências.
5. **Roteamento de Veículos**: Envolve restrições de capacidade, tempo e sequência, como limite de carga dos veículos e janelas de entrega.
6. **Montagem de Produtos**: Demonstra restrições de precedência, sincronização e tempo, comuns em ambientes industriais e de produção.


Dessa forma, os Problemas de Satisfação de Restrições (CSPs) demonstram ser uma ferramenta poderosa para modelar e resolver uma ampla variedade de problemas que envolvem restrições complexas. Durante a análise e desenvolvimento de soluções baseadas em CSPs, observou-se a importância da escolha adequada das estratégias de seleção de variáveis e valores, visto que essas decisões influenciam diretamente a eficiência e a viabilidade da resolução. Além disso, as técnicas de propagação de restrições mostraram ter um impacto significativo na redução do espaço de busca, permitindo identificar inconsistências precocemente e, assim, acelerar o processo de resolução. Também foi possível perceber que diferentes algoritmos apresentam desempenhos distintos dependendo do contexto e da estrutura do problema, o que reforça a necessidade de uma análise criteriosa para a seleção do método mais apropriado.

Entre as contribuições desenvolvidas ao longo do projeto, destacam-se a implementação de múltiplas estratégias de resolução, a realização de uma análise comparativa de desempenho entre essas estratégias e a adaptação de abordagens específicas para problemas clássicos, como o Sudoku, demonstrando a flexibilidade e aplicabilidade dos CSPs em cenários diversos.

## Referências

1. Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4ª ed.). Pearson.
2. Dechter, R. (2003). *Constraint Processing*. Morgan Kaufmann.
3. Kumar, V. (1992). Algorithms for Constraint Satisfaction Problems: A Survey. *AI Magazine*, 13(1), 32–44.
4. Mackworth, A. K. (1977). Consistency in networks of relations. *Artificial Intelligence*, 8(1), 99–118.
5. Haralick, R. M., & Elliott, G. L. (1980). Increasing tree search efficiency for constraint satisfaction problems. *Artificial Intelligence*, 14(3), 263–313.
6. Bessiere, C. (2006). Constraint propagation. In *Handbook of Constraint Programming*, Elsevier, 29–83.
7. Apt, K. R. (2003). *Principles of Constraint Programming*. Cambridge University Press.
8. Rossi, F., van Beek, P., & Walsh, T. (Eds.). (2006). *Handbook of Constraint Programming*. Elsevier.
