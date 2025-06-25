# Portifólio 4

## Agentes Lógicos 

Os **agentes lógicos** ou também denominados **agentes baseados em conhecimento**, representam uma categoria fundamental de agentes artificiais empregando a **lógica formal** como alicerce para a representar conhecimento e raciocínio. Em sua arquitetura, há concebição que possibilita a tomada de decisões fundamentadas numa compreensão estruturada e inferencial do ambiente, o que diferencia significativamente dos agentes puramente reativos, que operam com base em respostas diretas a estímulos imediatos.

---

### **Representação do Conhecimento: A Fundação dos Agentes Lógicos**

Um agente lógico, tem como capacidade central a **Representação do Conhecimento**. Por sua vez, este é componente que busca permitir que o agente codifique informações sobre o mundo de forma explícita e simbólica, utilizando **linguagens formais baseadas em lógica**, predominantemente a **lógica proposicional** e a **lógica de primeira ordem (LPO)**.

* **Lógica Proposicional:** Utilização de proposições (afirmações que podem ser verdadeiras ou falsas, dependendo do contexto) e conectivos lógicos como **"e" (and)**, **"ou" (or)**, **"não" (not)**, **"implica" (if...then)** e **"se e somente se" (iff)** na construção de sentenças complexas. É útil para representar fatos atômicos e suas inter-relações booleanas.
  Exemplo: *"É dia" e "Está sol" implica "Vou à praia".*

* **Lógica de Primeira Ordem (LPO):** Expande a lógica proposicional introduzindo **predicados**, **variáveis**, **quantificadores** como **"para todo"** e **"existe"**, além de **funções**. Isso confere à LPO um poder de expressão muito maior que permite representar relações entre objetos, propriedades de objetos e generalizações.
  Exemplo: *Para toda pessoa x, se x é uma pessoa, então x é mortal.*
  É uma capacidade crucial para modelar ambientes complexos, onde há entidades, atributos e relações que precisam ser representados com diferentes níveis de detalhamento.


A partir disso, o conhecimento é armazenado numa **Base de Conhecimento (BC)**, sendo um conjunto de sentenças (axiomas e fatos) expressas na linguagem lógica escolhida. Cada sentença na Base de Conhecimento contribui para a compreensão do agente sobre o estado do ambiente e as regras que o governam.

---

### **Mecanismo de Inferência: O Raciocínio Dedutivo**

O **Mecanismo de Inferência** é o motor do raciocínio em um agente baseado em conhecimento. Ele busca operar sobre a Base de Conhecimento, aplicando **regras de inferência** que deduzam novas sentenças (conclusões) a partir de sentenças existentes. Este processo é análogo ao raciocínio humano, onde novas verdades são derivadas de premissas conhecidas.

* **Regras de Inferência:** São procedimentos formais que garantem que, se as premissas forem verdadeiras, a conclusão derivada também será. Exemplos comuns incluem:
    * **Modus Ponens:** Se tivermos as sentenças $A$ e $A \Rightarrow B$, podemos inferir $B$.
    * **Resolução:** Um método de inferência completo e semi-decidível para a LPO, amplamente utilizado em provadores de teoremas automatizados.
* **Completude e Decidibilidade:** A escolha de um sistema lógico e das regras de inferência impactam a completude (se todas as verdades podem ser provadas) e a decidibilidade (se existe um algoritmo que sempre termina e determina a verdade de qualquer sentença). A lógica proposicional é decidível, enquanto a LPO é semi-decidível (se uma sentença é verdadeira, o algoritmo eventualmente a provará; se for falsa, pode não terminar).
* **Busca:** O processo de inferência, muitas vezes envolve uma busca heurística ou algorítmica no espaço de estados para encontrar uma prova ou uma solução para uma consulta. Algoritmos como busca em largura (BFS), busca em profundidade (DFS) e A* podem ser adaptados para atender a este propósito.

---

### **Robustez em Ambientes Parcialmente Observáveis e Dinâmicos**

Uma das grandes vantagens dos agentes lógicos é a sua aptidão para operar em **ambientes parcialmente observáveis**. Nesses cenários, nem todas as informações relevantes estão diretamente acessíveis através dos sensores do agente. Sendo assim, o agente lógico compensa essa limitação ao utilizar seu conhecimento prévio e suas capacidades de inferência para deduzir o estado oculto do mundo ou prever eventos futuros. Por exemplo, se um sensor falha, o agente pode inferir a probabilidade de um evento com base em outras observações e no conhecimento das leis do domínio.

A **flexibilidade e adaptabilidade** são características intrínsecas devido à natureza explícita da Base de Conhecimento. Pois alterações no ambiente ou nos objetivos do agente podem ser acomodadas através da modificação ou adição de sentenças lógicas à Base de conhecimento, sem a necessidade de reengenharia completa do agente. Essa modularidade tende a facilitar a manutenção e a evolução do sistema.

---

### **Raciocínio Proativo e Planejamento Orientado a Metas**

OS agentes puramente reativos, respondem a estímulos de forma condicionada. Porém, os agentes lógicos exibem **raciocínio proativo e orientado a metas**. Essa diferença, garante a capacidade de formular planos complexos, que são sequências de ações, para alcançar objetivos definidos. Este processo de planejamento envolve:

1.  **Definir o Estado Inicial:** Conhecimento atual do agente sobre o mundo.
2.  **Definir o Estado Objetivo:** Configuração desejada do mundo que o agente busca alcançar.
3.  **Ações:** Descrição das ações que o agente pode executar e seus efeitos no mundo (condições de pré-requisito e pós-condições).
4.  **Buscar o Espaço de Estados:** Mecanismo de inferência, busca uma sequência de ações que transforme o estado inicial no estado objetivo, e que considere as regras e restrições do ambiente. Algoritmos de planejamento como STRIPS, ADL, ou técnicas de Graphplan e SAT-based planning são empregados para este fim.

---

### **Ciclo Operacional de um Agente Lógico**

O funcionamento de um agente lógico pode ser encapsulado em um ciclo iterativo:

1.  **Percepção:** Recebimento de novas **percepções** do ambiente através de seus respectivos sensores. Essas percepções, são convertidas em sentenças lógicas e adicionadas à Base de Conhecimento.
2.  **Atualização da Base de Conhecimento (KB update):** Atualização da Base de Conhecimento incorpotando as novas percepções, garantindo que o conhecimento do agente reflita o estado atual do ambiente. Isso pode envolver adição, modificação ou remoção de sentenças.
3.  **Raciocínio/Inferência:** O mecanismo de inferência tem o objetivo de processar a Base de Conhecimento para:
    * Derivar novas verdades sobre o ambiente.
    * Responder a consultas (e.g., "Qual é a melhor ação a tomar?").
    * Detectar inconsistências.
    * Gerar um plano de ações para alcançar um objetivo específico.
4.  **Ação:** Com base nas conclusões do raciocínio, o agente busca seleciona e executa uma **ação** no ambiente através de seus atuadores. Esta ação pode alterar o estado do ambiente e gerar novas percepções no próximo ciclo.

---

### **Aplicações e Relevância**

Agentes lógicos são fundamentais em diversas áreas da IA:

* **Sistemas Especialistas:** Utilizam bases de conhecimento extensas e regras de inferência que emulam o raciocínio de especialistas humanos em domínios específicos (e.g., diagnóstico médico, configuração de sistemas).
* **Planejamento Automatizado:** Desenvolvimento de sequências de ações para robôs, sistemas autônomos ou para gerenciar projetos complexos.
* **Processamento de Linguagem Natural:** A lógica de primeira ordem pode ser usada na representação do significado semântico de frases, permitindo que máquinas compreendam e raciocinem sobre o texto.
* **Representação de Conhecimento na Web Semântica:** Fundamentam tecnologias como ontologias e linguagens como OWL permitindo que máquinas entendam o significado dos dados na web.

---

## **Referências**

* RUSSELL, Stuart J.; NORVIG, Peter. *Inteligência Artificial*. 3. ed. Rio de Janeiro: Elsevier, 2013. (Título original: *Artificial Intelligence: A Modern Approach*)
* LUGER, George F. *Inteligência Artificial: estruturas e estratégias para a resolução de problemas complexos*. 6. ed. Pearson, 2009.
* NILSSON, Nils J. *Artificial Intelligence: A New Synthesis*. Morgan Kaufmann, 1998.
* BRATKO, Ivan. *Prolog Programming for Artificial Intelligence*. 4. ed. Pearson Education, 2011.