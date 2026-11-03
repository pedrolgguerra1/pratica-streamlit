# Rascunho 2 — Dashboard de relações

Objetivo: mostrar o **princípio relacional** da EDA: associação entre duas variáveis quantitativas e entre fabricante e estado da bateria.

```
+------------------------------------------------------------------+
|  SAÚDE DE BATERIAS DE VEÍCULOS ELÉTRICOS                         |
|   Distribuições  [Relações]                                      |
+---------------------------+--------------------------------------+
| BARRA LATERAL (partilhada)| DISPERSÃO colorida por estado        |
|  Variável X = seletor     |  X: variável escolhida               |
|  Variável Y = 2.º seletor |  Y: variável complementar            |
|  Slider de intervalo em X |                                      |
|                           | MÉTRICAS                             |
|                           |  Spearman ρ | Pearson r              |
|                           |  Mann-Whitney U (X por estado)       |
|                           |                                      |
|                           | HEATMAP                              |
|                           |  fabricante × estado_bateria         |
|                           |  + tabela de contingência            |
+---------------------------+--------------------------------------+
```

## Decisões de desenho

- Spearman é a correlação principal porque o Shapiro-Wilk rejeitou normalidade nos sinais operacionais.
- A dispersão usa no máximo 2 500 pontos para manter a aplicação reativa com 20 000 linhas.
- O heatmap de fabricante reproduz a avaliação de fairness industrial do notebook: procurar disparidade de degradação entre montadoras/fabricantes, sem rebalancear a amostra.
- Os dois separadores partilham a mesma barra lateral para a navegação ser consistente.
