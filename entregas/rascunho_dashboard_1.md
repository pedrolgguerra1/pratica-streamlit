# Rascunho 1 — Dashboard de distribuições

Objetivo: mostrar o **princípio distribucional** da EDA da saúde das baterias, com filtro por intervalo e comparação visual entre baterias saudáveis e degradadas (SoH < 80%).

```
+------------------------------------------------------------------+
|  SAÚDE DE BATERIAS DE VEÍCULOS ELÉTRICOS                         |
|  [Distribuições]  Relações                                       |
+---------------------------+--------------------------------------+
| BARRA LATERAL             | MÉTRICAS                             |
|                           |  N filtrado | % amostra | Saudável | |
|  Seletor de variável      |  Degradada                           |
|  [ cycle_count        v ] |                                      |
|                           | HISTOGRAMAS LADO A LADO              |
|  Slider de intervalo      |  [Saudável / azul] [Degradada / roxo]|
|  |----●==========●----|   |                                      |
|                           | BOXPLOT POR ESTADO                   |
|  Variável Y (outro tab)   |  outliers em vermelho                |
|                           |                                      |
|                           | TABELA: média, mediana, desvio, IQR  |
|                           | Shapiro-Wilk (subamostra n=500)      |
+---------------------------+--------------------------------------+
```

## Decisões de desenho

- Um único seletor de variável na barra lateral alimenta histograma, boxplot e tabela.
- O slider corta a amostra **antes** de desenhar, para o utilizador inspecionar caudas e outliers sem perder o contexto do restante dashboard.
- Cores herdadas do notebook: azul celeste = Saudável, violeta = Degradada, vermelho = outliers.
- Não se removem outliers: picos de temperatura e de `battery_stress_index` são sinal físico de degradação, não erro de digitação.
