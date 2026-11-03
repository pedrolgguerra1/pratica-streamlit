# Documentação de uso de IA

Trabalho de EDA e dashboards Streamlit sobre o dataset *EV Battery Health Prediction Dataset (20k)*. Uso de IA permitido pelo Plano de Ensino; os prompts abaixo registam o que foi pedido nesta conversa para gerar a estrutura da aplicação.

## Contexto

- Aluno: Pedr0 (Pedro)
- Repositório: `https://github.com/pedrolgguerra1/pratica-streamlit`
- Notebook de origem: `notebooks/Cópia_de_Aula_11_Prática_EDA (2).ipynb`
- Dataset: `srisyra02/ev-battery-health-prediction-dataset-20k` (carregado com `kagglehub`)

## Prompt 1 — Estrutura da aplicação Streamlit

Pedido para gerar o código completo da aplicação a partir da EDA já feita no notebook, com:

- `app.py` em Streamlit, a descarregar o dataset automaticamente via `kagglehub`
- dois dashboards (separadores) alinhados com os rascunhos
- seletor de variáveis e slider de intervalo na barra lateral
- `requirements.txt` com `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `kagglehub` e `streamlit`
- `README.md` com a descrição do projeto e conclusões da EDA

## Prompt 2 — Rascunhos dos dashboards

Pedido para produzir os rascunhos de interface (`rascunho_dashboard_1.md` e `rascunho_dashboard_2.md`) a partir dos princípios da EDA:

1. Distribuições univariadas por estado da bateria (Saudável vs. Degradada, SoH < 80%)
2. Relações bivariadas (dispersão, Spearman, heatmap fabricante × estado)

## Prompt 3 — Sequência de publicação

Pedido para executar os 6 passos do enunciado:

1. Repositório público no GitHub e pastas `notebooks/`, `entregas/`, `prompts/`
2. Guardar o `.ipynb`, os rascunhos e esta documentação de prompts
3. Criar `app.py`, `requirements.txt` e `README.md` na raiz
4. Testar localmente com `pip install -r requirements.txt` e `streamlit run app.py`
5. Enviar para o GitHub com a mensagem `Estrutura completa da EDA e dashboards no Streamlit`
6. Fazer o deploy no Streamlit Cloud (`share.streamlit.io`) e colocar o URL público no topo do `README.md`

## Prompt 4 — Texto pronto para entrega

Confirmação explícita: gerar já o texto de `prompts/documentacao_ia.md` e um `README.md` completo e personalizado com as conclusões da EDA, e **fazer tudo** o que está na sequência (estrutura, teste local, push e deploy).

## O que a IA produziu a partir destes prompts

- Estrutura de pastas exigida pelo enunciado
- `app.py` com cache do dataset, controlos na sidebar e dois separadores
- Rascunhos em Markdown (wireframes em ASCII)
- `README.md` com conclusões da EDA (alvo SoH < 80%, outliers mantidos, ausência de normalidade, Spearman/Mann-Whitney)
- Este ficheiro de documentação de prompts
