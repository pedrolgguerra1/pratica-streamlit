import glob
import os

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats

st.set_page_config(
    page_title="EDA — Saúde de Baterias de VEs",
    page_icon="🔋",
    layout="wide",
)

COR_NEUTRA = "lightgray"
COR_CONTEXTO = "#38bdf8"
COR_DESTAQUE = "#7c3aed"
COR_ALERTA = "#ef4444"
PALETA_ESTADO = {"Saudável": COR_CONTEXTO, "Degradada": COR_DESTAQUE}

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 4.5)

KAGGLE_DATASET = "srisyra02/ev-battery-health-prediction-dataset-20k"
VARS_QUANT_PADRAO = [
    "battery_capacity_kwh",
    "odometer_km",
    "cycle_count",
    "cell_voltage_avg",
    "temperature_variance",
    "battery_stress_index",
    "predicted_remaining_life_cycles",
    "battery_health_percent",
]


@st.cache_data(show_spinner="A descarregar o dataset do Kaggle...")
def carregar_dados() -> pd.DataFrame:
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {path}")

    df = pd.read_csv(csv_files[0])

    soh_col = None
    for candidato in ("state_of_health", "battery_health_percent"):
        if candidato in df.columns:
            soh_col = candidato
            break

    if soh_col is not None:
        df["estado_bateria"] = np.where(
            df[soh_col] < 80.0, "Degradada", "Saudável"
        )
        df.loc[df[soh_col].isna(), "estado_bateria"] = np.nan
    elif "battery_failure" in df.columns:
        df["estado_bateria"] = np.where(
            df["battery_failure"] == 1, "Degradada", "Saudável"
        )
    else:
        df["estado_bateria"] = "Saudável"

    return df


def variaveis_numericas(df: pd.DataFrame) -> list[str]:
    numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    prioridade = [c for c in VARS_QUANT_PADRAO if c in numericas]
    restantes = [c for c in numericas if c not in prioridade]
    return prioridade + restantes


def aplicar_filtro(df: pd.DataFrame, coluna: str, intervalo: tuple[float, float]) -> pd.DataFrame:
    minimo, maximo = intervalo
    return df[df[coluna].between(minimo, maximo)]


def figura_histograma(df: pd.DataFrame, coluna: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    for ax, classe in zip(axes, ("Saudável", "Degradada")):
        subset = df[df["estado_bateria"] == classe]
        cor = PALETA_ESTADO[classe]
        sns.histplot(subset, x=coluna, kde=True, color=cor, ax=ax)
        ax.set_title(f"{coluna} — {classe}")
        ax.set_ylabel("Frequência")
    fig.suptitle("Distribuição por estado da bateria", fontsize=13)
    fig.tight_layout()
    return fig


def figura_boxplot(df: pd.DataFrame, coluna: str):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.boxplot(
        data=df,
        x="estado_bateria",
        y=coluna,
        hue="estado_bateria",
        palette=PALETA_ESTADO,
        legend=False,
        flierprops=dict(
            markerfacecolor=COR_ALERTA,
            markeredgecolor=COR_ALERTA,
            marker="o",
        ),
        ax=ax,
    )
    ax.set_title(f"Boxplot de {coluna} por estado da bateria")
    ax.set_xlabel("Estado da bateria")
    fig.tight_layout()
    return fig


def figura_dispersao(df: pd.DataFrame, x: str, y: str):
    amostra = df.sample(min(2500, len(df)), random_state=42) if len(df) > 2500 else df
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(
        data=amostra,
        x=x,
        y=y,
        hue="estado_bateria",
        palette=PALETA_ESTADO,
        alpha=0.55,
        ax=ax,
    )
    ax.set_title(f"Dispersão: {x} vs. {y}")
    ax.legend(title="Estado da bateria")
    fig.tight_layout()
    return fig


def figura_heatmap_fabricante(df: pd.DataFrame):
    col_cat = (
        "battery_manufacturer"
        if "battery_manufacturer" in df.columns
        else df.select_dtypes(exclude=[np.number]).columns[0]
    )
    contingencia = pd.crosstab(df[col_cat], df["estado_bateria"])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(contingencia, annot=True, fmt="d", cmap="magma", ax=ax)
    ax.set_title(f"{col_cat} × estado da bateria")
    fig.tight_layout()
    return fig, col_cat, contingencia


st.title("Saúde de baterias de veículos elétricos")
st.caption(
    "Dashboard interativo da EDA sobre o conjunto "
    "`srisyra02/ev-battery-health-prediction-dataset-20k`."
)

try:
    df = carregar_dados()
except Exception as exc:
    st.error(
        "Não foi possível descarregar o dataset via kagglehub. "
        "Confirme a ligação à internet. Se o Kaggle exigir consentimento, "
        "configure as credenciais no ambiente."
    )
    st.exception(exc)
    st.stop()

numericas = variaveis_numericas(df)
if not numericas:
    st.error("O dataset não contém colunas numéricas para explorar.")
    st.stop()

st.sidebar.header("Controlos")
variavel = st.sidebar.selectbox("Seletor de variável", numericas, index=0)

serie = df[variavel].dropna()
vmin, vmax = float(serie.min()), float(serie.max())
if vmin == vmax:
    vmax = vmin + 1.0

intervalo = st.sidebar.slider(
    "Intervalo da variável selecionada",
    min_value=vmin,
    max_value=vmax,
    value=(vmin, vmax),
    key=f"intervalo_{variavel}",
)

outras = [c for c in numericas if c != variavel]
variavel_y = st.sidebar.selectbox(
    "Variável do eixo Y (separador Relações)",
    outras if outras else numericas,
    index=0,
)

df_filtrado = aplicar_filtro(df, variavel, intervalo)

n_total = len(df)
n_filtrado = len(df_filtrado)
contagem_estado = df_filtrado["estado_bateria"].value_counts(dropna=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Registos filtrados", f"{n_filtrado:,}".replace(",", "."))
col2.metric("% da amostra", f"{100 * n_filtrado / n_total:.1f}%")
col3.metric("Saudável", int(contagem_estado.get("Saudável", 0)))
col4.metric("Degradada (SoH < 80%)", int(contagem_estado.get("Degradada", 0)))

tab_dist, tab_rel = st.tabs(["Distribuições", "Relações"])

with tab_dist:
    st.subheader("Análise univariada")
    st.write(
        "Este separador reproduz o princípio **distribucional** da EDA: "
        "histograma por classe, boxplot com outliers (vermelho) e métricas robustas. "
        "O alvo `estado_bateria` usa o limiar automotivo de fim de vida (SoH < 80%)."
    )

    if df_filtrado[variavel].dropna().empty:
        st.warning("Não há valores válidos desta variável no intervalo escolhido.")
    else:
        st.pyplot(figura_histograma(df_filtrado, variavel), width="stretch")
        st.pyplot(figura_boxplot(df_filtrado, variavel), width="stretch")

        estat = df_filtrado.groupby("estado_bateria", dropna=True)[variavel].agg(
            ["count", "mean", "median", "std"]
        )
        estat["IQR"] = df_filtrado.groupby("estado_bateria", dropna=True)[variavel].apply(
            lambda s: s.quantile(0.75) - s.quantile(0.25)
        )
        st.dataframe(estat.round(2), width="stretch")

        amostra = (
            df_filtrado[variavel]
            .dropna()
            .sample(min(500, df_filtrado[variavel].dropna().shape[0]), random_state=42)
        )
        if len(amostra) >= 8:
            stat_sw, p_sw = stats.shapiro(amostra)
            st.info(
                f"Shapiro-Wilk em subamostra de {len(amostra)} observações de `{variavel}`: "
                f"W = {stat_sw:.4f}, p-valor = {p_sw:.4g}. "
                "Na EDA, os sinais operacionais não seguem normalidade estrita; "
                "por isso privilegiamos mediana, IQR e testes não paramétricos."
            )

with tab_rel:
    st.subheader("Análise relacional")
    st.write(
        "Este separador cobre o princípio **relacional**: dispersão entre duas "
        "variáveis quantitativas, correlação de Spearman (mais segura sem normalidade) "
        "e associação entre fabricante e estado da bateria."
    )

    df_xy = df_filtrado.dropna(subset=[variavel, variavel_y, "estado_bateria"])
    if df_xy.empty:
        st.warning("Não há pares válidos no intervalo escolhido.")
    else:
        st.pyplot(figura_dispersao(df_xy, variavel, variavel_y), width="stretch")

        rho, p_rho = stats.spearmanr(df_xy[variavel], df_xy[variavel_y])
        r_p, p_p = stats.pearsonr(df_xy[variavel], df_xy[variavel_y])
        m1, m2 = st.columns(2)
        m1.metric("Spearman ρ", f"{rho:.3f}", help=f"p-valor = {p_rho:.4g}")
        m2.metric("Pearson r", f"{r_p:.3f}", help=f"p-valor = {p_p:.4g}")

        saudavel = df_xy.loc[df_xy["estado_bateria"] == "Saudável", variavel]
        degradada = df_xy.loc[df_xy["estado_bateria"] == "Degradada", variavel]
        if len(saudavel) > 0 and len(degradada) > 0:
            stat_u, p_u = stats.mannwhitneyu(saudavel, degradada, alternative="two-sided")
            st.write(
                f"**Mann-Whitney U** para `{variavel}` entre Saudável e Degradada: "
                f"U = {stat_u:,.0f}, p-valor = {p_u:.4g}."
            )

    fig_hm, col_cat, contingencia = figura_heatmap_fabricante(
        df_filtrado.dropna(subset=["estado_bateria"])
    )
    st.pyplot(fig_hm, width="stretch")
    st.caption(
        f"Tabela de contingência `{col_cat}` × estado da bateria "
        "(amostra já filtrada pelo slider da barra lateral)."
    )
    st.dataframe(contingencia, width="stretch")

st.divider()
st.markdown(
    "Fonte dos dados: [EV Battery Health Prediction Dataset (20k)]"
    "(https://www.kaggle.com/datasets/srisyra02/ev-battery-health-prediction-dataset-20k). "
    "Notebook de EDA em `notebooks/`."
)
