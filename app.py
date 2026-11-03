import copy
import glob
import os
import kagglehub
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats

def _corrigir_deepcopy_path(self, memo=None):
    if memo is None:
        memo = {}
    cls = type(self)
    p = cls.__new__(cls)
    memo[id(self)] = p
    for k, v in self.__dict__.items():
        setattr(p, k, copy.deepcopy(v, memo))
    p._readonly = False
    return p

mpath.Path.__deepcopy__ = _corrigir_deepcopy_path

st.set_page_config(
    page_title="EDA — Saúde de Baterias de VEs",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

COR_NEUTRA = "lightgray"
COR_CONTEXTO = "#38bdf8"
COR_DESTAQUE = "#7c3aed"
COR_ALERTA = "#ef4444"
PALETA_ESTADO = {"Saudável": COR_CONTEXTO, "Degradada": COR_DESTAQUE}

sns.set_theme(style="whitegrid")

KAGGLE_DATASET = "srisyra02/ev-battery-health-prediction-dataset-20k"

@st.cache_data(show_spinner="A descarregar e a preparar o dataset do Kaggle...")
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
        df["estado_bateria"] = np.where(df[soh_col] < 80.0, "Degradada", "Saudável")
        df.loc[df[soh_col].isna(), "estado_bateria"] = np.nan
    elif "battery_failure" in df.columns:
        df["estado_bateria"] = np.where(df["battery_failure"] == 1, "Degradada", "Saudável")
    else:
        df["estado_bateria"] = "Saudável"

    cols_sensiveis = ["vehicle_id", "battery_serial"]
    df.drop(columns=[c for c in cols_sensiveis if c in df.columns], inplace=True)

    return df

try:
    df_raw = carregar_dados()
except Exception as exc:
    st.error("Erro ao carregar os dados. Verifique a ligação à internet.")
    st.exception(exc)
    st.stop()

todas_numericas = df_raw.select_dtypes(include=[np.number]).columns.tolist()
vars_excluir = ["battery_failure", "manufacturing_year"]
vars_quant = [c for c in todas_numericas if c not in vars_excluir]

st.sidebar.header("⚙️ Painel de Controlo")
st.sidebar.markdown("Filtros globais para a exploração e os dashboards.")

vars_selecionadas = st.sidebar.multiselect(
    "1. Selecione as Variáveis Quantitativas:",
    options=vars_quant,
    default=vars_quant[:5] if len(vars_quant) >= 5 else vars_quant,
    help="Escolha as variáveis para análise no Laboratório e Mapas de Calor."
)

st.sidebar.markdown("---")

var_filtro = st.sidebar.selectbox(
    "Variável de referência para o filtro:",
    options=[c for c in ["odometer_km", "cycle_count", "battery_stress_index", "vehicle_age_years"] if c in df_raw.columns],
    index=0
)

min_val = float(df_raw[var_filtro].min())
max_val = float(df_raw[var_filtro].max())

intervalo_slider = st.sidebar.slider(
    f"2. Intervalo de exclusão ({var_filtro}):",
    min_value=min_val,
    max_value=max_val,
    value=(min_val, max_val),
    help="Deslize para restringir a amostra e excluir valores extremos da avaliação."
)

df_filtrado = df_raw[
    (df_raw[var_filtro] >= intervalo_slider[0]) & 
    (df_raw[var_filtro] <= intervalo_slider[1])
].copy()

st.sidebar.markdown("---")
st.sidebar.caption(f"📊 **Registos ativos:** {len(df_filtrado):,} de {len(df_raw):,}")

st.title("🔋 Plataforma de Telemetria e Saúde de Baterias de VEs")
st.markdown("Análise exploratória avançada e monitoramento preditivo para frotas de veículos elétricos.")

tab1, tab2 = st.tabs([
    "🔬 Dashboard 1: Laboratório de Distribuição e Correlação",
    "🚛 Dashboard 2: KPIs & Gestão de Frota (Público-Alvo)"
])

with tab1:
    st.subheader("🔬 Laboratório de Análise de Distribuição e Correlação")
    st.markdown(
        "Explore a estrutura multivariada das variáveis operacionais. "
        "Utilize o painel lateral para ajustar o recorte de dados."
    )

    if len(vars_selecionadas) < 2:
        st.warning("⚠️ Selecione pelo menos **2 variáveis quantitativas** no painel lateral para habilitar o mapa de calor.")
    else:
        c_cor, c_desc = st.columns([1.1, 1])

        with c_cor:
            st.markdown("#### Mapas de Calor para Correlação")
            metodo_corr = st.radio("Método:", ["pearson", "spearman"], horizontal=True, index=1)
            
            corr_mat = df_filtrado[vars_selecionadas].corr(method=metodo_corr)
            fig_hm, ax_hm = plt.subplots(figsize=(6, 4.5))
            mask = np.triu(np.ones_like(corr_mat, dtype=bool))
            sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax_hm)
            ax_hm.set_title(f"Correlação ({metodo_corr.capitalize()})", fontsize=11)
            fig_hm.tight_layout()
            st.pyplot(fig_hm)
            plt.close(fig_hm)

        with c_desc:
            st.markdown("#### Estatística Descritiva & Assimetria")
            desc_df = df_filtrado[vars_selecionadas].describe().T
            desc_df["skewness"] = df_filtrado[vars_selecionadas].skew()
            desc_df["kurtosis"] = df_filtrado[vars_selecionadas].kurtosis()
            tabela_resumo = desc_df[["mean", "50%", "std", "skewness", "kurtosis"]].rename(columns={"50%": "median"}).round(2)
            st.dataframe(tabela_resumo, use_container_width=True, height=330)

        st.markdown("---")
        st.markdown("#### 📈 Histogramas por Variável Quantitativa Avaliada")
        
        for col in vars_selecionadas:
            fig_h, axes_h = plt.subplots(1, 2, figsize=(11, 3.5), sharex=True)
            for ax, (cls, cor) in zip(axes_h, PALETA_ESTADO.items()):
                sub = df_filtrado[df_filtrado["estado_bateria"] == cls]
                sns.histplot(data=sub, x=col, kde=True, color=cor, ax=ax, element="step")
                ax.set_title(f"{col} — {cls}")
                ax.set_ylabel("Frequência")
            fig_h.tight_layout()
            st.pyplot(fig_h)
            plt.close(fig_h)

with tab2:
    st.subheader("🚛 Painel Executivo: Gestores de Frota & Engenheiros de BMS")
    st.markdown(
        "**Público-Alvo:** Equipa de Operações de Frotas Elétricas e Engenharia de Fiabilidade.\n"
        "*Foco:* Monitorização de risco de degradação prematura, ocorrências de falhas críticas e planeamento de substituição de packs."
    )
    st.markdown("---")
    
    st.markdown("#### 📌 KPIs Estratégicos (Métricas Próprias)")
    
    total_f = len(df_filtrado)
    if total_f > 0:
        p_deg = (df_filtrado["estado_bateria"] == "Degradada").mean() * 100
        p_falha = df_filtrado["battery_failure"].mean() * 100 if "battery_failure" in df_filtrado.columns else 0.0
        stress_medio = df_filtrado["battery_stress_index"].mean() if "battery_stress_index" in df_filtrado.columns else 0.0
        ciclos_med = df_filtrado["predicted_remaining_life_cycles"].median() if "predicted_remaining_life_cycles" in df_filtrado.columns else 0.0
        
        if "BMS_warning_count" in df_filtrado.columns:
            p_bms = (df_filtrado["BMS_warning_count"] >= 2).mean() * 100
        else:
            p_bms = 0.0
    else:
        p_deg = p_falha = stress_medio = ciclos_med = p_bms = 0.0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("1. Taxa de Degradação", f"{p_deg:.1f}%", help="Percentual com SoH < 80%.")
    k2.metric("2. Falhas Críticas", f"{p_falha:.2f}%", help="Baterias em estado de falha registada.")
    k3.metric("3. Índice Estresse Médio", f"{stress_medio:.1f} pts", help="Média do Battery Stress Index.")
    k4.metric("4. Mediana Vida Restante", f"{ciclos_med:,.0f}", help="Ciclos estimados antes do fim de vida.")
    k5.metric("5. Alertas BMS Elevados", f"{p_bms:.1f}%", help="Veículos com 2 ou mais avisos no BMS.")

    st.markdown("---")
    st.markdown("#### 📊 Visuais de Suporte Operacional")

    v1, v2 = st.columns(2)

    with v1:
        st.markdown("##### 1. Distribuição de Saúde por Fabricante")
        if "battery_manufacturer" in df_filtrado.columns:
            tab_fab = pd.crosstab(df_filtrado["battery_manufacturer"], df_filtrado["estado_bateria"], normalize="index") * 100
            fig_v1, ax_v1 = plt.subplots(figsize=(6, 3.8))
            tab_fab.plot(kind="bar", stacked=True, color=[COR_CONTEXTO, COR_DESTAQUE], ax=ax_v1)
            ax_v1.set_ylabel("Percentagem (%)")
            ax_v1.set_xlabel("Fabricante da Bateria")
            ax_v1.legend(title="Estado")
            ax_v1.tick_params(axis='x', rotation=25)
            fig_v1.tight_layout()
            st.pyplot(fig_v1)
            plt.close(fig_v1)

    with v2:
        st.markdown("##### 2. Estresse da Bateria por Tipo de Tração")
        col_drive = "drive_type" if "drive_type" in df_filtrado.columns else "vehicle_type"
        if col_drive in df_filtrado.columns and "battery_stress_index" in df_filtrado.columns:
            fig_v2, ax_v2 = plt.subplots(figsize=(6, 3.8))
            sns.boxplot(data=df_filtrado, x=col_drive, y="battery_stress_index", hue="estado_bateria", palette=PALETA_ESTADO, ax=ax_v2)
            ax_v2.set_ylabel("Battery Stress Index")
            ax_v2.set_xlabel("Tipo de Tração")
            ax_v2.legend(title="Estado")
            fig_v2.tight_layout()
            st.pyplot(fig_v2)
            plt.close(fig_v2)

    st.markdown("##### 3. Relação Operacional: Quilometragem vs. Contagem de Ciclos de Carga")
    if all(c in df_filtrado.columns for c in ["odometer_km", "cycle_count"]):
        fig_v3, ax_v3 = plt.subplots(figsize=(11, 4))
        amostra_v3 = df_filtrado.sample(min(2000, len(df_filtrado)), random_state=42)
        sns.scatterplot(
            data=amostra_v3,
            x="odometer_km",
            y="cycle_count",
            hue="estado_bateria",
            palette=PALETA_ESTADO,
            alpha=0.6,
            ax=ax_v3
        )
        ax_v3.set_xlabel("Quilometragem (Odometer km)")
        ax_v3.set_ylabel("Contagem de Ciclos (Cycle Count)")
        ax_v3.legend(title="Estado da Bateria")
        fig_v3.tight_layout()
        st.pyplot(fig_v3)
        plt.close(fig_v3)

st.divider()
st.caption("Repositório configurado para entrega académica individual de EDA & Streamlit.")