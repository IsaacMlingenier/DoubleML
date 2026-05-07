
# ============================================================
# Streamlit App: Double Machine Learning para tuberculosis pulmonar
# Versión mejorada con gráficos descriptivos y comparativos
#
# Modelos comparados:
#   1. Double Machine Learning - DoubleMLIRM
#   2. Regresión logística ajustada con G-computation
#   3. IPTW - Inverse Probability of Treatment Weighting
#   4. AIPW - Augmented Inverse Probability Weighting
#
# Instalación:
#   pip install -r requirements_double_ml_tb.txt
#
# Ejecución:
#   streamlit run app_double_ml_tb_v2.py
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path
from io import BytesIO

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

try:
    import doubleml as dml
    DOUBLEML_AVAILABLE = True
except Exception:
    DOUBLEML_AVAILABLE = False




def make_one_hot_encoder():
    """Crea OneHotEncoder compatible con versiones nuevas y antiguas de scikit-learn."""
    try:
        return make_one_hot_encoder()
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# ============================================================
# Configuración general
# ============================================================

st.set_page_config(
    page_title="Double ML - Tuberculosis Pulmonar",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 Análisis causal con Double Machine Learning en tuberculosis pulmonar")

st.markdown(
    """
    Esta aplicación permite cargar una base sintética de tuberculosis pulmonar y estimar el efecto causal del
    **inicio oportuno del tratamiento** sobre el **éxito terapéutico**, comparando Double ML con tres métodos adicionales.

    **Tratamiento causal sugerido:** `tratamiento_oportuno_7d`  
    **Resultado principal sugerido:** `exito_terapeutico`
    """
)


# ============================================================
# Funciones auxiliares
# ============================================================

def safe_numeric_binary(series):
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def prepare_model_matrix(df, y_col, d_col, x_cols):
    """
    Prepara matriz X codificada, Y y D.
    """
    work = df[[y_col, d_col] + x_cols].copy()
    work[y_col] = safe_numeric_binary(work[y_col])
    work[d_col] = safe_numeric_binary(work[d_col])
    work = work.dropna(subset=[y_col, d_col])

    y = work[y_col].astype(int).values
    d = work[d_col].astype(int).values
    X_raw = work[x_cols]

    categorical_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in X_raw.columns if c not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop"
    )

    X = preprocessor.fit_transform(X_raw)

    feature_names = []
    feature_names.extend(numeric_cols)

    if categorical_cols:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        feature_names.extend(ohe.get_feature_names_out(categorical_cols).tolist())

    X_df = pd.DataFrame(X, columns=feature_names, index=work.index)

    return work, X_df, y, d, preprocessor, numeric_cols, categorical_cols


def bootstrap_ci(values, alpha=0.05):
    values = np.array(values)
    lower = np.percentile(values, 100 * alpha / 2)
    upper = np.percentile(values, 100 * (1 - alpha / 2))
    return lower, upper


def fit_propensity_model(X, d):
    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model.fit(X, d)
    e = model.predict_proba(X)[:, 1]
    return model, np.clip(e, 0.03, 0.97)


def fit_outcome_model(X, d, y):
    Xd = np.column_stack([d, X])
    model = GradientBoostingClassifier(random_state=42)
    model.fit(Xd, y)
    return model


def predict_mu(model, X, d_value):
    d_vec = np.full(X.shape[0], d_value)
    Xd = np.column_stack([d_vec, X])
    return model.predict_proba(Xd)[:, 1]


def estimate_logistic_gcomp(X, d, y):
    """
    Regresión logística ajustada + G-computation.
    """
    Xd = np.column_stack([d, X])
    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model.fit(Xd, y)

    pred_observed = model.predict_proba(Xd)[:, 1]
    auc = roc_auc_score(y, pred_observed) if len(np.unique(y)) > 1 else np.nan

    mu1 = model.predict_proba(np.column_stack([np.ones(X.shape[0]), X]))[:, 1]
    mu0 = model.predict_proba(np.column_stack([np.zeros(X.shape[0]), X]))[:, 1]

    ate = np.mean(mu1 - mu0)
    return ate, auc


def estimate_iptw(X, d, y):
    """
    IPTW:
    ATE = E[D*Y/e(X)] - E[(1-D)*Y/(1-e(X))]
    """
    _, e = fit_propensity_model(X, d)
    ate = np.mean(d * y / e) - np.mean((1 - d) * y / (1 - e))
    weights = d / e + (1 - d) / (1 - e)
    return ate, e, weights


def estimate_aipw(X, d, y):
    """
    AIPW doubly robust.
    """
    _, e = fit_propensity_model(X, d)
    outcome_model = fit_outcome_model(X, d, y)

    mu1 = predict_mu(outcome_model, X, 1)
    mu0 = predict_mu(outcome_model, X, 0)

    aipw_scores = (
        mu1 - mu0
        + d * (y - mu1) / e
        - (1 - d) * (y - mu0) / (1 - e)
    )

    ate = np.mean(aipw_scores)
    se = np.std(aipw_scores) / np.sqrt(len(y))
    return ate, se


def estimate_double_ml_irm(X_df, d, y):
    """
    Double ML IRM para tratamiento binario.
    """
    if not DOUBLEML_AVAILABLE:
        raise ImportError("doubleml no está instalado. Ejecuta: pip install doubleml")

    df_dml = X_df.copy()
    df_dml["tratamiento"] = d
    df_dml["resultado"] = y

    dml_data = dml.DoubleMLData(
        df_dml,
        y_col="resultado",
        d_cols="tratamiento"
    )

    ml_g = RandomForestRegressor(
        n_estimators=120,
        max_depth=8,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )

    ml_m = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )

    model = dml.DoubleMLIRM(
        dml_data,
        ml_g=ml_g,
        ml_m=ml_m,
        n_folds=3,
        score="ATE"
    )

    model.fit()

    ci = model.confint(level=0.95)

    return {
        "ate": float(model.coef[0]),
        "se": float(model.se[0]),
        "pvalue": float(model.pval[0]),
        "ci_low": float(ci.iloc[0, 0]),
        "ci_high": float(ci.iloc[0, 1]),
        "summary": model.summary
    }


def estimate_bootstrap_for_methods(X, d, y, n_boot=50, sample_frac=0.7):
    """
    Bootstrap ligero para modelos comparativos.
    """
    rng = np.random.default_rng(42)
    n = len(y)
    size = int(n * sample_frac)

    gcomp_vals = []
    iptw_vals = []
    aipw_vals = []

    progress = st.progress(0, text="Calculando intervalos bootstrap...")

    for b in range(n_boot):
        idx = rng.choice(np.arange(n), size=size, replace=True)
        Xb = X[idx]
        db = d[idx]
        yb = y[idx]

        try:
            gcomp, _ = estimate_logistic_gcomp(Xb, db, yb)
            iptw, _, _ = estimate_iptw(Xb, db, yb)
            aipw, _ = estimate_aipw(Xb, db, yb)

            gcomp_vals.append(gcomp)
            iptw_vals.append(iptw)
            aipw_vals.append(aipw)
        except Exception:
            pass

        progress.progress((b + 1) / n_boot, text=f"Bootstrap {b + 1}/{n_boot}")

    progress.empty()

    return {
        "Regresión logística ajustada": bootstrap_ci(gcomp_vals) if len(gcomp_vals) > 5 else (np.nan, np.nan),
        "IPTW": bootstrap_ci(iptw_vals) if len(iptw_vals) > 5 else (np.nan, np.nan),
        "AIPW": bootstrap_ci(aipw_vals) if len(aipw_vals) > 5 else (np.nan, np.nan)
    }


def outcome_rate_plot(df, group_col, outcome_col, title, x_title, y_title):
    tmp = (
        df.groupby(group_col)[outcome_col]
        .mean()
        .reset_index()
    )
    tmp[outcome_col] = tmp[outcome_col].astype(float)

    fig = px.bar(
        tmp,
        x=group_col,
        y=outcome_col,
        text=tmp[outcome_col].round(3),
        title=title
    )
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title
    )
    return fig


# ============================================================
# Sidebar y carga robusta de datos
# ============================================================

@st.cache_data(show_spinner=False)
def load_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    Lee un CSV subido desde el navegador de forma robusta.
    Usa bytes y BytesIO para evitar errores de permisos o de objeto UploadedFile.
    """
    encodings = ["utf-8", "utf-8-sig", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(BytesIO(file_bytes), encoding=enc, low_memory=False)
        except UnicodeDecodeError as e:
            last_error = e
        except Exception as e:
            last_error = e

    raise ValueError(f"No fue posible leer el CSV cargado. Último error: {last_error}")


@st.cache_data(show_spinner=False)
def load_local_csv(path: str) -> pd.DataFrame:
    """Lee un CSV local cuando el archivo está en la carpeta del proyecto o en una ruta válida."""
    csv_path = Path(path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")
    return pd.read_csv(csv_path, low_memory=False)


st.sidebar.header("1. Cargar datos")

modo_carga = st.sidebar.radio(
    "Seleccione el método de carga",
    [
        "Subir archivo CSV manualmente",
        "Leer CSV local en la carpeta del proyecto"
    ],
    index=0
)

df = None

if modo_carga == "Subir archivo CSV manualmente":
    uploaded_file = st.sidebar.file_uploader(
        "Suba el archivo CSV desde su equipo",
        type=["csv"],
        accept_multiple_files=False,
        help="Descargue el CSV en su equipo y súbalo desde aquí. No use enlaces sandbox:/ ni URL externas."
    )

    if uploaded_file is None:
        st.info(
            "Suba el archivo CSV para iniciar el análisis. "
            "Si está usando Streamlit Cloud, cargue el archivo desde su computador con este botón."
        )
        st.stop()

    try:
        df = load_uploaded_csv(uploaded_file.getvalue())
    except Exception as e:
        st.error(f"No fue posible leer el CSV cargado: {e}")
        st.stop()

else:
    local_path = st.sidebar.text_input(
        "Ruta local del CSV",
        value="tb_pulmonar_sintetica_double_ml_100k.csv"
    )

    try:
        df = load_local_csv(local_path)
    except FileNotFoundError:
        st.warning(
            "No se encontró el archivo CSV. Colóquelo en la misma carpeta de esta app "
            "o escriba la ruta completa. En Streamlit Cloud, es más seguro usar la opción de subir archivo manualmente."
        )
        st.stop()
    except Exception as e:
        st.error(f"No fue posible leer el CSV local: {e}")
        st.stop()

# Limpieza básica de nombres de columnas para evitar espacios accidentales
if df is not None:
    df.columns = df.columns.astype(str).str.strip()

st.sidebar.success(f"Datos cargados correctamente: {df.shape[0]:,} filas y {df.shape[1]:,} columnas")

st.sidebar.header("2. Configurar análisis")

default_y = "exito_terapeutico" if "exito_terapeutico" in df.columns else df.columns[0]
default_d = "tratamiento_oportuno_7d" if "tratamiento_oportuno_7d" in df.columns else df.columns[1]

y_col = st.sidebar.selectbox(
    "Variable resultado Y",
    df.columns.tolist(),
    index=df.columns.tolist().index(default_y)
)

d_col = st.sidebar.selectbox(
    "Variable tratamiento D",
    df.columns.tolist(),
    index=df.columns.tolist().index(default_d)
)

default_x = [
    "edad", "sexo", "zona_residencia", "regimen_afiliacion",
    "grupo_poblacional", "indice_vulnerabilidad",
    "vih", "diabetes", "desnutricion", "tabaquismo",
    "alcohol_riesgo", "epoc", "enfermedad_renal",
    "condicion_ingreso", "resistencia_farmacologica",
    "bk_inicial", "cultivo_inicial", "hospitalizacion_inicial",
    "severidad_inicial", "demora_sintomas_diagnostico_dias",
    "misma_ips_seguimiento"
]

default_x = [c for c in default_x if c in df.columns]

x_cols = st.sidebar.multiselect(
    "Variables de control X",
    [c for c in df.columns if c not in [y_col, d_col]],
    default=default_x
)

max_rows = min(len(df), 100000)

n_rows = st.sidebar.slider(
    "Número de casos para modelar",
    min_value=1000,
    max_value=max_rows,
    value=min(25000, max_rows),
    step=1000
)

run_bootstrap = st.sidebar.checkbox(
    "Calcular intervalos bootstrap para modelos comparativos",
    value=False
)

n_boot = st.sidebar.slider(
    "Número de réplicas bootstrap",
    min_value=20,
    max_value=150,
    value=50,
    step=10
)

run_button = st.sidebar.button("Ejecutar análisis causal", type="primary")


# ============================================================
# Tabs
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Vista general",
    "Gráficos descriptivos",
    "Modelos causales",
    "Interpretación"
])


# ============================================================
# Tab 1: Vista general
# ============================================================

with tab1:
    st.subheader("Vista previa de la base")
    st.dataframe(df.head(20), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Filas", f"{df.shape[0]:,}")
    c2.metric("Columnas", f"{df.shape[1]:,}")

    if y_col in df:
        c3.metric(f"Media de {y_col}", f"{pd.to_numeric(df[y_col], errors='coerce').mean():.3f}")

    if d_col in df:
        c4.metric(f"Media de {d_col}", f"{pd.to_numeric(df[d_col], errors='coerce').mean():.3f}")

    st.subheader("Variables disponibles")
    st.write(", ".join(df.columns.tolist()))


# ============================================================
# Tab 2: Gráficos descriptivos
# ============================================================

with tab2:
    st.subheader("Exploración visual de la base sintética")

    col_a, col_b = st.columns(2)

    with col_a:
        if "resultado_final" in df.columns:
            fig = px.histogram(
                df,
                x="resultado_final",
                title="Distribución del resultado final",
                text_auto=True
            )
            fig.update_layout(
                xaxis_title="Resultado final",
                yaxis_title="Número de casos"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        if "resistencia_farmacologica" in df.columns:
            fig = px.histogram(
                df,
                x="resistencia_farmacologica",
                color=y_col if y_col in df.columns else None,
                barmode="group",
                title="Resistencia farmacológica según resultado"
            )
            fig.update_layout(
                xaxis_title="Resistencia farmacológica",
                yaxis_title="Número de casos"
            )
            st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        if "tratamiento_oportuno_7d" in df.columns and "exito_terapeutico" in df.columns:
            tmp = df.copy()
            tmp["tratamiento_oportuno_label"] = tmp["tratamiento_oportuno_7d"].map({
                0: "No oportuno",
                1: "Oportuno"
            })

            fig = outcome_rate_plot(
                tmp,
                "tratamiento_oportuno_label",
                "exito_terapeutico",
                "Éxito terapéutico según inicio oportuno del tratamiento",
                "Inicio del tratamiento",
                "Proporción de éxito terapéutico"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_d:
        if "tratamiento_oportuno_7d" in df.columns and "muerte" in df.columns:
            tmp = df.copy()
            tmp["tratamiento_oportuno_label"] = tmp["tratamiento_oportuno_7d"].map({
                0: "No oportuno",
                1: "Oportuno"
            })

            fig = outcome_rate_plot(
                tmp,
                "tratamiento_oportuno_label",
                "muerte",
                "Mortalidad según inicio oportuno del tratamiento",
                "Inicio del tratamiento",
                "Proporción de mortalidad"
            )
            st.plotly_chart(fig, use_container_width=True)

    col_e, col_f = st.columns(2)

    with col_e:
        if "edad" in df.columns:
            fig = px.histogram(
                df,
                x="edad",
                color="tratamiento_oportuno_7d" if "tratamiento_oportuno_7d" in df.columns else None,
                nbins=40,
                title="Distribución de edad según tratamiento oportuno"
            )
            fig.update_layout(
                xaxis_title="Edad",
                yaxis_title="Número de pacientes"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_f:
        if "demora_diagnostico_tratamiento_dias" in df.columns and "resultado_final" in df.columns:
            sample_box = df.sample(min(10000, len(df)), random_state=42)
            fig = px.box(
                sample_box,
                x="resultado_final",
                y="demora_diagnostico_tratamiento_dias",
                title="Demora entre diagnóstico y tratamiento según desenlace"
            )
            fig.update_layout(
                xaxis_title="Resultado final",
                yaxis_title="Días entre diagnóstico e inicio de tratamiento"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Riesgo clínico visual")

    if "edad" in df.columns and "severidad_inicial" in df.columns and "exito_terapeutico" in df.columns:
        sample_scatter = df.sample(min(6000, len(df)), random_state=42)
        fig = px.scatter(
            sample_scatter,
            x="edad",
            y="severidad_inicial",
            color="exito_terapeutico",
            opacity=0.55,
            title="Edad y severidad inicial según éxito terapéutico",
            hover_data=[
                c for c in [
                    "vih", "diabetes", "desnutricion",
                    "resistencia_farmacologica", "tratamiento_oportuno_7d"
                ] if c in sample_scatter.columns
            ]
        )
        fig.update_layout(
            xaxis_title="Edad",
            yaxis_title="Severidad inicial"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Subgrupos clínicos relevantes")

    subgroup_cols = [c for c in ["vih", "diabetes", "desnutricion", "epoc"] if c in df.columns]

    if subgroup_cols and "exito_terapeutico" in df.columns:
        sub_data = []

        for col in subgroup_cols:
            temp = (
                df.groupby(col)["exito_terapeutico"]
                .mean()
                .reset_index()
            )
            temp["variable"] = col
            temp["categoria"] = temp[col].astype(str)
            temp = temp[["variable", "categoria", "exito_terapeutico"]]
            sub_data.append(temp)

        sub_df = pd.concat(sub_data, ignore_index=True)

        fig = px.bar(
            sub_df,
            x="variable",
            y="exito_terapeutico",
            color="categoria",
            barmode="group",
            text=sub_df["exito_terapeutico"].round(3),
            title="Éxito terapéutico por subgrupos clínicos"
        )
        fig.update_layout(
            xaxis_title="Subgrupo clínico",
            yaxis_title="Proporción de éxito terapéutico"
        )
        st.plotly_chart(fig, use_container_width=True)

    if "tiempo_hasta_desenlace_dias" in df.columns and "muerte" in df.columns:
        st.subheader("Tiempo hasta desenlace")
        sample_time = df.sample(min(12000, len(df)), random_state=42)

        fig = px.histogram(
            sample_time,
            x="tiempo_hasta_desenlace_dias",
            color="muerte",
            nbins=60,
            barmode="overlay",
            title="Distribución del tiempo hasta desenlace según mortalidad"
        )
        fig.update_layout(
            xaxis_title="Días hasta desenlace",
            yaxis_title="Número de pacientes"
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Tab 3: Modelos causales
# ============================================================

with tab3:
    st.subheader("Comparación de modelos causales")

    st.markdown(
        """
        Los modelos estiman el efecto promedio del tratamiento:

        **ATE = E[Y(1) - Y(0)]**

        En esta aplicación, el ATE representa el cambio promedio esperado en la probabilidad de éxito terapéutico
        cuando el paciente recibe tratamiento oportuno frente a no recibirlo.
        """
    )

    if not x_cols:
        st.warning("Selecciona al menos una variable de control X en el panel lateral.")
        st.stop()

    if run_button:
        with st.spinner("Preparando datos y entrenando modelos..."):
            df_sample = df.sample(n=n_rows, random_state=42).reset_index(drop=True)

            work, X_df, y, d, preprocessor, numeric_cols, categorical_cols = prepare_model_matrix(
                df_sample,
                y_col=y_col,
                d_col=d_col,
                x_cols=x_cols
            )

            X = X_df.values

        st.success(f"Matriz lista: {X.shape[0]:,} casos y {X.shape[1]:,} variables después de codificación.")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Resultado promedio Y", f"{np.mean(y):.3f}")
        c2.metric("Tratamiento promedio D", f"{np.mean(d):.3f}")
        c3.metric("Controles originales", f"{len(x_cols)}")
        c4.metric("Controles codificados", f"{X.shape[1]}")

        results = []

        with st.spinner("Ejecutando Double Machine Learning..."):
            try:
                dml_result = estimate_double_ml_irm(X_df, d, y)

                results.append({
                    "Modelo": "Double ML - IRM",
                    "ATE": dml_result["ate"],
                    "Error estándar": dml_result["se"],
                    "IC 95% inferior": dml_result["ci_low"],
                    "IC 95% superior": dml_result["ci_high"],
                    "p-valor": dml_result["pvalue"]
                })

                st.success("Double ML ejecutado correctamente.")

            except Exception as e:
                st.error(f"No fue posible ejecutar Double ML: {e}")
                dml_result = None

        with st.spinner("Ejecutando regresión logística ajustada..."):
            try:
                gcomp_ate, gcomp_auc = estimate_logistic_gcomp(X, d, y)

                results.append({
                    "Modelo": "Regresión logística ajustada",
                    "ATE": gcomp_ate,
                    "Error estándar": np.nan,
                    "IC 95% inferior": np.nan,
                    "IC 95% superior": np.nan,
                    "p-valor": np.nan
                })

            except Exception as e:
                st.error(f"Error en regresión logística ajustada: {e}")
                gcomp_auc = np.nan

        with st.spinner("Ejecutando IPTW..."):
            try:
                iptw_ate, propensity, weights = estimate_iptw(X, d, y)

                results.append({
                    "Modelo": "IPTW",
                    "ATE": iptw_ate,
                    "Error estándar": np.nan,
                    "IC 95% inferior": np.nan,
                    "IC 95% superior": np.nan,
                    "p-valor": np.nan
                })

            except Exception as e:
                st.error(f"Error en IPTW: {e}")
                propensity = None
                weights = None

        with st.spinner("Ejecutando AIPW..."):
            try:
                aipw_ate, aipw_se = estimate_aipw(X, d, y)

                results.append({
                    "Modelo": "AIPW",
                    "ATE": aipw_ate,
                    "Error estándar": aipw_se,
                    "IC 95% inferior": aipw_ate - 1.96 * aipw_se,
                    "IC 95% superior": aipw_ate + 1.96 * aipw_se,
                    "p-valor": np.nan
                })

            except Exception as e:
                st.error(f"Error en AIPW: {e}")

        result_df = pd.DataFrame(results)

        if run_bootstrap and len(results) > 1:
            with st.spinner("Calculando intervalos bootstrap para modelos comparativos..."):
                ci_dict = estimate_bootstrap_for_methods(X, d, y, n_boot=n_boot)

            for model_name, (low, high) in ci_dict.items():
                mask = result_df["Modelo"] == model_name
                result_df.loc[mask, "IC 95% inferior"] = low
                result_df.loc[mask, "IC 95% superior"] = high

        st.subheader("Resultados comparativos")

        st.dataframe(
            result_df.style.format({
                "ATE": "{:.4f}",
                "Error estándar": "{:.4f}",
                "IC 95% inferior": "{:.4f}",
                "IC 95% superior": "{:.4f}",
                "p-valor": "{:.4f}"
            }),
            use_container_width=True
        )

        fig = px.bar(
            result_df,
            x="Modelo",
            y="ATE",
            error_y=result_df["IC 95% superior"] - result_df["ATE"],
            error_y_minus=result_df["ATE"] - result_df["IC 95% inferior"],
            title="Comparación del efecto causal estimado ATE",
            text=result_df["ATE"].round(4)
        )
        fig.update_layout(
            yaxis_title="ATE estimado",
            xaxis_title="Modelo"
        )
        st.plotly_chart(fig, use_container_width=True)

        if propensity is not None:
            st.subheader("Diagnóstico de propensity score")

            prop_df = pd.DataFrame({
                "propensity_score": propensity,
                "tratamiento": d
            })

            fig = px.histogram(
                prop_df,
                x="propensity_score",
                color="tratamiento",
                nbins=40,
                barmode="overlay",
                title="Distribución del propensity score por grupo de tratamiento"
            )
            fig.update_layout(
                xaxis_title="Propensity score",
                yaxis_title="Número de pacientes"
            )
            st.plotly_chart(fig, use_container_width=True)

        if weights is not None:
            st.subheader("Diagnóstico de pesos IPTW")

            w1, w2, w3 = st.columns(3)

            w1.metric("Peso promedio", f"{np.mean(weights):.2f}")
            w2.metric("Peso p95", f"{np.percentile(weights, 95):.2f}")
            w3.metric("Peso máximo", f"{np.max(weights):.2f}")

            fig = px.histogram(
                pd.DataFrame({"peso_iptw": weights}),
                x="peso_iptw",
                nbins=60,
                title="Distribución de pesos IPTW"
            )
            fig.update_layout(
                xaxis_title="Peso IPTW",
                yaxis_title="Número de pacientes"
            )
            st.plotly_chart(fig, use_container_width=True)

            if np.max(weights) > 20:
                st.warning(
                    "Hay pesos IPTW muy altos. Esto puede indicar baja superposición entre tratados y no tratados."
                )

    else:
        st.info("Configura las variables en el panel lateral y pulsa **Ejecutar análisis causal**.")


# ============================================================
# Tab 4: Interpretación
# ============================================================

with tab4:
    st.subheader("Guía de interpretación")

    st.markdown(
        """
        ### 1. Interpretación del ATE

        El **ATE** representa el cambio promedio esperado en la probabilidad del resultado cuando el tratamiento
        cambia de 0 a 1.

        Por ejemplo, si:

        - `Y = exito_terapeutico`
        - `D = tratamiento_oportuno_7d`
        - `ATE = 0.08`

        Entonces, bajo los supuestos del modelo, el inicio oportuno del tratamiento se asocia con un aumento promedio
        de **8 puntos porcentuales** en la probabilidad de éxito terapéutico.

        ### 2. Lectura de los modelos

        - **Double ML - IRM:** modelo causal principal. Usa machine learning para controlar confusión de forma flexible.
        - **Regresión logística ajustada:** referencia clásica e interpretable.
        - **IPTW:** pondera los casos según su probabilidad de recibir tratamiento.
        - **AIPW:** estimador doblemente robusto que combina modelo de resultado y modelo de tratamiento.

        ### 3. Qué revisar antes de afirmar causalidad

        1. Las variables de control deben ocurrir antes del tratamiento.
        2. No deben faltar confusores clínicos importantes.
        3. Debe existir superposición entre tratados y no tratados.
        4. No se deben incluir mediadores posteriores al tratamiento como controles principales.
        5. Debe hacerse análisis de sensibilidad.

        ### 4. Precaución metodológica

        Variables como `adherencia_alta`, `conversion_bacteriologica`, `bk_mes2_negativa` o `controles_cumplidos`
        pueden ser mediadoras si se estudia el efecto del inicio oportuno. Para el análisis causal principal,
        conviene revisar cuidadosamente si deben excluirse del conjunto X.

        ### 5. Recomendación para artículo

        Para un artículo Q1, se recomienda complementar esta app con:

        - DAG causal;
        - análisis de balance;
        - análisis de sensibilidad;
        - heterogeneidad del efecto por VIH, diabetes, resistencia, edad y ruralidad;
        - análisis de supervivencia;
        - validación con datos reales anonimizados.
        """
    )
