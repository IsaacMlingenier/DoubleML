# ============================================================
# Streamlit App: Double Machine Learning para tuberculosis pulmonar
# Versión corregida y robusta
#
# Modelos comparados:
#   1. Double Machine Learning - DoubleMLIRM
#   2. Regresión logística ajustada + G-computation
#   3. IPTW - Inverse Probability of Treatment Weighting
#   4. AIPW - Augmented Inverse Probability Weighting
#
# Ejecución local:
#   pip install -r requirements.txt
#   streamlit run app_doubleml_tb_streamlit_corregida.py
# ============================================================

import warnings
warnings.filterwarnings("ignore")

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import doubleml as dml
    DOUBLEML_AVAILABLE = True
except Exception:
    DOUBLEML_AVAILABLE = False


# ============================================================
# Configuración general
# ============================================================

st.set_page_config(
    page_title="DoubleML - Tuberculosis Pulmonar",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 Ensayo de Double Machine Learning en tuberculosis pulmonar")

st.markdown(
    """
    Esta aplicación permite ensayar un análisis causal con **Double Machine Learning** para tuberculosis pulmonar.
    Puede trabajar con una base simulada generada dentro de la app, con un archivo local o con un archivo cargado
    manualmente.

    **Modelo principal:** `DoubleMLIRM`, adecuado cuando el tratamiento o exposición `D` es binario.
    """
)


# ============================================================
# Utilidades de lectura de datos
# ============================================================

@st.cache_data(show_spinner=False)
def read_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    """
    Lee CSV desde bytes con varios encodings y separadores probables.

    Nota técnica:
    - Cuando sep=None se usa engine="python" para autodetectar separador.
    - No se usa low_memory con engine="python", porque pandas no lo soporta.
    - Para separadores explícitos se usa primero engine="c" y low_memory=False.
    """
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    separators = [None, ",", ";", "\t", "|"]
    last_error = None

    for enc in encodings:
        for sep in separators:
            try:
                buffer = BytesIO(file_bytes)

                if sep is None:
                    # Autodetección de separador: requiere engine="python".
                    return pd.read_csv(
                        buffer,
                        encoding=enc,
                        sep=None,
                        engine="python"
                    )

                # Separador explícito: engine C permite low_memory=False.
                return pd.read_csv(
                    buffer,
                    encoding=enc,
                    sep=sep,
                    engine="c",
                    low_memory=False
                )

            except Exception as exc:
                last_error = exc

    raise ValueError(f"No fue posible leer el CSV. Último error: {last_error}")


@st.cache_data(show_spinner=False)
def read_excel_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Lee Excel desde bytes."""
    return pd.read_excel(BytesIO(file_bytes), engine="openpyxl")


@st.cache_data(show_spinner=False)
def read_local_file(path_str: str) -> pd.DataFrame:
    """Lee CSV o Excel desde una ruta local."""
    path = Path(path_str).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_csv_from_bytes(path.read_bytes())
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path, engine="openpyxl")

    raise ValueError("Formato no soportado. Use .csv, .xlsx o .xls")


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas para evitar espacios accidentales."""
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.replace("\n", " ", regex=False)
        .str.replace("\r", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )
    return out


# ============================================================
# Datos sintéticos para ejecución sin carga externa
# ============================================================

@st.cache_data(show_spinner=False)
def simulate_tb_data(n: int = 12000, seed: int = 123) -> pd.DataFrame:
    """
    Genera una base sintética de pacientes con TB pulmonar.
    Y sugerida: exito_terapeutico.
    D sugerida: tratamiento_oportuno_7d.
    """
    rng = np.random.default_rng(seed)

    edad = np.clip(rng.normal(44, 18, n).round(), 1, 95).astype(int)
    sexo = rng.choice(["Femenino", "Masculino"], size=n, p=[0.42, 0.58])
    zona_residencia = rng.choice(["Urbana", "Rural"], size=n, p=[0.86, 0.14])
    regimen_afiliacion = rng.choice(
        ["Contributivo", "Subsidiado", "No asegurado", "Especial"],
        size=n,
        p=[0.34, 0.52, 0.10, 0.04]
    )
    grupo_poblacional = rng.choice(
        ["General", "Migrante", "Privado libertad", "Habitante calle", "Indígena"],
        size=n,
        p=[0.78, 0.09, 0.04, 0.05, 0.04]
    )

    indice_vulnerabilidad = np.clip(rng.beta(2.3, 3.0, n), 0, 1)
    vih = rng.binomial(1, np.clip(0.06 + 0.14 * indice_vulnerabilidad, 0, 0.45))
    diabetes = rng.binomial(1, np.clip(0.05 + 0.003 * edad, 0, 0.45))
    desnutricion = rng.binomial(1, np.clip(0.04 + 0.18 * indice_vulnerabilidad, 0, 0.40))
    tabaquismo = rng.binomial(1, np.clip(0.12 + 0.18 * (sexo == "Masculino") + 0.06 * indice_vulnerabilidad, 0, 0.50))
    alcohol_riesgo = rng.binomial(1, np.clip(0.05 + 0.15 * (sexo == "Masculino") + 0.10 * indice_vulnerabilidad, 0, 0.45))
    epoc = rng.binomial(1, np.clip(0.02 + 0.0025 * edad + 0.08 * tabaquismo, 0, 0.35))
    enfermedad_renal = rng.binomial(1, np.clip(0.01 + 0.0015 * edad + 0.04 * diabetes, 0, 0.25))

    condicion_ingreso = rng.choice(
        ["Nuevo", "Reingreso", "Recaída", "Fracaso"],
        size=n,
        p=[0.78, 0.09, 0.10, 0.03]
    )
    resistencia_farmacologica = rng.binomial(
        1,
        np.clip(0.03 + 0.05 * (condicion_ingreso != "Nuevo") + 0.03 * indice_vulnerabilidad, 0, 0.25)
    )
    bk_inicial = rng.choice(["Negativa", "1+", "2+", "3+"], size=n, p=[0.12, 0.34, 0.32, 0.22])
    cultivo_inicial = rng.choice(["Negativo", "Positivo", "No realizado"], size=n, p=[0.10, 0.72, 0.18])
    severidad_inicial = np.clip(
        rng.normal(0, 1, n)
        + 0.8 * vih + 0.45 * desnutricion + 0.35 * diabetes + 0.25 * epoc
        + 0.35 * resistencia_farmacologica,
        -2.5,
        4.5
    )

    demora_sintomas_diagnostico_dias = np.clip(
        rng.gamma(shape=2.2, scale=8, size=n)
        + 12 * indice_vulnerabilidad
        + 5 * (zona_residencia == "Rural"),
        0,
        120
    ).round().astype(int)

    # Propensión a tratamiento oportuno: depende de X.
    logit_d = (
        1.0
        - 1.5 * indice_vulnerabilidad
        - 0.018 * demora_sintomas_diagnostico_dias
        - 0.55 * (regimen_afiliacion == "No asegurado")
        - 0.30 * (zona_residencia == "Rural")
        - 0.15 * vih
        + 0.20 * (regimen_afiliacion == "Contributivo")
    )
    p_d = 1 / (1 + np.exp(-logit_d))
    tratamiento_oportuno_7d = rng.binomial(1, np.clip(p_d, 0.03, 0.97))

    # Resultado Y: éxito terapéutico. El tratamiento oportuno tiene efecto positivo.
    logit_y = (
        -0.20
        + 0.55 * tratamiento_oportuno_7d
        - 0.020 * edad
        - 0.85 * vih
        - 0.45 * diabetes
        - 0.65 * desnutricion
        - 0.70 * resistencia_farmacologica
        - 0.35 * epoc
        - 0.25 * enfermedad_renal
        - 0.28 * (condicion_ingreso != "Nuevo")
        - 0.12 * severidad_inicial
        + 0.25 * (regimen_afiliacion == "Contributivo")
    )
    p_y = 1 / (1 + np.exp(-logit_y))
    exito_terapeutico = rng.binomial(1, np.clip(p_y, 0.02, 0.98))
    muerte = rng.binomial(1, np.clip(0.05 + 0.25 * (1 - exito_terapeutico) + 0.10 * vih + 0.06 * edad / 80, 0, 0.75))

    resultado_final = np.where(
        exito_terapeutico == 1,
        "Éxito terapéutico",
        np.where(muerte == 1, "Fallecido", "No exitoso")
    )

    tiempo_hasta_desenlace_dias = np.clip(
        rng.normal(155, 40, n) - 20 * muerte + 12 * exito_terapeutico,
        10,
        365
    ).round().astype(int)

    return pd.DataFrame({
        "edad": edad,
        "sexo": sexo,
        "zona_residencia": zona_residencia,
        "regimen_afiliacion": regimen_afiliacion,
        "grupo_poblacional": grupo_poblacional,
        "indice_vulnerabilidad": indice_vulnerabilidad,
        "vih": vih,
        "diabetes": diabetes,
        "desnutricion": desnutricion,
        "tabaquismo": tabaquismo,
        "alcohol_riesgo": alcohol_riesgo,
        "epoc": epoc,
        "enfermedad_renal": enfermedad_renal,
        "condicion_ingreso": condicion_ingreso,
        "resistencia_farmacologica": resistencia_farmacologica,
        "bk_inicial": bk_inicial,
        "cultivo_inicial": cultivo_inicial,
        "severidad_inicial": severidad_inicial,
        "demora_sintomas_diagnostico_dias": demora_sintomas_diagnostico_dias,
        "tratamiento_oportuno_7d": tratamiento_oportuno_7d,
        "exito_terapeutico": exito_terapeutico,
        "muerte": muerte,
        "resultado_final": resultado_final,
        "tiempo_hasta_desenlace_dias": tiempo_hasta_desenlace_dias,
    })


# ============================================================
# Utilidades de modelamiento
# ============================================================

def make_one_hot_encoder():
    """OneHotEncoder compatible con scikit-learn reciente y anterior."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def binary_to_numeric(series: pd.Series) -> pd.Series:
    """Convierte variables binarias comunes a 0/1 sin convertir todo texto desconocido a cero."""
    s = series.copy()

    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)

    if pd.api.types.is_numeric_dtype(s):
        out = pd.to_numeric(s, errors="coerce")
        unique = set(out.dropna().unique().tolist())
        if unique.issubset({0, 1}):
            return out.astype("Int64").astype(float)
        # Si viene codificada como 1/2, se transforma 2 -> 1, 1 -> 0 solo si hay exactamente dos valores.
        if len(unique) == 2:
            values = sorted(list(unique))
            return out.map({values[0]: 0, values[1]: 1}).astype(float)
        return out.astype(float)

    mapping = {
        "si": 1, "sí": 1, "s": 1, "yes": 1, "y": 1, "true": 1, "verdadero": 1,
        "positivo": 1, "positiva": 1, "1": 1, "exito": 1, "éxito": 1,
        "exitoso": 1, "oportuno": 1, "fallecido": 1,
        "no": 0, "n": 0, "false": 0, "falso": 0,
        "negativo": 0, "negativa": 0, "0": 0, "no oportuno": 0,
        "no exitoso": 0, "vivo": 0
    }
    normalized = (
        s.astype(str)
        .str.strip()
        .str.lower()
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
    )
    out = normalized.map(mapping)

    if out.notna().sum() == 0:
        # Último recurso: factorizar si solo hay dos categorías reales.
        non_missing = s.dropna().astype(str).str.strip()
        if non_missing.nunique() == 2:
            cats = sorted(non_missing.unique().tolist())
            return s.astype(str).str.strip().map({cats[0]: 0, cats[1]: 1}).astype(float)

    return out.astype(float)


def prepare_model_matrix(df: pd.DataFrame, y_col: str, d_col: str, x_cols: list[str]):
    """Prepara matriz X codificada, y y d para tratamiento binario."""
    required_cols = [y_col, d_col] + x_cols
    work = df[required_cols].copy()

    work[y_col] = binary_to_numeric(work[y_col])
    work[d_col] = binary_to_numeric(work[d_col])
    work = work.dropna(subset=[y_col, d_col])

    # Mantener únicamente observaciones con Y y D binarias.
    work = work[work[y_col].isin([0, 1]) & work[d_col].isin([0, 1])].copy()

    y = work[y_col].astype(int).to_numpy()
    d = work[d_col].astype(int).to_numpy()

    if len(work) < 50:
        raise ValueError("Después de limpiar Y y D quedan menos de 50 registros. Revise la codificación de las variables.")
    if len(np.unique(y)) < 2:
        raise ValueError("La variable resultado Y debe tener dos clases: 0 y 1.")
    if len(np.unique(d)) < 2:
        raise ValueError("La variable tratamiento D debe tener dos clases: 0 y 1.")

    X_raw = work[x_cols].copy()
    categorical_cols = X_raw.select_dtypes(include=["object", "category", "string"]).columns.tolist()
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

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    X = preprocessor.fit_transform(X_raw)

    feature_names = []
    feature_names.extend(numeric_cols)
    if categorical_cols:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        feature_names.extend(ohe.get_feature_names_out(categorical_cols).tolist())

    X_df = pd.DataFrame(np.asarray(X), columns=feature_names, index=work.index)
    return work, X_df, y, d, numeric_cols, categorical_cols


def model_diagnostics(X: np.ndarray, d: np.ndarray, y: np.ndarray, n_splits: int = 5) -> pd.DataFrame:
    """Métricas out-of-fold para modelos auxiliares: propensity y outcome."""
    n_splits = max(2, min(n_splits, int(np.bincount(d).min()), int(np.bincount(y).min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    prop_model = RandomForestClassifier(
        n_estimators=180,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )
    out_model = RandomForestClassifier(
        n_estimators=180,
        max_depth=6,
        min_samples_leaf=20,
        random_state=43,
        n_jobs=-1
    )

    e_hat = cross_val_predict(clone(prop_model), X, d, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    y_hat = cross_val_predict(clone(out_model), X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]

    rows = []
    rows.append({
        "Modelo auxiliar": "m(X) = P(D=1|X)",
        "AUC": roc_auc_score(d, e_hat),
        "Brier": brier_score_loss(d, e_hat),
        "Accuracy 0.5": accuracy_score(d, (e_hat >= 0.5).astype(int))
    })
    rows.append({
        "Modelo auxiliar": "g(X) = E[Y|X]",
        "AUC": roc_auc_score(y, y_hat),
        "Brier": brier_score_loss(y, y_hat),
        "Accuracy 0.5": accuracy_score(y, (y_hat >= 0.5).astype(int))
    })
    return pd.DataFrame(rows), e_hat, y_hat


def fit_propensity_model(X: np.ndarray, d: np.ndarray, trim: float):
    model = LogisticRegression(max_iter=3000, solver="lbfgs")
    model.fit(X, d)
    e = model.predict_proba(X)[:, 1]
    return model, np.clip(e, trim, 1 - trim)


def estimate_logistic_gcomp(X: np.ndarray, d: np.ndarray, y: np.ndarray):
    Xd = np.column_stack([d, X])
    model = LogisticRegression(max_iter=3000, solver="lbfgs")
    model.fit(Xd, y)
    pred_observed = model.predict_proba(Xd)[:, 1]
    auc = roc_auc_score(y, pred_observed)
    mu1 = model.predict_proba(np.column_stack([np.ones(X.shape[0]), X]))[:, 1]
    mu0 = model.predict_proba(np.column_stack([np.zeros(X.shape[0]), X]))[:, 1]
    ate = float(np.mean(mu1 - mu0))
    return ate, auc


def estimate_iptw(X: np.ndarray, d: np.ndarray, y: np.ndarray, trim: float):
    _, e = fit_propensity_model(X, d, trim=trim)
    scores = d * y / e - (1 - d) * y / (1 - e)
    ate = float(np.mean(scores))
    se = float(np.std(scores, ddof=1) / np.sqrt(len(y)))
    weights = d / e + (1 - d) / (1 - e)
    return ate, se, e, weights


def fit_outcome_model(X: np.ndarray, d: np.ndarray, y: np.ndarray):
    Xd = np.column_stack([d, X])
    model = GradientBoostingClassifier(random_state=42)
    model.fit(Xd, y)
    return model


def predict_mu(model, X: np.ndarray, d_value: int):
    d_vec = np.full(X.shape[0], d_value)
    Xd = np.column_stack([d_vec, X])
    return model.predict_proba(Xd)[:, 1]


def estimate_aipw(X: np.ndarray, d: np.ndarray, y: np.ndarray, trim: float):
    _, e = fit_propensity_model(X, d, trim=trim)
    outcome_model = fit_outcome_model(X, d, y)
    mu1 = predict_mu(outcome_model, X, 1)
    mu0 = predict_mu(outcome_model, X, 0)

    scores = mu1 - mu0 + d * (y - mu1) / e - (1 - d) * (y - mu0) / (1 - e)
    ate = float(np.mean(scores))
    se = float(np.std(scores, ddof=1) / np.sqrt(len(y)))
    return ate, se


def estimate_double_ml_irm(X_df: pd.DataFrame, d: np.ndarray, y: np.ndarray, n_folds: int, n_rep: int):
    if not DOUBLEML_AVAILABLE:
        raise ImportError("doubleml no está instalado. Instale con: pip install doubleml")

    df_dml = X_df.copy()
    df_dml["tratamiento"] = d
    df_dml["resultado"] = y

    dml_data = dml.DoubleMLData(df_dml, y_col="resultado", d_cols="tratamiento")

    ml_g = RandomForestRegressor(
        n_estimators=250,
        max_depth=7,
        min_samples_leaf=15,
        random_state=42,
        n_jobs=-1
    )
    ml_m = RandomForestClassifier(
        n_estimators=250,
        max_depth=7,
        min_samples_leaf=15,
        random_state=43,
        n_jobs=-1
    )

    model = dml.DoubleMLIRM(
        dml_data,
        ml_g=ml_g,
        ml_m=ml_m,
        n_folds=n_folds,
        n_rep=n_rep,
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
        "summary": model.summary,
    }


def bootstrap_ci_for_simple_methods(X: np.ndarray, d: np.ndarray, y: np.ndarray, trim: float, n_boot: int):
    rng = np.random.default_rng(42)
    n = len(y)
    values = {"Regresión logística ajustada": [], "IPTW": [], "AIPW": []}
    progress = st.progress(0, text="Calculando bootstrap...")

    for b in range(n_boot):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        Xb, db, yb = X[idx], d[idx], y[idx]
        try:
            values["Regresión logística ajustada"].append(estimate_logistic_gcomp(Xb, db, yb)[0])
            values["IPTW"].append(estimate_iptw(Xb, db, yb, trim=trim)[0])
            values["AIPW"].append(estimate_aipw(Xb, db, yb, trim=trim)[0])
        except Exception:
            pass
        progress.progress((b + 1) / n_boot, text=f"Bootstrap {b + 1}/{n_boot}")

    progress.empty()
    out = {}
    for key, vals in values.items():
        if len(vals) >= 10:
            out[key] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
        else:
            out[key] = (np.nan, np.nan)
    return out


# ============================================================
# Sidebar: carga de datos
# ============================================================

st.sidebar.header("1. Cargar datos")

modo_carga = st.sidebar.radio(
    "Seleccione el método de carga",
    [
        "Generar datos sintéticos dentro de la app",
        "Leer archivo local en la carpeta del proyecto",
        "Subir archivo CSV/Excel manualmente"
    ],
    index=0,
    help=(
        "Si el cargador manual presenta AxiosError 403, use la base sintética o la lectura local. "
        "El error suele estar relacionado con el entorno de despliegue, no con el modelo."
    )
)

if modo_carga == "Generar datos sintéticos dentro de la app":
    n_sim = st.sidebar.slider("Número de registros simulados", 1000, 50000, 12000, 1000)
    seed_sim = st.sidebar.number_input("Semilla de simulación", min_value=1, max_value=999999, value=123, step=1)
    df = simulate_tb_data(n=int(n_sim), seed=int(seed_sim))

elif modo_carga == "Leer archivo local en la carpeta del proyecto":
    local_path = st.sidebar.text_input("Ruta local", value="tb_pulmonar_sintetica_double_ml.csv")
    try:
        df = read_local_file(local_path)
    except Exception as exc:
        st.warning(f"No se pudo leer el archivo local: {exc}")
        st.info("Puede usar la opción de datos sintéticos para probar el modelo sin depender del cargador de archivos.")
        st.stop()

else:
    uploaded_file = st.sidebar.file_uploader(
        "Suba un archivo CSV o Excel desde su equipo",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False,
        help="Si aparece AxiosError 403, pruebe con lectura local o revise la configuración de Streamlit."
    )

    if uploaded_file is None:
        st.info("Suba un archivo o cambie a la opción de datos sintéticos para iniciar el análisis.")
        st.stop()

    try:
        file_bytes = uploaded_file.getvalue()
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix == ".csv":
            df = read_csv_from_bytes(file_bytes)
        elif suffix in [".xlsx", ".xls"]:
            df = read_excel_from_bytes(file_bytes)
        else:
            raise ValueError("Formato no soportado.")
    except Exception as exc:
        st.error(f"No fue posible leer el archivo cargado: {exc}")
        st.stop()

# Limpieza general
if df is None or df.empty:
    st.error("La base está vacía.")
    st.stop()

df = clean_column_names(df)
st.sidebar.success(f"Datos cargados: {df.shape[0]:,} filas y {df.shape[1]:,} columnas")


# ============================================================
# Sidebar: configuración del análisis causal
# ============================================================

st.sidebar.header("2. Configurar análisis causal")

columns = df.columns.tolist()
default_y = "exito_terapeutico" if "exito_terapeutico" in columns else columns[0]
default_d = "tratamiento_oportuno_7d" if "tratamiento_oportuno_7d" in columns else columns[min(1, len(columns) - 1)]

y_col = st.sidebar.selectbox("Variable resultado Y binaria", columns, index=columns.index(default_y))
d_col = st.sidebar.selectbox("Variable tratamiento/exposición D binaria", columns, index=columns.index(default_d))

recommended_x = [
    "edad", "sexo", "zona_residencia", "regimen_afiliacion", "grupo_poblacional",
    "indice_vulnerabilidad", "vih", "diabetes", "desnutricion", "tabaquismo",
    "alcohol_riesgo", "epoc", "enfermedad_renal", "condicion_ingreso",
    "resistencia_farmacologica", "bk_inicial", "cultivo_inicial", "severidad_inicial",
    "demora_sintomas_diagnostico_dias"
]
default_x = [c for c in recommended_x if c in columns and c not in [y_col, d_col]]

x_cols = st.sidebar.multiselect(
    "Variables de control X",
    [c for c in columns if c not in [y_col, d_col]],
    default=default_x
)

min_cases = 50
max_rows = int(min(len(df), 100000))
if max_rows < min_cases:
    st.error("La base tiene muy pocos registros para ejecutar el modelo.")
    st.stop()

slider_min = min(min_cases, max_rows)
slider_step = 1000 if max_rows >= 1000 else max(1, max_rows // 10)
slider_value = min(25000, max_rows)

n_rows = st.sidebar.slider(
    "Número máximo de casos para modelar",
    min_value=slider_min,
    max_value=max_rows,
    value=slider_value,
    step=slider_step
)

trim = st.sidebar.slider("Recorte de propensity score", 0.01, 0.10, 0.03, 0.01)
n_folds = st.sidebar.slider("Folds para DoubleML", 2, 5, 3, 1)
n_rep = st.sidebar.slider("Repeticiones para DoubleML", 1, 5, 1, 1)
run_bootstrap = st.sidebar.checkbox("Bootstrap para métodos comparativos", value=False)
n_boot = st.sidebar.slider("Réplicas bootstrap", 20, 150, 50, 10, disabled=not run_bootstrap)
run_button = st.sidebar.button("Ejecutar análisis causal", type="primary")


# ============================================================
# Tabs
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Vista general",
    "Descriptivos",
    "Modelos causales",
    "Inferencias",
    "Notas técnicas"
])


# ============================================================
# Tab 1
# ============================================================

with tab1:
    st.subheader("Vista previa de la base")
    st.dataframe(df.head(30), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", f"{df.shape[0]:,}")
    c2.metric("Columnas", f"{df.shape[1]:,}")

    y_preview = binary_to_numeric(df[y_col]) if y_col in df.columns else pd.Series(dtype=float)
    d_preview = binary_to_numeric(df[d_col]) if d_col in df.columns else pd.Series(dtype=float)
    c3.metric(f"Media Y: {y_col}", f"{y_preview.mean():.3f}" if y_preview.notna().any() else "No binaria")
    c4.metric(f"Media D: {d_col}", f"{d_preview.mean():.3f}" if d_preview.notna().any() else "No binaria")

    st.subheader("Porcentaje de valores faltantes")
    miss = df.isna().mean().sort_values(ascending=False).reset_index()
    miss.columns = ["variable", "porcentaje_faltante"]
    st.dataframe(miss.head(30), use_container_width=True)


# ============================================================
# Tab 2
# ============================================================

with tab2:
    st.subheader("Exploración descriptiva")

    col_a, col_b = st.columns(2)
    with col_a:
        y_tmp = binary_to_numeric(df[y_col])
        fig = px.histogram(pd.DataFrame({y_col: y_tmp.dropna()}), x=y_col, title=f"Distribución de Y: {y_col}")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        d_tmp = binary_to_numeric(df[d_col])
        fig = px.histogram(pd.DataFrame({d_col: d_tmp.dropna()}), x=d_col, title=f"Distribución de D: {d_col}")
        st.plotly_chart(fig, use_container_width=True)

    if y_tmp.notna().any() and d_tmp.notna().any():
        tmp = pd.DataFrame({"D": d_tmp, "Y": y_tmp}).dropna()
        tmp = tmp[tmp["D"].isin([0, 1]) & tmp["Y"].isin([0, 1])]
        if not tmp.empty:
            rate = tmp.groupby("D")["Y"].mean().reset_index()
            rate["D"] = rate["D"].map({0: "D=0", 1: "D=1"})
            fig = px.bar(rate, x="D", y="Y", text=rate["Y"].round(3), title="Resultado promedio por grupo de tratamiento")
            fig.update_layout(yaxis_title="Media del resultado Y")
            st.plotly_chart(fig, use_container_width=True)

    numeric_available = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_available:
        selected_num = st.multiselect(
            "Variables numéricas para visualizar",
            numeric_available,
            default=[c for c in ["edad", "indice_vulnerabilidad", "severidad_inicial", "demora_sintomas_diagnostico_dias"] if c in numeric_available]
        )
        for col in selected_num[:4]:
            fig = px.histogram(df, x=col, nbins=40, title=f"Distribución de {col}")
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Tab 3
# ============================================================

with tab3:
    st.subheader("Comparación de modelos causales")
    st.markdown(
        """
        Los modelos estiman el **ATE = E[Y(1) - Y(0)]**. En este contexto, el ATE se interpreta como el
        cambio promedio en la probabilidad del resultado cuando la exposición/tratamiento `D` pasa de 0 a 1,
        manteniendo el ajuste por las covariables `X`.
        """
    )

    if not x_cols:
        st.warning("Seleccione al menos una variable de control X en el panel lateral.")
    elif run_button:
        df_sample = df.sample(n=int(n_rows), random_state=42).reset_index(drop=True)

        try:
            work, X_df, y, d, numeric_cols, categorical_cols = prepare_model_matrix(df_sample, y_col, d_col, x_cols)
            X = X_df.to_numpy()
        except Exception as exc:
            st.error(f"No fue posible preparar la matriz de modelamiento: {exc}")
            st.stop()

        st.success(f"Matriz lista: {X.shape[0]:,} registros y {X.shape[1]:,} covariables codificadas.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Media de Y", f"{np.mean(y):.3f}")
        c2.metric("Media de D", f"{np.mean(d):.3f}")
        c3.metric("X originales", len(x_cols))
        c4.metric("X codificadas", X.shape[1])

        st.subheader("Métricas de modelos auxiliares")
        try:
            diag_df, e_hat_oof, y_hat_oof = model_diagnostics(X, d, y, n_splits=n_folds)
            st.dataframe(diag_df.style.format({"AUC": "{:.3f}", "Brier": "{:.3f}", "Accuracy 0.5": "{:.3f}"}), use_container_width=True)
        except Exception as exc:
            st.warning(f"No fue posible calcular métricas auxiliares: {exc}")
            e_hat_oof = None
            y_hat_oof = None

        results = []
        dml_result = None
        propensity = None
        weights = None

        with st.spinner("Ejecutando DoubleMLIRM..."):
            try:
                dml_result = estimate_double_ml_irm(X_df, d, y, n_folds=n_folds, n_rep=n_rep)
                results.append({
                    "Modelo": "DoubleMLIRM",
                    "ATE": dml_result["ate"],
                    "Error estándar": dml_result["se"],
                    "IC 95% inferior": dml_result["ci_low"],
                    "IC 95% superior": dml_result["ci_high"],
                    "p-valor": dml_result["pvalue"]
                })
                st.success("DoubleMLIRM ejecutado correctamente.")
            except Exception as exc:
                st.warning(f"DoubleMLIRM no se ejecutó: {exc}")

        with st.spinner("Ejecutando métodos comparativos..."):
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
            except Exception as exc:
                st.error(f"Error en regresión logística ajustada: {exc}")

            try:
                iptw_ate, iptw_se, propensity, weights = estimate_iptw(X, d, y, trim=trim)
                results.append({
                    "Modelo": "IPTW",
                    "ATE": iptw_ate,
                    "Error estándar": iptw_se,
                    "IC 95% inferior": iptw_ate - 1.96 * iptw_se,
                    "IC 95% superior": iptw_ate + 1.96 * iptw_se,
                    "p-valor": np.nan
                })
            except Exception as exc:
                st.error(f"Error en IPTW: {exc}")

            try:
                aipw_ate, aipw_se = estimate_aipw(X, d, y, trim=trim)
                results.append({
                    "Modelo": "AIPW",
                    "ATE": aipw_ate,
                    "Error estándar": aipw_se,
                    "IC 95% inferior": aipw_ate - 1.96 * aipw_se,
                    "IC 95% superior": aipw_ate + 1.96 * aipw_se,
                    "p-valor": np.nan
                })
            except Exception as exc:
                st.error(f"Error en AIPW: {exc}")

        result_df = pd.DataFrame(results)

        if run_bootstrap and not result_df.empty:
            ci_dict = bootstrap_ci_for_simple_methods(X, d, y, trim=trim, n_boot=int(n_boot))
            for model_name, (low, high) in ci_dict.items():
                mask = result_df["Modelo"] == model_name
                result_df.loc[mask, "IC 95% inferior"] = low
                result_df.loc[mask, "IC 95% superior"] = high

        st.subheader("Resultados comparativos")
        if result_df.empty:
            st.error("No se obtuvo ningún resultado de modelamiento.")
        else:
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

            error_plus = result_df["IC 95% superior"] - result_df["ATE"]
            error_minus = result_df["ATE"] - result_df["IC 95% inferior"]
            fig = px.bar(
                result_df,
                x="Modelo",
                y="ATE",
                text=result_df["ATE"].round(4),
                error_y=error_plus,
                error_y_minus=error_minus,
                title="Comparación del efecto promedio estimado ATE"
            )
            fig.update_layout(yaxis_title="ATE estimado")
            st.plotly_chart(fig, use_container_width=True)

            csv_out = result_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Descargar resultados en CSV",
                data=csv_out,
                file_name="resultados_doubleml_tb.csv",
                mime="text/csv"
            )

        if propensity is not None:
            st.subheader("Diagnóstico de superposición: propensity score")
            prop_df = pd.DataFrame({"propensity_score": propensity, "D": d})
            fig = px.histogram(prop_df, x="propensity_score", color="D", nbins=40, barmode="overlay", title="Distribución del propensity score por grupo D")
            st.plotly_chart(fig, use_container_width=True)

        if weights is not None:
            st.subheader("Diagnóstico de pesos IPTW")
            c1, c2, c3 = st.columns(3)
            c1.metric("Peso promedio", f"{np.mean(weights):.2f}")
            c2.metric("Percentil 95", f"{np.percentile(weights, 95):.2f}")
            c3.metric("Peso máximo", f"{np.max(weights):.2f}")
            fig = px.histogram(pd.DataFrame({"peso_iptw": weights}), x="peso_iptw", nbins=60, title="Distribución de pesos IPTW")
            st.plotly_chart(fig, use_container_width=True)
            if np.max(weights) > 20:
                st.warning("Hay pesos IPTW altos. Esto sugiere revisar la superposición entre grupos tratado/no tratado.")
    else:
        st.info("Configure las variables y pulse **Ejecutar análisis causal**.")


# ============================================================
# Tab 4
# ============================================================

with tab4:
    st.subheader("Discusión, conclusiones e inferencias")
    st.markdown(
        f"""
        ### Interpretación general

        Para esta aplicación, el parámetro central es el **ATE**. Si se define:

        - **Y = `{y_col}`**
        - **D = `{d_col}`**
        - **X = variables de control seleccionadas**

        entonces el ATE representa el cambio promedio esperado en la probabilidad de `Y=1` cuando `D` cambia de 0 a 1,
        después de ajustar por las covariables seleccionadas.

        ### Criterios para discutir los resultados

        1. **Signo del ATE:**
           - ATE positivo: `D=1` se asocia con mayor probabilidad del resultado.
           - ATE negativo: `D=1` se asocia con menor probabilidad del resultado.

        2. **Magnitud:**
           - Un ATE de 0.08 equivale aproximadamente a 8 puntos porcentuales de diferencia promedio.

        3. **Intervalo de confianza:**
           - Si el IC 95% incluye 0, la evidencia estadística es más débil.
           - Si no incluye 0, hay mayor soporte estadístico para una diferencia promedio ajustada.

        4. **Comparación entre métodos:**
           - Si DoubleMLIRM, AIPW, IPTW y G-computation apuntan en una dirección similar, la inferencia gana estabilidad.
           - Si difieren mucho, revise codificación de variables, superposición, selección de X y posible inclusión de mediadores.

        ### Conclusión metodológica

        DoubleML es útil cuando se desea estimar un efecto causal con muchas covariables y relaciones potencialmente no lineales.
        En tuberculosis pulmonar, puede emplearse para evaluar exposiciones clínicas, sociales u operativas como inicio oportuno
        del tratamiento, VIH, resistencia farmacológica, vulnerabilidad social o barreras de acceso.

        ### Precaución epidemiológica

        El resultado no debe interpretarse como causal definitivo sin un diseño epidemiológico sólido. Se recomienda construir un
        **DAG causal**, verificar temporalidad, excluir variables posteriores al tratamiento cuando funcionen como mediadoras y
        realizar análisis de sensibilidad.
        """
    )


# ============================================================
# Tab 5
# ============================================================

with tab5:
    st.subheader("Notas técnicas y solución del error AxiosError 403")
    st.markdown(
        """
        ### Cambios aplicados en esta versión

        - Se corrigió la función `make_one_hot_encoder()`, que en la versión previa se llamaba a sí misma de forma recursiva.
        - Se agregó una opción para **generar datos sintéticos dentro de la app**, de manera que el modelo pueda ejecutarse sin depender del cargador de archivos.
        - Se agregó lectura de **CSV y Excel** por carga manual y por ruta local.
        - Se ajustó el selector de tamaño de muestra para que no falle con bases pequeñas.
        - Se incluyeron métricas de modelos auxiliares: AUC, Brier y accuracy.
        - Se incluyeron diagnósticos de propensity score y pesos IPTW.        
        """
    )

    st.code(
        """# Ejecución local sugerida
pip install -r requirements.txt
streamlit run app_doubleml_tb_streamlit_corregida.py
""",
        language="bash"
    )
