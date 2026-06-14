# =============================================================
# ATBSPACE - Treinamento, comparação e predição por modelos
# =============================================================

import os
import warnings
import math
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

from config import (
    ATB_CONFIG,
    DATA_DIR,
    OUTPUT_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    PREFIXO_SIMULADO,
    PREFIXO_REAL,
    ESTRUTURAS,
    ESTRUTURA_PARAMS,
)

PASTA_MODELOS = os.path.join(OUTPUT_DIR, "modelos_salvos")
REGISTRO_CSV = os.path.join(PASTA_MODELOS, "registro.csv")

# Features originais mais features temporais/PK.
# As features novas impedem o modelo de tratar q8h, q12h, 500 mg e 1000 mg
# como situações praticamente indistintas quando o treino original tem pouca
# variação de intervalo ou dose.
FEATURES = [
    "idade",
    "peso_kg",
    "sexo_num",
    "creat_basal",
    "creat_atual",
    "clcr",
    "dose_mg_kg",
    "dose_mg",
    "intervalo_h",
    "tempo_infusao_h",
    "tempo_horas",
    "tempo_no_intervalo_h",
    "fase_intervalo",
    "numero_dose_teorica",
    "durante_infusao",
    "taxa_infusao_mg_h",
    "dose_diaria_mg",
    "decaimento_pop",
]

TARGET = "concentracao_mg_L"

VERSAO_FEATURES = "v6"


def _params_estrutura(atb: str) -> dict:
    return ESTRUTURA_PARAMS.get(atb, ESTRUTURA_PARAMS["_default"])


# =============================================================
# UTILITÁRIOS
# =============================================================

def _f(valor, padrao=0.0):
    try:
        if pd.isna(valor):
            return float(padrao)
        return float(valor)
    except Exception:
        return float(padrao)


def _round_dose(dose):
    dose = max(float(dose), 0.0)
    if dose <= 0:
        return 0.0
    return float(round(dose / 50.0) * 50.0)


def _montar_features_temporais(dados_base: dict, tempo_h: float, pk_base_mg_L: float = 0.0) -> dict:
    idade = _f(dados_base.get("idade"), 55)
    peso = max(_f(dados_base.get("peso_kg"), 70), 1.0)
    sexo_num = _f(dados_base.get("sexo_num"), 1)
    creat_basal = max(_f(dados_base.get("creat_basal"), dados_base.get("creat_atual", 1.0)), 0.1)
    creat_atual = max(_f(dados_base.get("creat_atual"), creat_basal), 0.1)
    vd_L_kg = max(_f(dados_base.get("vd_L_kg"), 0.3), 0.01)
    vd_total = max(_f(dados_base.get("vd_total_L"), vd_L_kg * peso), 0.1)
    dose = max(_f(dados_base.get("dose_mg"), 0.0), 0.0)
    intervalo = max(_f(dados_base.get("intervalo_h"), 8.0), 0.1)
    tempo_inf = max(_f(dados_base.get("tempo_infusao_h"), 1.0), 0.05)
    cl = max(_f(dados_base.get("clearance_atual_L_h"), 1.0), 0.01)
    tempo_h = max(_f(tempo_h, 0.0), 0.0)

    tempo_no_intervalo = tempo_h % intervalo
    fase_intervalo = tempo_no_intervalo / intervalo if intervalo > 0 else 0.0
    numero_dose_teorica = math.floor(tempo_h / intervalo) + 1
    durante_infusao = 1.0 if tempo_no_intervalo <= min(tempo_inf, intervalo) else 0.0
    taxa_infusao = dose / tempo_inf if tempo_inf > 0 else 0.0
    dose_diaria = dose * (24.0 / intervalo) if intervalo > 0 else 0.0
    dose_por_vd = dose / vd_total if vd_total > 0 else 0.0
    ke = cl / vd_total if vd_total > 0 else 0.0
    meia_vida = 0.693 / ke if ke > 0 else 999.0

    # ClCr de Cockcroft-Gault (feature de upstream, derivável de idade/peso/sexo/creat)
    clcr = ((140 - idade) * peso) / (72 * creat_atual)
    if sexo_num == 0:
        clcr *= 0.85
    clcr = max(clcr, 1.0)

    # Feature de decaimento exponencial com ke populacional.
    # Entrega ao estimador linear a forma funcional da eliminação farmacocinética
    # sem expor o ke individual (que seria informação circular). O ke populacional
    # é fixo (CL_pop / Vss_pop) e derivável sem doseamento sérico.
    CL_POP = 7.5; VSS_POP = 15.9 + 14.8
    ke_pop = CL_POP / VSS_POP
    decaimento_pop = math.exp(-ke_pop * tempo_no_intervalo)

    return {
        "idade": idade,
        "peso_kg": peso,
        "sexo_num": sexo_num,
        "creat_basal": creat_basal,
        "creat_atual": creat_atual,
        "clcr": clcr,
        "dose_mg_kg": dose / peso,
        "dose_mg": dose,
        "intervalo_h": intervalo,
        "tempo_infusao_h": tempo_inf,
        "tempo_horas": tempo_h,
        "tempo_no_intervalo_h": tempo_no_intervalo,
        "fase_intervalo": fase_intervalo,
        "numero_dose_teorica": float(numero_dose_teorica),
        "durante_infusao": durante_infusao,
        "taxa_infusao_mg_h": taxa_infusao,
        "dose_diaria_mg": dose_diaria,
        "decaimento_pop": decaimento_pop,
    }


def _regimes_treino_para_linha(atb: str, row: pd.Series, max_regimes: int = 18) -> list:
    cfg = ATB_CONFIG.get(atb, {})
    admin = cfg.get("administracao", {})
    peso = max(_f(row.get("peso_kg"), 70), 1.0)
    dose_row = _round_dose(row.get("dose_mg", 1000))
    intervalo_row = _f(row.get("intervalo_h"), 8)
    tempo_row = _f(row.get("tempo_infusao_h"), admin.get("tempo_infusao_h", 1.0))

    doses = {dose_row}
    intervalos = {float(intervalo_row)}
    tempos_inf = {float(tempo_row), float(admin.get("tempo_infusao_h", tempo_row) or tempo_row)}

    def add_parametros(params):
        if not params:
            return
        d = params.get("dose_mg")
        if d is None and params.get("dose_mg_kg") is not None:
            d = params.get("dose_mg_kg") * peso
        if d:
            doses.add(_round_dose(d))
        if params.get("intervalo_h"):
            intervalos.add(float(params.get("intervalo_h")))

    for _, p in cfg.get("ajuste_renal", {}).items():
        add_parametros(p)
    for _, p in cfg.get("terapia_renal_substitutiva", {}).items():
        add_parametros(p)

    for mult in [0.5, 0.75, 1.0, 1.25, 1.5]:
        doses.add(_round_dose(dose_row * mult))

    for it in admin.get("intervalo_opcoes_h", []):
        intervalos.add(float(it))
    for it in [6, 8, 12, 24, 48]:
        intervalos.add(float(it))

    for ti in [0.5, 1.0, 2.0, 3.0, 4.0]:
        tempos_inf.add(float(ti))

    doses = sorted([d for d in doses if 50 <= d <= 10000])[:8]
    intervalos = sorted([i for i in intervalos if 1 <= i <= 72])[:7]
    tempos_inf = sorted([t for t in tempos_inf if 0.05 <= t <= 24])[:5]

    regimes = []
    vistos = set()

    # Primeiro entram os pares clinicamente configurados, para garantir que
    # 500 q12, 1000 q8, 1000 q12 etc. apareçam no treino quando existirem.
    for grupo in ["ajuste_renal", "terapia_renal_substitutiva"]:
        for _, p in cfg.get(grupo, {}).items():
            d = p.get("dose_mg")
            if d is None and p.get("dose_mg_kg") is not None:
                d = p.get("dose_mg_kg") * peso
            it = p.get("intervalo_h")
            if d and it:
                chave = (_round_dose(d), float(it), float(tempo_row))
                if chave not in vistos:
                    regimes.append({"dose_mg": chave[0], "intervalo_h": chave[1], "tempo_infusao_h": min(chave[2], chave[1])})
                    vistos.add(chave)

    # Depois entra o regime observado no paciente simulado.
    chave_row = (dose_row, float(intervalo_row), min(float(tempo_row), float(intervalo_row)))
    if chave_row not in vistos:
        regimes.append({"dose_mg": chave_row[0], "intervalo_h": chave_row[1], "tempo_infusao_h": chave_row[2]})
        vistos.add(chave_row)

    # Completa com grade de dose x intervalo. Mantém o treino leve.
    for d in doses:
        for it in intervalos:
            for ti in tempos_inf:
                ti_ok = min(float(ti), float(it))
                chave = (float(d), float(it), float(ti_ok))
                if chave in vistos:
                    continue
                regimes.append({"dose_mg": float(d), "intervalo_h": float(it), "tempo_infusao_h": float(ti_ok)})
                vistos.add(chave)
                if len(regimes) >= max_regimes:
                    return regimes

    return regimes[:max_regimes]


# =============================================================
# PREPARAÇÃO DOS DADOS
# =============================================================

def carregar_dataset(atb: str, usar_real: bool = False) -> pd.DataFrame:
    pasta = os.path.join(DATA_DIR, atb)
    nome = f"{PREFIXO_REAL}{atb}.csv" if usar_real else f"{PREFIXO_SIMULADO}{atb}.csv"
    caminho = os.path.join(pasta, nome)
    if not os.path.exists(caminho):
        tipo = "real" if usar_real else "simulado"
        raise FileNotFoundError(f"Dataset {tipo} não encontrado: {caminho}")
    return pd.read_csv(caminho)


def preparar_formato_longo(df: pd.DataFrame, atb: str = None, usar_curva_densa: bool = True,
                           estrutura: str = "mono") -> pd.DataFrame:
    if usar_curva_densa:
        return preparar_formato_curva_densa(df, atb=atb, estrutura=estrutura)

    from config import TEMPOS_HORAS

    tempos_map = {
        0: ("basal", "basal"),
        24: ("24h", "24h"),
        48: ("48h", "48h"),
        72: ("72h", "72h"),
        168: ("7dias", "7dias"),
    }
    registros = []
    for _, row in df.iterrows():
        for tempo_h, (sufixo_conc, sufixo_cl) in tempos_map.items():
            conc_col = f"conc_{sufixo_conc}_mg_L"
            cl_col = f"cl_{sufixo_cl}_L_h"
            creat_col = f"creat_{sufixo_conc}" if sufixo_conc != "basal" else "creat_basal"
            if conc_col not in df.columns:
                continue
            dados = {
                "idade": row.get("idade"),
                "peso_kg": row.get("peso_kg"),
                "sexo_num": 1 if row.get("sexo") == "masculino" else 0,
                "creat_basal": row.get("creat_basal"),
                "creat_atual": row.get(creat_col, row.get("creat_basal")),
                "vd_L_kg": row.get("vd_L_kg"),
                "vd_total_L": row.get("vd_total_L"),
                "dose_mg": row.get("dose_mg"),
                "intervalo_h": row.get("intervalo_h"),
                "tempo_infusao_h": row.get("tempo_infusao_h"),
                "clearance_atual_L_h": row.get(cl_col, row.get("cl_basal_L_h")),
            }
            reg = _montar_features_temporais(dados, tempo_h, row[conc_col])
            reg["paciente_id"] = row.get("paciente_id")
            reg[TARGET] = row[conc_col]
            registros.append(reg)
    return pd.DataFrame(registros)


def _parametros_verdadeiros(atb: str, row: pd.Series, rng) -> dict:
    """
    Deriva os parâmetros PK VERDADEIROS do paciente a partir das covariáveis,
    com efeito não-linear de função renal, ARC, idade e sexo, mais variabilidade
    interindividual (IIV) lognormal. Esses parâmetros geram o alvo, mas NUNCA
    entram como feature (senão o problema vira circular).
    """
    P = _params_estrutura(atb)
    idade = _f(row.get("idade"), 60)
    peso = max(_f(row.get("peso_kg"), 70), 1.0)
    sexo = 1 if row.get("sexo") == "masculino" else 0
    creat = max(_f(row.get("creat_basal"), 1.0), 0.1)

    clcr = ((140 - idade) * peso) / (72 * creat)
    if sexo == 0:
        clcr *= 0.85
    clcr = max(clcr, 5.0)

    fator_idade = 1.0 - 0.12 * float(np.clip((idade - 60) / 40, 0, 1))
    cl_typ = P["CL_POP"] * (clcr / P["CLCR_REF"]) ** 0.8 * fator_idade
    if clcr > 130:
        cl_typ *= 1.0 + 0.005 * (clcr - 130)          # clearance renal aumentado
    CL = cl_typ * math.exp(rng.normal(0, P["CV_CL"]))

    fator_sexo = 1.15 if sexo == 0 else 1.0
    V1 = P["V1_POP"] * (peso / 70.0) * fator_sexo * math.exp(rng.normal(0, P["CV_V1"]))
    V2 = P["V2_POP"] * (peso / 70.0) * math.exp(rng.normal(0, P["CV_V2"]))
    Q2 = P["Q2_POP"] * (peso / 70.0) ** 0.75 * math.exp(rng.normal(0, P["CV_Q"]))
    V3 = P["V3_POP"] * (peso / 70.0) * math.exp(rng.normal(0, P["CV_V2"]))
    Q3 = P["Q3_POP"] * (peso / 70.0) ** 0.75 * math.exp(rng.normal(0, P["CV_Q"]))
    Vss = V1 + V2

    return {
        "CL": max(CL, 0.3), "V1": max(V1, 1.0), "V2": max(V2, 1.0),
        "Q2": max(Q2, 0.3), "V3": max(V3, 1.0), "Q3": max(Q3, 0.3),
        "Vss": max(Vss, 1.0), "clcr": clcr,
    }


def _curva_verdade(estrutura, par, dose, intervalo, tempo_inf, numero_doses, resol):
    """Gera a curva-verdade para a estrutura escolhida."""
    from pk_engine import (gerar_curva_concentracao,
                           gerar_curva_bicompartimental,
                           gerar_curva_tricompartimental)
    if estrutura == "mono":
        return gerar_curva_concentracao(
            dose_mg=dose, vd_L=par["Vss"], clearance_L_h=par["CL"],
            intervalo_h=intervalo, tempo_infusao_h=tempo_inf,
            numero_doses=numero_doses, resolucao_h=resol)
    if estrutura == "bi":
        return gerar_curva_bicompartimental(
            dose_mg=dose, V1_L=par["V1"], CL_L_h=par["CL"], Q2_L_h=par["Q2"], V2_L=par["V2"],
            intervalo_h=intervalo, tempo_infusao_h=tempo_inf,
            numero_doses=numero_doses, resolucao_h=resol)
    if estrutura == "tri":
        return gerar_curva_tricompartimental(
            dose_mg=dose, V1_L=par["V1"], CL_L_h=par["CL"], Q2_L_h=par["Q2"], V2_L=par["V2"],
            Q3_L_h=par["Q3"], V3_L=par["V3"], intervalo_h=intervalo, tempo_infusao_h=tempo_inf,
            numero_doses=numero_doses, resolucao_h=resol)
    raise ValueError(f"Estrutura desconhecida: {estrutura}")


def preparar_formato_curva_densa(
    df: pd.DataFrame,
    atb: str = None,
    estrutura: str = "mono",
    duracao_h: float = 72.0,
    resolucao_h: float = 1.0,
) -> pd.DataFrame:
    registros = []
    atb = atb or "meropenem"

    for _, row in df.iterrows():
        pid = row.get("paciente_id", 0)
        rng = np.random.default_rng(RANDOM_STATE + int(pid) if pid is not None else RANDOM_STATE)
        par = _parametros_verdadeiros(atb, row, rng)

        base_paciente = {
            "idade": row.get("idade"),
            "peso_kg": row.get("peso_kg"),
            "sexo_num": 1 if row.get("sexo") == "masculino" else 0,
            "creat_basal": row.get("creat_basal"),
            "creat_atual": row.get("creat_basal"),
        }

        for regime in _regimes_treino_para_linha(atb, row):
            dados_regime = dict(base_paciente)
            dados_regime.update(regime)
            intervalo = max(_f(regime["intervalo_h"], 8), 0.1)
            numero_doses = max(1, int(math.ceil(duracao_h / intervalo)))

            curva = _curva_verdade(
                estrutura, par,
                float(regime["dose_mg"]), intervalo, float(regime["tempo_infusao_h"]),
                numero_doses, resolucao_h)

            for tempo_h, conc in zip(curva["tempos_h"], curva["concentracoes"]):
                if tempo_h > duracao_h:
                    continue
                reg = _montar_features_temporais(dados_regime, tempo_h, conc)
                reg["paciente_id"] = pid
                reg[TARGET] = conc
                registros.append(reg)

    return pd.DataFrame(registros)


def dividir_dados(df: pd.DataFrame):
    """
    Divide treino e teste evitando vazamento por paciente.

    Antes o split era linha a linha; como a curva densa gera vários pontos
    do mesmo paciente, o mesmo paciente podia cair em treino e teste.
    Isso inflava R² e deixava a comparação dos modelos otimista demais.
    """
    X = df[FEATURES]
    y = df[TARGET]

    if "paciente_id" in df.columns and df["paciente_id"].nunique() > 1:
        grupos = df["paciente_id"].astype(str)
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        idx_train, idx_test = next(splitter.split(X, y, groups=grupos))
        return X.iloc[idx_train], X.iloc[idx_test], y.iloc[idx_train], y.iloc[idx_test]

    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


# =============================================================
# MODELOS
# =============================================================

def obter_modelos() -> dict:
    return {
        "bayesianridge": {
            "modelo": BayesianRidge(),
            "scaler": StandardScaler(),
            "requer_scaler": True,
        },
        "lightgbm": {
            "modelo": LGBMRegressor(
                n_estimators=90,
                learning_rate=0.06,
                max_depth=4,
                num_leaves=15,
                min_data_in_leaf=8,
                random_state=RANDOM_STATE,
                n_jobs=1,
                force_col_wise=True,
                verbose=-1,
            ),
            "scaler": None,
            "requer_scaler": False,
        },
        "gradientboosting": {
            "modelo": GradientBoostingRegressor(
                n_estimators=120,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE,
            ),
            "scaler": None,
            "requer_scaler": False,
        },
        "randomforest": {
            "modelo": RandomForestRegressor(
                n_estimators=80,
                max_depth=14,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "scaler": None,
            "requer_scaler": False,
        },
    }


def treinar_modelo(atb: str, nome_modelo: str, X_train, y_train, X_test, y_test,
                   estrutura: str = "mono") -> dict:
    modelos = obter_modelos()
    if nome_modelo not in modelos:
        raise ValueError(f"Modelo '{nome_modelo}' não disponível.")

    cfg = modelos[nome_modelo]
    modelo = cfg["modelo"]
    scaler = cfg["scaler"]

    if cfg["requer_scaler"]:
        X_train_fit = scaler.fit_transform(X_train)
        X_test_fit = scaler.transform(X_test)
    else:
        X_train_fit = X_train
        X_test_fit = X_test

    modelo.fit(X_train_fit, y_train)
    y_pred = modelo.predict(X_test_fit)

    # BayesianRidge: treina no espaço log(1+c) e avalia no espaço original.
    # A concentração PK é exponencial; um estimador linear ajusta muito melhor
    # uma relação log-linear do que uma relação exponencial direta.
    log_transform = (nome_modelo == "bayesianridge")
    if log_transform:
        y_train_fit = np.log1p(np.maximum(np.asarray(y_train, dtype=float), 0.0))
        y_test_eval = np.maximum(np.asarray(y_test, dtype=float), 0.0)
        modelo.fit(X_train_fit, y_train_fit)
        y_pred_log = modelo.predict(X_test_fit)
        y_pred = np.expm1(np.maximum(y_pred_log, 0.0))
    else:
        y_pred = modelo.predict(X_test_fit)

    y_test_eval = np.asarray(y_test, dtype=float)
    r2   = round(r2_score(y_test_eval, y_pred), 4)
    mae  = round(mean_absolute_error(y_test_eval, y_pred), 4)
    rmse = round(np.sqrt(mean_squared_error(y_test_eval, y_pred)), 4)
    ic_disponivel = nome_modelo == "bayesianridge"

    os.makedirs(PASTA_MODELOS, exist_ok=True)
    data_treino = datetime.today().strftime("%Y%m%d")
    nome_arquivo = f"{atb}_{estrutura}_{nome_modelo}_{data_treino}.pkl"
    caminho_pkl = os.path.join(PASTA_MODELOS, nome_arquivo)

    joblib.dump({"modelo": modelo, "scaler": scaler, "features": FEATURES,
                 "versao_features": VERSAO_FEATURES, "estrutura": estrutura,
                 "log_transform": log_transform}, caminho_pkl)

    # Salva y_test/y_pred em arquivo separado para sobreviver ao restart do Streamlit
    nome_diag = nome_arquivo.replace(".pkl", "_diagnostico.pkl")
    caminho_diag = os.path.join(PASTA_MODELOS, nome_diag)
    joblib.dump({
        "y_test": [round(float(v), 3) for v in y_test_eval],
        "y_pred": [round(float(v), 3) for v in y_pred],
        "r2": r2, "mae_mg_L": mae, "rmse_mg_L": rmse,
    }, caminho_diag)

    _atualizar_registro(atb, nome_modelo, nome_arquivo, r2, mae, rmse, data_treino, estrutura)

    return {
        "atb": atb,
        "estrutura": estrutura,
        "modelo": nome_modelo,
        "r2": r2,
        "mae_mg_L": mae,
        "rmse_mg_L": rmse,
        "ic_disponivel": ic_disponivel,
        "arquivo": nome_arquivo,
        "y_test": list(y_test_eval),
        "y_pred": list(y_pred),
    }


def _conc_mono_pop(clcr, peso, dose, intervalo, tempo_infusao, tempo, atb):
    """Concentração de 1 compartimento com parâmetros POPULACIONAIS (a convenção),
    usada como baseline: é contra ela que se mede se o ML é eficiente."""
    P = _params_estrutura(atb)
    CL = max(P["CL_POP"] * (clcr / P["CLCR_REF"]), 0.3)
    V = max((P["V1_POP"] + P["V2_POP"]) / 70.0 * peso, 1.0)
    ke = CL / V
    tinf = max(tempo_infusao, 0.05)
    rin = dose / tinf
    n_doses = int(tempo // intervalo) + 1
    conc = 0.0
    for n in range(n_doses):
        tr = tempo - n * intervalo
        if tr < 0:
            continue
        if tr <= tinf:
            conc += (rin / CL) * (1 - math.exp(-ke * tr))
        else:
            c_fim = (rin / CL) * (1 - math.exp(-ke * tinf))
            conc += c_fim * math.exp(-ke * (tr - tinf))
    return max(conc, 0.0)


def comparar_modelos(atb: str, usar_real: bool = False, retornar_detalhes: bool = False,
                     estrutura: str = "mono"):
    df_wide = carregar_dataset(atb, usar_real=usar_real)
    df_longo = preparar_formato_longo(df_wide, atb=atb, usar_curva_densa=True, estrutura=estrutura)

    max_linhas_treino = 25000
    if len(df_longo) > max_linhas_treino:
        df_longo = df_longo.sample(max_linhas_treino, random_state=RANDOM_STATE).reset_index(drop=True)

    X_train, X_test, y_train, y_test = dividir_dados(df_longo)
    resultados = []
    detalhes = {}

    # baseline: convenção monocompartimental populacional vs a verdade
    base_pred = X_test.apply(
        lambda r: _conc_mono_pop(r["clcr"], r["peso_kg"], r["dose_mg"],
                                 r["intervalo_h"], r["tempo_infusao_h"], r["tempo_horas"], atb),
        axis=1).values
    yt = np.asarray(y_test, dtype=float)
    resultados.append({
        "modelo": "PK_simples_baseline",
        "r2": round(r2_score(yt, base_pred), 4),
        "mae_mg_L": round(mean_absolute_error(yt, base_pred), 4),
        "rmse_mg_L": round(np.sqrt(mean_squared_error(yt, base_pred)), 4),
        "ic_disponivel": False,
    })
    detalhes["PK_simples_baseline"] = {
        "y_test": [round(float(v), 3) for v in yt],
        "y_pred": [round(float(v), 3) for v in base_pred],
    }

    for nome_modelo in obter_modelos().keys():
        resultado = treinar_modelo(atb, nome_modelo, X_train, y_train, X_test, y_test, estrutura=estrutura)
        resultados.append({
            "modelo": nome_modelo,
            "r2": resultado["r2"],
            "mae_mg_L": resultado["mae_mg_L"],
            "rmse_mg_L": resultado["rmse_mg_L"],
            "ic_disponivel": resultado["ic_disponivel"],
        })
        detalhes[nome_modelo] = {
            "y_test": [round(float(v), 3) for v in resultado["y_test"]],
            "y_pred": [round(float(v), 3) for v in resultado["y_pred"]],
        }

    tabela = pd.DataFrame(resultados).sort_values("mae_mg_L").reset_index(drop=True)

    if retornar_detalhes:
        return tabela, detalhes
    return tabela


# =============================================================
# REGISTRO DOS MODELOS
# =============================================================

def _carregar_registro() -> pd.DataFrame:
    if os.path.exists(REGISTRO_CSV):
        df = pd.read_csv(REGISTRO_CSV)
    else:
        df = pd.DataFrame(columns=["atb", "estrutura", "modelo", "arquivo",
                                   "data_treino", "r2", "mae_mg_L", "rmse_mg_L", "ativo"])
    if "estrutura" not in df.columns:
        df["estrutura"] = "mono"   # registros antigos = monocompartimental
    df["estrutura"] = df["estrutura"].fillna("mono")
    return df


def _atualizar_registro(atb, nome_modelo, nome_arquivo, r2, mae, rmse, data_treino, estrutura="mono"):
    df = _carregar_registro()

    mask = (df["atb"] == atb) & (df["modelo"] == nome_modelo) & (df["estrutura"] == estrutura)
    df.loc[mask, "ativo"] = "nao"

    nova = pd.DataFrame([{
        "atb": atb,
        "estrutura": estrutura,
        "modelo": nome_modelo,
        "arquivo": nome_arquivo,
        "data_treino": data_treino,
        "r2": r2,
        "mae_mg_L": mae,
        "rmse_mg_L": rmse,
        "ativo": "sim",
    }])
    df = pd.concat([df, nova], ignore_index=True)
    df["data_treino"] = df["data_treino"].astype(str)
    df = df.sort_values("data_treino", ascending=False)
    df = df.drop_duplicates(subset=["atb", "estrutura", "modelo"], keep="first")
    df = df.sort_values(["atb", "estrutura", "modelo"]).reset_index(drop=True)
    df.to_csv(REGISTRO_CSV, index=False)


def _buscar_modelo_ativo(atb: str, nome_modelo: str, estrutura: str = "mono") -> str:
    if not os.path.exists(REGISTRO_CSV):
        raise FileNotFoundError("Registro de modelos não encontrado. Treine os modelos primeiro.")
    df = _carregar_registro()
    mask = ((df["atb"] == atb) & (df["modelo"] == nome_modelo)
            & (df["estrutura"] == estrutura) & (df["ativo"] == "sim"))
    rows = df[mask]
    if rows.empty:
        raise FileNotFoundError(
            f"Nenhum modelo ativo para {atb} - {estrutura} - {nome_modelo}. "
            f"Treine a estrutura '{estrutura}' em 'Comparação de Modelos'.")
    arquivo = rows.iloc[-1]["arquivo"]
    return os.path.join(PASTA_MODELOS, arquivo)


def ativar_modelos(atb: str, nomes: list, estrutura: str = "mono"):
    """Ativa os modelos da lista (na estrutura dada) e desativa os demais dessa estrutura."""
    if not os.path.exists(REGISTRO_CSV):
        return
    df = _carregar_registro()
    mask_ctx = (df["atb"] == atb) & (df["estrutura"] == estrutura)
    df.loc[mask_ctx, "ativo"] = "nao"
    for nome in nomes:
        mask = mask_ctx & (df["modelo"] == nome)
        if mask.any():
            df.loc[mask, "ativo"] = "sim"
    df.to_csv(REGISTRO_CSV, index=False)


def listar_modelos_ativos(atb: str, estrutura: str = "mono") -> list:
    if not os.path.exists(REGISTRO_CSV):
        return []
    df = _carregar_registro()
    mask = (df["atb"] == atb) & (df["estrutura"] == estrutura) & (df["ativo"] == "sim")
    return list(df[mask]["modelo"].unique())


# =============================================================
# PREDIÇÃO
# =============================================================

def carregar_detalhes_diagnostico(atb: str, estrutura: str = "mono") -> dict:
    """
    Carrega y_test/y_pred salvos em disco para todos os modelos ativos
    na estrutura dada. Sobrevive a reinicializações do Streamlit.
    Retorna dict {nome_modelo: {"y_test":[], "y_pred":[]}} ou {} se não encontrado.
    """
    if not os.path.exists(REGISTRO_CSV):
        return {}
    df = _carregar_registro()
    mask = (df["atb"] == atb) & (df["estrutura"] == estrutura) & (df["ativo"] == "sim")
    detalhes = {}
    for _, row in df[mask].iterrows():
        nome_arq = row["arquivo"].replace(".pkl", "_diagnostico.pkl")
        caminho  = os.path.join(PASTA_MODELOS, nome_arq)
        if os.path.exists(caminho):
            try:
                d = joblib.load(caminho)
                detalhes[row["modelo"]] = {
                    "y_test": d.get("y_test", []),
                    "y_pred": d.get("y_pred", []),
                }
            except Exception:
                pass
    return detalhes


def _carregar_modelo(atb: str, nome_modelo: str, estrutura: str = "mono"):
    caminho_pkl = _buscar_modelo_ativo(atb, nome_modelo, estrutura)
    obj = joblib.load(caminho_pkl)
    if "features" not in obj or obj.get("versao_features") != VERSAO_FEATURES:
        raise FileNotFoundError(
            f"Modelo ativo para {atb}-{estrutura}-{nome_modelo} foi treinado em formato antigo. "
            "Apague os .pkl antigos e treine novamente em 'Comparação de Modelos'."
        )
    return obj


def _suavizar_ratio(ratio: np.ndarray, janela: int = 5) -> np.ndarray:
    if len(ratio) < 3:
        return ratio
    janela = max(3, min(janela, len(ratio)))
    if janela % 2 == 0:
        janela += 1
    pad = janela // 2
    padded = np.pad(ratio, (pad, pad), mode="edge")
    kernel = np.ones(janela) / janela
    return np.convolve(padded, kernel, mode="valid")


def predizer_serie_temporal(atb: str, nome_modelo: str, dados_base: dict, tempos_h: list = None,
                            estrutura: str = "mono") -> dict:
    from pk_engine import gerar_curva_concentracao
    from config import TEMPOS_HORAS

    if tempos_h is None:
        tempos_h = TEMPOS_HORAS

    tempos = np.array([float(t) for t in tempos_h], dtype=float)
    max_t = float(np.max(tempos)) if len(tempos) else 0.0
    intervalo = max(_f(dados_base.get("intervalo_h"), 8.0), 0.1)
    resolucao = min(np.diff(np.unique(tempos)).min() if len(np.unique(tempos)) > 1 else 0.5, 1.0)
    resolucao = max(float(resolucao), 0.1)
    numero_doses = max(1, int(math.ceil(max_t / intervalo)))

    curva_pk = gerar_curva_concentracao(
        dose_mg=max(_f(dados_base.get("dose_mg"), 0), 0),
        vd_L=max(_f(dados_base.get("vd_total_L"), 1), 0.1),
        clearance_L_h=max(_f(dados_base.get("clearance_atual_L_h"), 1), 0.01),
        intervalo_h=intervalo,
        tempo_infusao_h=max(_f(dados_base.get("tempo_infusao_h"), 1), 0.05),
        numero_doses=numero_doses,
        resolucao_h=resolucao,
    )
    tempos_pk = np.array(curva_pk["tempos_h"], dtype=float)
    conc_pk = np.array(curva_pk["concentracoes"], dtype=float)
    pk_interp = np.interp(tempos, tempos_pk, conc_pk)

    linhas = [_montar_features_temporais(dados_base, t, pk) for t, pk in zip(tempos, pk_interp)]
    obj = _carregar_modelo(atb, nome_modelo, estrutura)
    modelo = obj["modelo"]
    scaler = obj["scaler"]
    features = obj["features"]
    X = pd.DataFrame(linhas)[features]

    X_fit = scaler.transform(X) if scaler is not None else X

    log_transform = obj.get("log_transform", False)
    ic_inf = None
    ic_sup = None

    if nome_modelo == "bayesianridge":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pred_raw, std = modelo.predict(X_fit, return_std=True)
        std_safe = np.where(np.isfinite(std), std, 0.0)
        if log_transform:
            # Reverte do espaço log: exp(pred ± 1.96*std) - 1
            pred_raw = np.expm1(np.maximum(pred_raw, 0.0))
            ic_inf = [round(max(float(np.expm1(max(p - 1.96*s, 0.0))), 0.0), 2)
                      for p, s in zip(np.log1p(np.maximum(pred_raw,0)), std_safe)]
            ic_sup = [round(max(float(np.expm1(p + 1.96*s)), 0.0), 2)
                      for p, s in zip(np.log1p(np.maximum(pred_raw,0)), std_safe)]
        else:
            ic_inf = [round(max(float(p - 1.96*s), 0.0), 2) for p, s in zip(pred_raw, std_safe)]
            ic_sup = [round(max(float(p + 1.96*s), 0.0), 2) for p, s in zip(pred_raw, std_safe)]
    else:
        pred_raw = modelo.predict(X_fit)
        if log_transform:
            pred_raw = np.expm1(np.maximum(pred_raw, 0.0))

    pred_raw = np.maximum(np.asarray(pred_raw, dtype=float), 0.0)

    # O modelo prevê a concentração diretamente das features de upstream, na
    # estrutura em que foi treinado. NÃO há mais ancoragem na curva PK: a curva
    # PK fica só como referência/convenção para comparação. Sem ancoragem, cada
    # modelo (e cada estrutura) responde de forma própria.
    pred_final = pred_raw

    return {
        "tempos_h": [float(t) for t in tempos],
        "concentracoes": [round(float(c), 2) for c in pred_final],
        "concentracoes_ml_brutas": [round(float(c), 2) for c in pred_raw],
        "pk_base": [round(float(c), 2) for c in pk_interp],
        "ic_inferior": ic_inf,
        "ic_superior": ic_sup,
    }


def predizer_curva_modelo(atb: str, nome_modelo: str, dados_base: dict, duracao_total_h: float,
                          resolucao_h: float = 0.5, estrutura: str = "mono") -> dict:
    """
    Prediz uma curva densa nos mesmos tempos que serão desenhados.

    A versão anterior predizia só 0, 24, 48, 72 e 168 h e depois ligava
    esses pontos por interpolação linear. Para regimes q6h, q8h ou q12h isso
    escondia picos e vales e deixava o gráfico incompatível com T>MIC.
    """
    if nome_modelo == "pkpd":
        raise ValueError("predizer_curva_modelo deve ser usado apenas para modelos treinados de ML.")

    duracao_total_h = max(float(duracao_total_h), 0.0)
    resolucao_h = max(float(resolucao_h), 0.1)
    tempos_densos = np.arange(0, duracao_total_h + resolucao_h, resolucao_h)

    serie = predizer_serie_temporal(atb, nome_modelo, dados_base, tempos_h=list(tempos_densos), estrutura=estrutura)
    concs = np.array(serie["concentracoes"], dtype=float)

    return {
        "tempos_h"              : [float(t) for t in tempos_densos],
        "concentracoes"         : [round(float(c), 2) for c in concs],
        "concentracoes_ml_brutas": serie.get("concentracoes_ml_brutas"),
        "pk_base"               : serie.get("pk_base"),
        "pontos_treino_h"       : [float(t) for t in tempos_densos],
        "pontos_treino_conc"    : [round(float(c), 2) for c in concs],
        "pico_mg_L"             : round(float(np.max(concs)), 2) if len(concs) else 0.0,
        "vale_mg_L"             : round(float(concs[-1]), 2) if len(concs) else 0.0,
        "ic_inferior"           : serie.get("ic_inferior"),
        "ic_superior"           : serie.get("ic_superior"),
        "modelo"                : nome_modelo,
        "estrutura"             : estrutura,
    }


def predizer(atb: str, nome_modelo: str, dados_paciente: dict, retornar_ic: bool = False,
             estrutura: str = "mono") -> dict:
    tempo = _f(dados_paciente.get("tempo_horas"), 0)
    serie = predizer_serie_temporal(atb, nome_modelo, dados_paciente, tempos_h=[tempo], estrutura=estrutura)
    conc = serie["concentracoes"][0]
    return {
        "concentracao_mg_L": conc,
        "ic_95_inferior": serie.get("ic_inferior", [None])[0] if serie.get("ic_inferior") else None,
        "ic_95_superior": serie.get("ic_superior", [None])[0] if serie.get("ic_superior") else None,
    }


if __name__ == "__main__":
    print("Use este arquivo pelo app Streamlit ou importe comparar_modelos().")