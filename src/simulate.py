# =============================================================
# ATBSPACE - Simulação de Pacientes Virtuais
# =============================================================

import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime

from config import (
    ATB_CONFIG,
    DATA_DIR,
    RANDOM_STATE,
    N_PACIENTES,
    TEMPOS_HORAS,
    PREFIXO_COLETA,
    PREFIXO_SIMULADO,
)
from pk_engine import (
    calcular_clcr_cockcroft_gault,
    calcular_concentracao,
    calcular_auc,
    calcular_t_mic,
    gerar_curva_concentracao,
)

np.random.seed(RANDOM_STATE)


# =============================================================
# DESCOBERTA AUTOMÁTICA DE ARQUIVOS DE COLETA
# =============================================================

def descobrir_arquivos_coleta() -> list:
    """
    Busca automaticamente todos os arquivos coletaliteratura_*.xlsx
    dentro de data/[atb]/.
    """
    padrao = os.path.join(DATA_DIR, "**", f"{PREFIXO_COLETA}*.xlsx")
    arquivos = glob.glob(padrao, recursive=True)
    return arquivos


def extrair_atb_do_nome(caminho: str) -> str:
    nome = os.path.basename(caminho)
    atb = nome.replace(PREFIXO_COLETA, "").replace(".xlsx", "")
    return atb


# =============================================================
# LEITURA DOS PARÂMETROS DE COLETA
# =============================================================

def ler_parametros_coleta(caminho: str) -> dict:
    """
    Lê o arquivo de coleta de literatura e extrai
    os parâmetros PK para calibração da simulação.
    """
    df = pd.read_excel(caminho, sheet_name="coleta")

    params = {}

    def media_valida(coluna):
        if coluna in df.columns:
            vals = pd.to_numeric(df[coluna], errors="coerce").dropna()
            return float(vals.mean()) if len(vals) > 0 else None
        return None

    params["idade_media"]       = media_valida("idade_media_anos") or 55.0
    params["peso_medio"]        = media_valida("peso_medio_kg") or 72.0
    params["sexo_masc_pct"]     = media_valida("percentual_masculino") or 60.0
    params["clcr_medio"]        = media_valida("clearance_creatinina_medio_mL_min") or 60.0
    params["vd_L_kg"]           = media_valida("volume_distribuicao_L_kg") or 0.82
    params["clearance_L_h"]     = media_valida("clearance_farmaco_L_h")
    params["meia_vida_h"]       = media_valida("meia_vida_h")
    params["dose_mg_kg"]        = media_valida("dose_mg_kg") or 15.0
    params["intervalo_h"]       = media_valida("intervalo_doses_h") or 12.0
    params["tempo_infusao_h"]   = media_valida("tempo_infusao_h") or 1.0
    params["creat_basal"]       = media_valida("creatinina_basal_mg_dL") or 1.2
    params["albumina_basal"]    = media_valida("albumina_basal_g_dL")

    # Creatininas seriadas
    for tempo in ["24h", "48h", "72h", "7dias"]:
        col = f"creatinina_{tempo}_mg_dL"
        params[f"creat_{tempo}"] = media_valida(col)

    return params


# =============================================================
# SIMULAÇÃO DE PACIENTES VIRTUAIS
# =============================================================

def simular_pacientes(atb: str, params: dict, n: int = N_PACIENTES) -> pd.DataFrame:
    """
    Gera dataset de pacientes virtuais calibrado pelos
    parâmetros extraídos da literatura.
    """

    config_atb = ATB_CONFIG.get(atb, {})
    pk = config_atb.get("pk_populacional", {})
    admin = config_atb.get("administracao", {})

    registros = []

    for i in range(n):

        # Dados demográficos
        idade   = np.random.normal(params["idade_media"], params["idade_media"] * 0.15)
        idade   = float(np.clip(idade, 18, 90))
        peso    = np.random.normal(params["peso_medio"], params["peso_medio"] * 0.12)
        peso    = float(np.clip(peso, 40, 150))
        sexo    = "masculino" if np.random.rand() < params["sexo_masc_pct"] / 100 else "feminino"

        # Creatinina basal
        creat_basal = np.random.normal(params["creat_basal"], params["creat_basal"] * 0.25)
        creat_basal = float(np.clip(creat_basal, 0.4, 10.0))

        # Progressão da creatinina ao longo do tempo
        creatininas = {"basal": creat_basal}
        creat_anterior = creat_basal
        for tempo in ["24h", "48h", "72h", "7dias"]:
            val_lit = params.get(f"creat_{tempo}")
            if val_lit:
                fator = val_lit / params["creat_basal"]
                nova = creat_anterior * np.random.normal(fator, fator * 0.1)
            else:
                nova = creat_anterior * np.random.normal(1.05, 0.05)
            creatininas[tempo] = float(np.clip(nova, 0.4, 15.0))
            creat_anterior = creatininas[tempo]

        # Albumina
        albumina = None
        if params.get("albumina_basal"):
            albumina = float(np.clip(
                np.random.normal(params["albumina_basal"], 0.4), 1.5, 5.0
            ))

        # Volume de distribuição
        vd_L_kg = pk.get("vd_L_kg") or params["vd_L_kg"]
        vd_L_kg = float(np.random.normal(vd_L_kg, vd_L_kg * 0.15))
        vd_L_kg = float(np.clip(vd_L_kg, vd_L_kg * 0.5, vd_L_kg * 2.0))
        vd_total = vd_L_kg * peso

        # Dose
        dose_mg_kg = float(np.random.normal(params["dose_mg_kg"], params["dose_mg_kg"] * 0.1))
        dose_mg_kg = float(np.clip(dose_mg_kg, params["dose_mg_kg"] * 0.5, params["dose_mg_kg"] * 1.5))
        dose_mg    = dose_mg_kg * peso

        intervalo_h     = params["intervalo_h"]
        tempo_infusao_h = admin.get("tempo_infusao_h") or params["tempo_infusao_h"]

        # Clearance e concentrações seriadas
        concentracoes = {}
        aucs          = {}
        clearances    = {}

        for tempo_label, creat_val in creatininas.items():

            fator_correcao = pk.get("fator_correcao_clcr", 1.0)
            clcr = calcular_clcr_cockcroft_gault(
                idade=idade,
                peso_kg=peso,
                sexo=sexo,
                creatinina_mg_dL=creat_val,
                fator_correcao=fator_correcao,
            )

            # Clearance do fármaco
            if params.get("clearance_L_h"):
                # Usa valor da literatura com proporcionalidade ao ClCr
                cl_ref   = params["clearance_L_h"]
                clcr_ref = params["clcr_medio"]
                cl_L_h   = cl_ref * (clcr / clcr_ref)
            elif pk.get("equacao_clearance") and "3.66" in str(pk.get("equacao_clearance", "")):
                # Equação de Matzke (vancomicina)
                cl_L_h = (3.66 + 0.689 * clcr) / 1000 * 60
            elif params.get("meia_vida_h") or pk.get("meia_vida_h"):
                meia_vida = pk.get("meia_vida_h") or params["meia_vida_h"]
                ke        = 0.693 / meia_vida
                cl_L_h    = ke * vd_total
            else:
                cl_L_h = vd_total * 0.693 / 6.0

            cl_L_h = float(np.clip(cl_L_h, 0.1, 50.0))
            clearances[tempo_label] = round(cl_L_h, 3)

            # Curva de concentração
            curva = gerar_curva_concentracao(
                dose_mg=dose_mg,
                vd_L=vd_total,
                clearance_L_h=cl_L_h,
                intervalo_h=intervalo_h,
                tempo_infusao_h=tempo_infusao_h,
                numero_doses=4,
            )

            # Adiciona ruído biológico de 8%
            vale = curva["vale_mg_L"] * np.random.normal(1.0, 0.08)
            concentracoes[tempo_label] = round(float(np.clip(vale, 0.1, 200.0)), 2)

            auc = calcular_auc(
                curva["tempos_h"], curva["concentracoes"], intervalo_h=intervalo_h
            )
            aucs[tempo_label] = round(auc, 1)

        # Monta registro do paciente
        registro = {
            "paciente_id"           : i + 1,
            "origem"                : "simulado_literatura",
            "data_simulacao"        : datetime.today().strftime("%Y-%m-%d"),
            "idade"                 : round(idade, 1),
            "peso_kg"               : round(peso, 1),
            "sexo"                  : sexo,
            "creat_basal"           : round(creatininas["basal"], 2),
            "creat_24h"             : round(creatininas["24h"], 2),
            "creat_48h"             : round(creatininas["48h"], 2),
            "creat_72h"             : round(creatininas["72h"], 2),
            "creat_7dias"           : round(creatininas["7dias"], 2),
            "albumina_basal"        : round(albumina, 2) if albumina else None,
            "vd_L_kg"               : round(vd_L_kg, 3),
            "vd_total_L"            : round(vd_total, 1),
            "dose_mg_kg"            : round(dose_mg_kg, 2),
            "dose_mg"               : round(dose_mg, 1),
            "intervalo_h"           : intervalo_h,
            "tempo_infusao_h"       : tempo_infusao_h,
            "cl_basal_L_h"          : clearances["basal"],
            "cl_24h_L_h"            : clearances["24h"],
            "cl_48h_L_h"            : clearances["48h"],
            "cl_72h_L_h"            : clearances["72h"],
            "cl_7dias_L_h"          : clearances["7dias"],
            "conc_basal_mg_L"       : concentracoes["basal"],
            "conc_24h_mg_L"         : concentracoes["24h"],
            "conc_48h_mg_L"         : concentracoes["48h"],
            "conc_72h_mg_L"         : concentracoes["72h"],
            "conc_7dias_mg_L"       : concentracoes["7dias"],
            "auc_basal_mg_h_L"      : aucs["basal"],
            "auc_24h_mg_h_L"        : aucs["24h"],
            "auc_48h_mg_h_L"        : aucs["48h"],
            "auc_72h_mg_h_L"        : aucs["72h"],
            "auc_7dias_mg_h_L"      : aucs["7dias"],
        }

        registros.append(registro)

    return pd.DataFrame(registros)


# =============================================================
# EXECUÇÃO PRINCIPAL
# =============================================================

def rodar_simulacao():
    """
    Descobre automaticamente todos os arquivos coletaliteratura_*.xlsx
    e gera o dataset simulado correspondente para cada ATB.
    """

    arquivos = descobrir_arquivos_coleta()

    if not arquivos:
        print("Nenhum arquivo coletaliteratura_*.xlsx encontrado em data/.")
        return

    for caminho in arquivos:
        atb = extrair_atb_do_nome(caminho)
        print(f"\nSimulando: {atb}")
        print(f"  Arquivo de coleta: {caminho}")

        params = ler_parametros_coleta(caminho)
        df     = simular_pacientes(atb=atb, params=params)

        pasta_saida  = os.path.dirname(caminho)
        nome_saida   = f"{PREFIXO_SIMULADO}{atb}.csv"
        caminho_saida = os.path.join(pasta_saida, nome_saida)

        df.to_csv(caminho_saida, index=False)
        print(f"  Dataset salvo: {caminho_saida}")
        print(f"  Pacientes simulados: {len(df)}")
        print(f"  Colunas: {len(df.columns)}")


if __name__ == "__main__":
    rodar_simulacao()