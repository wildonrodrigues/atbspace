# =============================================================
# ATBSPACE - Configurações Globais
# =============================================================

import os

# -------------------------------------------------------------
# PATHS
# -------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")

# -------------------------------------------------------------
# SIMULAÇÃO
# -------------------------------------------------------------
RANDOM_STATE    = 42
N_PACIENTES     = 400
TEST_SIZE       = 0.2
TEMPOS_HORAS    = [0, 24, 48, 72, 168]

# -------------------------------------------------------------
# MODELOS DISPONÍVEIS
# -------------------------------------------------------------
MODELOS_DISPONIVEIS = [
    "pkpd",
    "bayesianridge",
    "lightgbm",
    "gradientboosting",
    "randomforest",
]

# -------------------------------------------------------------
# ESTRUTURAS PK DISPONÍVEIS
# -------------------------------------------------------------
ESTRUTURAS = ["mono", "bi", "tri"]

# Parâmetros estruturais da "verdade" simulada.
# Valores típicos de popPK de meropenem (J Pharm Sci 2022; Frontiers 2021).
# Reconciliar com coletaliteratura_meropenem.xlsx antes de publicar.
# IIV e efeitos de covariável são ilustrativos — definem o gap aprendível.
ESTRUTURA_PARAMS = {
    "_default": {
        "CL_POP"  : 7.5,    # L/h — clearance populacional de referência
        "V1_POP"  : 15.9,   # L   — volume central
        "V2_POP"  : 14.8,   # L   — volume periférico 1
        "Q2_POP"  : 15.8,   # L/h — clearance intercompartimental 1
        "V3_POP"  : 6.0,    # L   — volume periférico 2 (só tri)
        "Q3_POP"  : 4.0,    # L/h — clearance intercompartimental 2 (só tri)
        "CLCR_REF": 90.0,   # mL/min — ClCr de referência para o típico
        "CV_CL"   : 0.22,   # coeficiente de variação IIV — CL
        "CV_V1"   : 0.15,   # coeficiente de variação IIV — V1
        "CV_V2"   : 0.20,   # coeficiente de variação IIV — V2
        "CV_Q"    : 0.25,   # coeficiente de variação IIV — Q
    },
}

# -------------------------------------------------------------
# NOMENCLATURA DE ARQUIVOS
# -------------------------------------------------------------
PREFIXO_COLETA    = "coletaliteratura_"
PREFIXO_SIMULADO  = "pacientesimulado_"
PREFIXO_REAL      = "pacientesreais_"

# =============================================================
# ANTIMICROBIANOS
# =============================================================

ATB_CONFIG = {

    # ---------------------------------------------------------
    # VANCOMICINA
    # ---------------------------------------------------------
    "vancomicina": {
        "nome_completo"         : "Vancomicina",
        "classe"                : "glicopeptideo",
        "mecanismo_pk"          : "tempo_dependente",

        "alvo_fd": {
            "indice"            : "AUC/MIC",
            "alvo_minimo"       : 400,
            "alvo_maximo"       : 600,
            "unidade"           : "mg.h/L",
        },

        "concentracao_alvo": {
            "minimo_mg_L"       : 15.0,
            "maximo_mg_L"       : 20.0,
        },

        "pk_populacional": {
            "vd_L_kg"                   : 0.7,
            "equacao_clearance"         : "CL = 3.66 + 0.689 * ClCr",
            "meia_vida_h"               : 6.0,
        },

        # Dosagem baseada no guideline ASHP/IDSA 2020
        "dosagem": {
            "dose_ataque_mg_kg"         : 25.0,   # 20-35 mg/kg (usa 25)
            "dose_ataque_max_mg"        : 3000,
            "dose_manutencao_mg_kg"     : 17.5,   # 15-20 mg/kg (usa 17.5)
            "dose_diaria_max_mg"        : 4500,
            "auc_alvo_min"              : 400,
            "auc_alvo_max"              : 600,
            "mic_referencia_mg_L"       : 1.0,
        },

        # Padrões populacionais — usados quando coleta não tem o dado
        "populacao_padrao": {
            "idade_media_anos"          : 55.0,
            "peso_medio_kg"             : 72.0,
            "percentual_masculino"      : 62.0,
            "clcr_medio_mL_min"         : 60.0,
            "albumina_basal_g_dL"       : 3.2,
        },

        "administracao": {
            "via"                       : "intravenosa",
            "tempo_infusao_h"           : 1.5,
            "intervalo_opcoes_h"        : [6, 8, 12, 24],
        },

        "ajuste_renal": {
            ">50"   : {"dose_mg_kg": 17.5, "intervalo_h": 12},
            "20-50" : {"dose_mg_kg": 17.5, "intervalo_h": 24},
            "10-20" : {"dose_mg_kg": 15.0, "intervalo_h": 48},
            "<10"   : {"dose_mg_kg": 15.0, "intervalo_h": 96},
        },

        # Toxicidade — guideline ASHP/IDSA 2020
        "toxicidade": {
            "tipo"                      : "nefrotoxicidade",
            "auc_limiar_mg_h_L"         : 600,
            "concentracao_limiar_mg_L"  : 20.0,
            "concentracao_alerta_mg_L"  : 25.0,
            "fonte"                     : "ASHP/IDSA 2020",
        },
    },

    # ---------------------------------------------------------
    # CEFTAZIDIMA/AVIBACTAM
    # ---------------------------------------------------------
    "cazavi": {
        "nome_completo"         : "Ceftazidima/Avibactam",
        "classe"                : "cefalosporina_inibidor_betalactamase",
        "mecanismo_pk"          : "tempo_dependente",

        "alvo_fd": {
            "ceftazidima": {
                "indice"                : "T>MIC",
                "alvo_percentual"       : 50,
                "concentracao_alvo_mg_L": 8.0,
                "unidade"               : "% do intervalo",
            },
            "avibactam": {
                "indice"                : "T>limiar",
                "limiar_mg_L"           : 1.0,
                "alvo_percentual"       : 100,
                "unidade"               : "% do intervalo",
            },
        },

        "pk_populacional": {
            "vd_ceftazidima_L"          : 17.0,
            "vd_avibactam_L"            : 22.0,
            "meia_vida_ceftazidima_h"   : 2.7,
            "meia_vida_avibactam_h"     : 2.7,
            "clearance_L_h"             : 5.5,
        },

        # Padrões populacionais — usados quando coleta não tem o dado
        "populacao_padrao": {
            "idade_media_anos"          : 58.0,
            "peso_medio_kg"             : 72.0,
            "percentual_masculino"      : 55.0,
            "clcr_medio_mL_min"         : 60.0,
        },

        "administracao": {
            "via"                       : "intravenosa",
            "tempo_infusao_h"           : 2.0,
            "intervalo_opcoes_h"        : [8, 12, 24, 48],
        },

        "ajuste_renal": {
            ">50"   : {"dose_mg": 2500, "intervalo_h": 8},
            "31-50" : {"dose_mg": 1250, "intervalo_h": 8},
            "16-30" : {"dose_mg": 940,  "intervalo_h": 12},
            "6-15"  : {"dose_mg": 940,  "intervalo_h": 24},
            "<6"    : {"dose_mg": 940,  "intervalo_h": 48},
        },

        "terapia_renal_substitutiva": {
            "hemofiltração continua"    : {"dose_mg": 940,  "intervalo_h": 8},
            "hemodiálise continua"      : {"dose_mg": 940,  "intervalo_h": 12},
            "hemodiálise"               : {"dose_mg": 940,  "intervalo_h": 48},
        },

        # Toxicidade — limiar de neurotoxicidade da ceftazidima
        # Fonte: Bui 2024 (PubMed 38305827); CRRT 104 mg/L (PMC12888888)
        "toxicidade": {
            "tipo"                      : "neurotoxicidade",
            "concentracao_limiar_mg_L"  : 78.0,
            "concentracao_alerta_mg_L"  : 104.0,
            "fonte"                     : "Bui et al. 2024",
        },
    },

    # ---------------------------------------------------------
    # MEROPENEM
    # ---------------------------------------------------------
    "meropenem": {
        "nome_completo"         : "Meropenem",
        "classe"                : "carbapenem",
        "mecanismo_pk"          : "tempo_dependente",

        "alvo_fd": {
            "indice"                    : "T>MIC",
            "alvo_minimo_percentual"    : 40,
            "alvo_otimo_percentual"     : 100,
            "multiplo_mic_alvo"         : 4.0,
            "concentracao_alvo_mg_L"    : 4.0,
            "unidade"                   : "% do intervalo",
        },

        "pk_populacional": {
            # Paciente crítico — base: múltiplos estudos PopPK
            # Fonte: Jaruratanasirikul 2022 (AAC), Roberts 2024 (PMC)
            "vd_L_kg"                   : 0.30,
            "vd_L"                      : 21.0,
            "clearance_L_h"             : 4.8,
            "meia_vida_h"               : 1.0,

            # Paciente em CRRT
            # Fonte: Dovepress 2025
            "clearance_crrt_L_h"        : 2.89,
            "vd_crrt_L"                 : 26.0,
        },

        # Padrões populacionais — usados quando coleta não tem o dado
        # Fonte: Jaruratanasirikul 2022 (AAC) — 52 pacientes críticos
        "populacao_padrao": {
            "idade_media_anos"          : 63.0,
            "peso_medio_kg"             : 61.5,
            "percentual_masculino"      : 53.0,
            "clcr_medio_mL_min"         : 44.6,
            "albumina_basal_g_dL"       : 2.8,
        },

        "administracao": {
            "via"                       : "intravenosa",
            "tempo_infusao_h"           : 3.0,
            "tempo_infusao_padrao_h"    : 0.5,
            "intervalo_opcoes_h"        : [6, 8],
            "infusao_estendida"         : True,
        },

        "ajuste_renal": {
            ">50"   : {"dose_mg": 1000, "intervalo_h": 8},
            "26-50" : {"dose_mg": 1000, "intervalo_h": 12},
            "10-25" : {"dose_mg": 500,  "intervalo_h": 12},
            "<10"   : {"dose_mg": 500,  "intervalo_h": 24},
        },

        "terapia_renal_substitutiva": {
            "hemofiltração continua"    : {"dose_mg": 1000, "intervalo_h": 8},
            "hemodiálise continua"      : {"dose_mg": 500,  "intervalo_h": 8},
            "hemodiálise"               : {"dose_mg": 500,  "intervalo_h": 24},
        },

        # Toxicidade — neurotoxicidade por acúmulo
        # Fonte: Nature 2025 (s41598-025-20630-5); PMC8306322
        "toxicidade": {
            "tipo"                      : "neurotoxicidade",
            "concentracao_limiar_mg_L"  : 45.0,
            "concentracao_alerta_mg_L"  : 64.0,
            "fonte"                     : "Nature 2025; beta-lactam ICU review",
        },
    },
}