# =============================================================
# ATBSPACE - Comparação de Cenários
# =============================================================

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config import ATB_CONFIG
from pk_engine import (
    gerar_curva_concentracao,
    calcular_t_mic,
    calcular_auc,
    calcular_auc_mic,
    calcular_pico_mic,
    avaliar_alvo_fd,
    avaliar_toxicidade,
)
from timing import calcular_timing


# =============================================================
# ESTRUTURA DE UM CENÁRIO
# =============================================================

def criar_cenario(
    nome: str,
    dose_mg: float,
    intervalo_h: float,
    vd_total_L: float,
    clearance_L_h: float,
    tempo_infusao_h: float,
    horario_primeira_dose: str = "08:00",
    dose_foi_administrada: bool = True,
    horario_dose_perdida: str = None,
    numero_doses: int = 5,
    mic_mg_L: float = 1.0,
    atb: str = "vancomicina",
    modelo: str = "pkpd",
    dados_paciente: dict = None,
    estrutura: str = "mono",
) -> dict:
    """
    Define um cenário de dosagem para simulação.
    """
    return {
        "nome"                  : nome,
        "dose_mg"               : dose_mg,
        "intervalo_h"           : intervalo_h,
        "vd_total_L"            : vd_total_L,
        "clearance_L_h"         : clearance_L_h,
        "tempo_infusao_h"       : tempo_infusao_h,
        "horario_primeira_dose" : horario_primeira_dose,
        "dose_foi_administrada" : dose_foi_administrada,
        "horario_dose_perdida"  : horario_dose_perdida,
        "numero_doses"          : numero_doses,
        "mic_mg_L"              : mic_mg_L,
        "atb"                   : atb,
        "modelo"                : modelo,
        "estrutura"             : estrutura,
        "dados_paciente"        : dados_paciente or {},
    }


# =============================================================
# SIMULAÇÃO DE UM CENÁRIO
# =============================================================

def simular_cenario(cenario: dict) -> dict:
    """
    Simula a curva de concentração de um cenário
    e calcula os índices farmacodinâmicos.
    """

    modelo = cenario.get("modelo", "pkpd")

    # Curva PK clássica: sempre usada para cálculo de índices FD e toxicidade
    curva_pk = gerar_curva_concentracao(
        dose_mg         = cenario["dose_mg"],
        vd_L            = cenario["vd_total_L"],
        clearance_L_h   = cenario["clearance_L_h"],
        intervalo_h     = cenario["intervalo_h"],
        tempo_infusao_h = cenario["tempo_infusao_h"],
        numero_doses    = cenario["numero_doses"],
    )

    # Curva do modelo ML: usada para visualização (pontos sobrepostos)
    if modelo != "pkpd":
        from models import predizer_curva_modelo
        dados_base = dict(cenario.get("dados_paciente", {}))
        dados_base.update({
            "vd_total_L"          : cenario["vd_total_L"],
            "dose_mg_kg"          : cenario["dose_mg"] / max(dados_base.get("peso_kg", 1), 0.1),
            "dose_mg"             : cenario["dose_mg"],
            "intervalo_h"         : cenario["intervalo_h"],
            "tempo_infusao_h"     : cenario["tempo_infusao_h"],
            "clearance_atual_L_h" : cenario["clearance_L_h"],
        })
        try:
            curva_ml = predizer_curva_modelo(
                atb             = cenario["atb"],
                nome_modelo     = modelo,
                dados_base      = dados_base,
                duracao_total_h = cenario["numero_doses"] * cenario["intervalo_h"],
                resolucao_h     = 0.5,
                estrutura       = cenario.get("estrutura", "mono"),
            )
        except Exception:
            curva_ml = None
    else:
        curva_ml = None

    # A curva que alimenta os índices clínicos segue o modelo do cenário.
    # Quando o cenário usa "pkpd", ou quando o modelo treinado não pôde ser
    # carregado (curva_ml = None), os índices caem de volta na curva PK/PD.
    if curva_ml is not None:
        curva_base = curva_ml
        origem     = modelo
    else:
        curva_base = curva_pk
        origem     = "pkpd"

    curva = curva_base

    tempos = curva_base["tempos_h"]
    concs  = curva_base["concentracoes"]
    mic    = cenario["mic_mg_L"]
    atb    = cenario["atb"]

    # Índices farmacodinâmicos
    auc    = calcular_auc(tempos, concs, cenario["intervalo_h"])
    t_mic  = calcular_t_mic(tempos, concs, mic, cenario["intervalo_h"])
    auc_mic = calcular_auc_mic(auc, mic)
    pico_mic = calcular_pico_mic(curva_base["pico_mg_L"], mic)

    resultado_fd = {
        "t_mic_percentual"  : t_mic,
        "auc_mic"           : auc_mic,
        "pico_mic"          : pico_mic,
        "vale_mg_L"         : curva_base["vale_mg_L"],
        "mic_mg_L"          : mic,
    }

    avaliacao = avaliar_alvo_fd(atb, resultado_fd)
    atingiu_fd = all(v["atingiu_alvo"] for v in avaliacao.values()) if avaliacao else None

    toxicidade = avaliar_toxicidade(
        atb,
        pico_mg_L=curva_base["pico_mg_L"],
        vale_mg_L=curva_base["vale_mg_L"],
        auc_mg_h_L=auc,
    )
    toxicidade_nivel = toxicidade.get("nivel", "indefinido")
    toxico = toxicidade_nivel in ["atencao", "risco"]

    # Janela terapêutica
    config_atb  = ATB_CONFIG.get(atb, {})
    conc_alvo   = config_atb.get("concentracao_alvo", {})
    alvo_min    = conc_alvo.get("minimo_mg_L")
    alvo_max    = conc_alvo.get("maximo_mg_L")

    # Horários reais das doses
    formato = "%H:%M"
    hoje    = datetime.today()
    t_inicio = datetime.strptime(cenario["horario_primeira_dose"], formato).replace(
        year=hoje.year, month=hoje.month, day=hoje.day
    )
    horarios_doses = [
        (t_inicio + timedelta(hours=i * cenario["intervalo_h"])).strftime(formato)
        for i in range(cenario["numero_doses"])
    ]

    return {
        "nome"              : cenario["nome"],
        "dose_mg"           : cenario["dose_mg"],
        "intervalo_h"       : cenario["intervalo_h"],
        "modelo"            : modelo,
        "origem_indices"    : origem,
        # curva PK clássica — sempre presente, usada no gráfico como referência
        "tempos_h"          : curva_pk["tempos_h"],
        "concentracoes"     : curva_pk["concentracoes"],
        "pico_mg_L"         : curva_base["pico_mg_L"],
        "vale_mg_L"         : curva_base["vale_mg_L"],
        "auc_mg_h_L"        : auc,
        "t_mic_percentual"  : t_mic,
        "auc_mic"           : auc_mic,
        "pico_mic"          : pico_mic,
        "mic_mg_L"          : mic,
        "avaliacao_fd"      : avaliacao,
        "atingiu_alvo_fd"   : "sim" if atingiu_fd else "nao" if atingiu_fd is not None else "indefinido",
        "toxicidade"        : toxicidade,
        "toxicidade_nivel"  : toxicidade_nivel,
        "toxico"            : toxico,
        "alvo_min_mg_L"     : alvo_min,
        "alvo_max_mg_L"     : alvo_max,
        "horarios_doses"    : horarios_doses,
        "curva_ml"          : curva_ml,   # None quando modelo == "pkpd"
    }


# =============================================================
# COMPARAÇÃO DE MÚLTIPLOS CENÁRIOS
# =============================================================

def comparar_cenarios(cenarios: list) -> dict:
    """
    Simula e compara múltiplos cenários.

    O melhor cenário é escolhido apenas entre os cenários classificados
    como seguros pela avaliação de toxicidade. Cenários em atenção ou risco
    não podem ser selecionados como melhor cenário.
    """

    resultados = [simular_cenario(c) for c in cenarios]

    linhas = []
    for r in resultados:
        linhas.append({
            "cenario"               : r["nome"],
            "dose_mg"               : r["dose_mg"],
            "intervalo_h"           : r["intervalo_h"],
            "modelo"                : r.get("modelo", "pkpd"),
            "pico_mg_L"             : r["pico_mg_L"],
            "vale_mg_L"             : r["vale_mg_L"],
            "auc_mg_h_L"            : r["auc_mg_h_L"],
            "t_mic_percentual"      : r["t_mic_percentual"],
            "auc_mic"               : r["auc_mic"],
            "mic_mg_L"              : r.get("mic_mg_L"),
            "vale_mic"              : round(r["vale_mg_L"] / r.get("mic_mg_L", np.nan), 2) if r.get("mic_mg_L") else None,
            "vale_acima_mic"        : "sim" if r.get("mic_mg_L") and r["vale_mg_L"] >= r["mic_mg_L"] else "nao",
            "atingiu_alvo_fd"       : r["atingiu_alvo_fd"],
            "toxicidade"            : r["toxicidade_nivel"],
            "toxico"                : "sim" if r["toxico"] else "nao",
        })

    tabela = pd.DataFrame(linhas)

    melhor = None
    mensagem_melhor = None
    atb = cenarios[0]["atb"] if cenarios else None

    candidatos = [r for r in resultados if r.get("toxicidade_nivel") == "seguro"]

    if not candidatos:
        mensagem_melhor = "Nenhum cenário seguro. Ajuste dose, intervalo ou modo de infusão antes de escolher um regime."
    elif atb:
        config_atb = ATB_CONFIG.get(atb, {})
        alvo_fd    = config_atb.get("alvo_fd", {})

        if "AUC/MIC" in str(alvo_fd.get("indice", "")):
            alvo_min = alvo_fd.get("alvo_minimo")
            alvo_max = alvo_fd.get("alvo_maximo")
            if alvo_min is not None and alvo_max is not None:
                alvo_centro = (alvo_min + alvo_max) / 2
                candidatos_no_alvo = [
                    r for r in candidatos
                    if alvo_min <= r.get("auc_mic", -np.inf) <= alvo_max
                ]
                base = candidatos_no_alvo or candidatos
                melhor = min(base, key=lambda r: abs(r["auc_mic"] - alvo_centro))["nome"]
            else:
                melhor = min(candidatos, key=lambda r: r["auc_mic"])["nome"]

        elif "T>MIC" in str(alvo_fd):
            alvo_pct = (
                alvo_fd.get("alvo_otimo_percentual")
                or alvo_fd.get("alvo_percentual")
                or alvo_fd.get("alvo_minimo_percentual")
                or 40
            )
            candidatos_no_alvo = [
                r for r in candidatos
                if r.get("atingiu_alvo_fd") == "sim"
            ]

            if candidatos_no_alvo:
                # Entre cenários seguros que atingem o alvo, prefere menor exposição.
                melhor = min(
                    candidatos_no_alvo,
                    key=lambda r: (
                        r["auc_mg_h_L"],
                        r["pico_mg_L"],
                        r["dose_mg"] * (24 / r["intervalo_h"]),
                    )
                )["nome"]
            else:
                # Se nenhum cenário seguro atinge o alvo, escolhe o que mais se aproxima,
                # ainda mantendo a restrição de segurança.
                melhor = max(
                    candidatos,
                    key=lambda r: (r["t_mic_percentual"], -r["auc_mg_h_L"])
                )["nome"]
        else:
            melhor = min(candidatos, key=lambda r: (r["auc_mg_h_L"], r["pico_mg_L"]))["nome"]

    return {
        "resultados"       : resultados,
        "tabela_resumo"    : tabela,
        "melhor_cenario"   : melhor,
        "mensagem_melhor"  : mensagem_melhor,
    }


# =============================================================
# CENÁRIO COM DOSE SÉRICA DISPONÍVEL
# =============================================================

def incorporar_doseamento_real(
    resultado_cenario: dict,
    concentracao_real_mg_L: float,
    horario_coleta: str,
    horario_ultima_dose: str,
) -> dict:
    """
    Incorpora um doseamento sérico real ao cenário
    e calcula o erro do modelo.

    Retorna o resultado atualizado com:
        erro_absoluto_mg_L  : diferença entre predito e real
        erro_percentual     : erro relativo
        interpretacao       : subdose / dentro do alvo / sobredose
    """

    formato = "%H:%M"
    hoje    = datetime.today()

    t_coleta = datetime.strptime(horario_coleta, formato).replace(
        year=hoje.year, month=hoje.month, day=hoje.day
    )
    t_ultima = datetime.strptime(horario_ultima_dose, formato).replace(
        year=hoje.year, month=hoje.month, day=hoje.day
    )

    if t_coleta < t_ultima:
        t_coleta += timedelta(days=1)

    tempo_apos_dose_h = (t_coleta - t_ultima).total_seconds() / 3600

    # Concentração predita no mesmo momento da coleta
    tempos = np.array(resultado_cenario["tempos_h"])
    concs  = np.array(resultado_cenario["concentracoes"])

    idx       = np.argmin(np.abs(tempos % resultado_cenario["intervalo_h"] - tempo_apos_dose_h))
    conc_pred = round(float(concs[idx]), 2)

    erro_abs = round(abs(conc_pred - concentracao_real_mg_L), 2)
    erro_pct = round((erro_abs / concentracao_real_mg_L) * 100, 1) if concentracao_real_mg_L > 0 else None

    alvo_min = resultado_cenario.get("alvo_min_mg_L")
    alvo_max = resultado_cenario.get("alvo_max_mg_L")

    if alvo_min and alvo_max:
        if concentracao_real_mg_L < alvo_min:
            interpretacao = "subdose"
        elif concentracao_real_mg_L > alvo_max:
            interpretacao = "sobredose"
        else:
            interpretacao = "dentro do alvo"
    else:
        interpretacao = "alvo nao definido"

    resultado_cenario.update({
        "concentracao_real_mg_L"    : concentracao_real_mg_L,
        "horario_coleta"            : horario_coleta,
        "concentracao_predita_mg_L" : conc_pred,
        "tempo_apos_dose_h"         : round(tempo_apos_dose_h, 1),
        "erro_absoluto_mg_L"        : erro_abs,
        "erro_percentual"           : erro_pct,
        "interpretacao_real"        : interpretacao,
    })

    return resultado_cenario