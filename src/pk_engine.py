# =============================================================
# ATBSPACE - Motor de Farmacocinética/Farmacodinâmica
# =============================================================
# Responsável por:
# → Calcular clearance de creatinina (Cockcroft-Gault)
# → Determinar dose por peso e clearance de creatinina
# → Calcular concentração sérica por modelo monocompartimental
# → Calcular índices farmacodinâmicos (T>MIC, AUC/MIC, Pico/MIC)
# → Ajuste por terapia renal substitutiva
# =============================================================

import math
import numpy as np
from config import ATB_CONFIG


# =============================================================
# CLEARANCE DE CREATININA
# =============================================================

def calcular_clcr_cockcroft_gault(
    idade: float,
    peso_kg: float,
    sexo: str,
    creatinina_mg_dL: float,
    fator_correcao: float = 1.0,
) -> float:
    """
    Calcula o clearance de creatinina pela fórmula de Cockcroft-Gault.

    Parâmetros:
    -----------
    idade            : idade em anos
    peso_kg          : peso em kg
    sexo             : "masculino" ou "feminino"
    creatinina_mg_dL : creatinina sérica em mg/dL
    fator_correcao   : fator de correção para populações especiais
                       (ex: 0.65 para hepatopatas — Taber et al. 2003)

    Retorna:
    --------
    clearance de creatinina estimado em mL/min
    """

    if creatinina_mg_dL <= 0:
        raise ValueError("Creatinina deve ser maior que zero.")

    clcr = ((140 - idade) * peso_kg) / (72 * creatinina_mg_dL)

    if sexo.lower() == "feminino":
        clcr *= 0.85

    clcr *= fator_correcao

    return round(clcr, 1)


def calcular_peso_ideal(altura_cm: float, sexo: str) -> float:
    """
    Calcula o peso ideal (fórmula de Devine).

    Parâmetros:
    -----------
    altura_cm : altura em centímetros
    sexo      : "masculino" ou "feminino"

    Retorna:
    --------
    peso ideal em kg
    """

    altura_polegadas = (altura_cm - 152.4) / 2.54

    if sexo.lower() == "masculino":
        peso_ideal = 50 + 2.3 * altura_polegadas
    else:
        peso_ideal = 45.5 + 2.3 * altura_polegadas

    return round(max(peso_ideal, 0), 1)


def calcular_peso_ajustado(peso_real_kg: float, peso_ideal_kg: float) -> float:
    """
    Calcula o peso ajustado para pacientes obesos
    (peso real > 120% do peso ideal).

    Retorna:
    --------
    peso ajustado em kg
    """

    if peso_real_kg > 1.2 * peso_ideal_kg:
        peso_ajustado = peso_ideal_kg + 0.4 * (peso_real_kg - peso_ideal_kg)
        return round(peso_ajustado, 1)

    return peso_real_kg


# =============================================================
# SELEÇÃO DE DOSE POR AJUSTE RENAL
# =============================================================

def selecionar_ajuste_renal(atb: str, clcr_mL_min: float, modalidade_trs: str = None) -> dict:
    """
    Seleciona dose e intervalo conforme clearance de creatinina
    e modalidade de terapia renal substitutiva.

    Parâmetros:
    -----------
    atb             : chave do antimicrobiano no config.py
    clcr_mL_min     : clearance de creatinina em mL/min
    modalidade_trs  : modalidade de terapia renal substitutiva se aplicável

    Retorna:
    --------
    dicionário com dose e intervalo recomendados
    """

    config = ATB_CONFIG.get(atb)
    if config is None:
        raise ValueError(f"Antimicrobiano '{atb}' não encontrado no config.")

    # Terapia renal substitutiva tem prioridade
    if modalidade_trs:
        trs_config = config.get("terapia_renal_substitutiva", {})
        for modalidade, parametros in trs_config.items():
            if modalidade_trs.lower() in modalidade.lower():
                return {
                    "fonte"         : f"terapia_renal_substitutiva: {modalidade}",
                    "dose_mg"       : parametros.get("dose_mg"),
                    "dose_mg_kg"    : parametros.get("dose_mg_kg"),
                    "intervalo_h"   : parametros["intervalo_h"],
                }

    # Ajuste por faixa de clearance de creatinina
    ajuste = config.get("ajuste_renal", {})

    for faixa, parametros in ajuste.items():
        if _clcr_na_faixa(clcr_mL_min, faixa):
            return {
                "fonte"         : f"ajuste_renal: ClCr {faixa} mL/min",
                "dose_mg"       : parametros.get("dose_mg"),
                "dose_mg_kg"    : parametros.get("dose_mg_kg"),
                "intervalo_h"   : parametros["intervalo_h"],
            }

    raise ValueError(f"Faixa de ClCr {clcr_mL_min} mL/min não encontrada para {atb}.")


def _clcr_na_faixa(clcr: float, faixa: str) -> bool:
    """
    Verifica se o clearance de creatinina está dentro de uma faixa.
    Interpreta strings como '>50', '31-50', '<10'.
    """

    faixa = faixa.strip()

    if faixa.startswith(">"):
        return clcr > float(faixa[1:])
    elif faixa.startswith("<"):
        return clcr < float(faixa[1:])
    elif "-" in faixa:
        partes = faixa.split("-")
        return float(partes[0]) <= clcr <= float(partes[1])

    return False


# =============================================================
# MODELO MONOCOMPARTIMENTAL — CONCENTRAÇÃO SÉRICA
# =============================================================

def calcular_concentracao(
    dose_mg: float,
    vd_L: float,
    clearance_L_h: float,
    tempo_apos_dose_h: float,
    tempo_infusao_h: float = 0.5,
    concentracao_inicial_mg_L: float = 0.0,
) -> float:
    """
    Calcula a concentração sérica pelo modelo monocompartimental.

    Parâmetros:
    -----------
    dose_mg                   : dose administrada em mg
    vd_L                      : volume de distribuição em litros
    clearance_L_h             : clearance total em L/h
    tempo_apos_dose_h         : tempo após o início da infusão em horas
    tempo_infusao_h           : duração da infusão em horas
    concentracao_inicial_mg_L : concentração residual antes da dose (vale)

    Retorna:
    --------
    concentração sérica em mg/L
    """

    ke = clearance_L_h / vd_L  # constante de eliminação (h-1)
    taxa_infusao = dose_mg / tempo_infusao_h  # mg/h

    if tempo_apos_dose_h <= tempo_infusao_h:
        # Durante a infusão
        concentracao = (
            concentracao_inicial_mg_L * math.exp(-ke * tempo_apos_dose_h)
            + (taxa_infusao / clearance_L_h)
            * (1 - math.exp(-ke * tempo_apos_dose_h))
        )
    else:
        # Após o término da infusão
        concentracao_fim_infusao = (
            concentracao_inicial_mg_L * math.exp(-ke * tempo_infusao_h)
            + (taxa_infusao / clearance_L_h)
            * (1 - math.exp(-ke * tempo_infusao_h))
        )
        tempo_pos_infusao = tempo_apos_dose_h - tempo_infusao_h
        concentracao = concentracao_fim_infusao * math.exp(-ke * tempo_pos_infusao)

    return round(max(concentracao, 0.0), 2)


def gerar_curva_concentracao(
    dose_mg: float,
    vd_L: float,
    clearance_L_h: float,
    intervalo_h: float,
    tempo_infusao_h: float = 0.5,
    numero_doses: int = 5,
    resolucao_h: float = 0.5,
) -> dict:
    """
    Gera a curva completa de concentração sérica ao longo do tempo
    para múltiplas doses consecutivas.

    A concentração total é calculada por superposição linear. Cada dose
    continua contribuindo para todos os tempos posteriores ao início da
    infusão, e não apenas até o fim do próprio intervalo. Isso é essencial
    para mostrar acúmulo em insuficiência renal.
    """

    if dose_mg <= 0:
        raise ValueError("Dose deve ser maior que zero.")
    if vd_L <= 0:
        raise ValueError("Volume de distribuição deve ser maior que zero.")
    if clearance_L_h <= 0:
        raise ValueError("Clearance deve ser maior que zero.")
    if intervalo_h <= 0:
        raise ValueError("Intervalo deve ser maior que zero.")
    if tempo_infusao_h <= 0:
        raise ValueError("Tempo de infusão deve ser maior que zero.")
    if numero_doses <= 0:
        raise ValueError("Número de doses deve ser maior que zero.")

    tempos = np.arange(0, numero_doses * intervalo_h + resolucao_h, resolucao_h)
    concentracoes = np.zeros(len(tempos), dtype=float)

    ke = clearance_L_h / vd_L
    taxa_infusao = dose_mg / tempo_infusao_h

    for n_dose in range(numero_doses):
        inicio_dose = n_dose * intervalo_h
        tempo_relativo = tempos - inicio_dose
        mascara = tempo_relativo >= 0

        if not np.any(mascara):
            continue

        tr = tempo_relativo[mascara]
        contribuicao = np.zeros(len(tr), dtype=float)

        durante_infusao = tr <= tempo_infusao_h
        apos_infusao = ~durante_infusao

        contribuicao[durante_infusao] = (
            (taxa_infusao / clearance_L_h)
            * (1 - np.exp(-ke * tr[durante_infusao]))
        )

        concentracao_fim_infusao = (
            (taxa_infusao / clearance_L_h)
            * (1 - np.exp(-ke * tempo_infusao_h))
        )
        contribuicao[apos_infusao] = (
            concentracao_fim_infusao
            * np.exp(-ke * (tr[apos_infusao] - tempo_infusao_h))
        )

        concentracoes[mascara] += contribuicao

    return {
        "tempos_h"      : list(tempos),
        "concentracoes" : [round(float(c), 2) for c in concentracoes],
        "pico_mg_L"     : round(float(np.max(concentracoes)), 2),
        "vale_mg_L"     : round(float(concentracoes[-1]), 2),
        "tempo_pico_h"  : round(float(tempos[np.argmax(concentracoes)]), 1),
    }


# =============================================================
# ÍNDICES FARMACODINÂMICOS
# =============================================================

def calcular_t_mic(
    tempos_h: list,
    concentracoes: list,
    mic_mg_L: float,
    intervalo_h: float,
) -> float:
    """
    Calcula o percentual do último intervalo em que a concentração
    permanece acima da MIC.

    O cálculo usa interpolação linear entre dois pontos consecutivos.
    Isso evita erro visual quando a curva cruza a MIC entre dois tempos
    da malha numérica.
    """

    if mic_mg_L <= 0:
        raise ValueError("MIC deve ser maior que zero.")
    if intervalo_h <= 0:
        raise ValueError("Intervalo deve ser maior que zero.")

    tempos = np.asarray(tempos_h, dtype=float)
    concs = np.asarray(concentracoes, dtype=float)

    if len(tempos) != len(concs) or len(tempos) < 2:
        return 0.0

    ordem = np.argsort(tempos)
    tempos = tempos[ordem]
    concs = concs[ordem]

    tempo_fim = float(tempos[-1])
    tempo_inicio = tempo_fim - float(intervalo_h)

    # Garante pontos exatamente no começo e no fim do intervalo avaliado.
    grade = [tempo_inicio]
    grade.extend([float(t) for t in tempos if tempo_inicio < t < tempo_fim])
    grade.append(tempo_fim)
    grade = np.asarray(sorted(set(grade)), dtype=float)
    conc_grade = np.interp(grade, tempos, concs)

    tempo_acima = 0.0
    for t0, t1, c0, c1 in zip(grade[:-1], grade[1:], conc_grade[:-1], conc_grade[1:]):
        dt = float(t1 - t0)
        if dt <= 0:
            continue
        acima0 = c0 >= mic_mg_L
        acima1 = c1 >= mic_mg_L

        if acima0 and acima1:
            tempo_acima += dt
        elif acima0 != acima1 and c1 != c0:
            frac = (mic_mg_L - c0) / (c1 - c0)
            frac = min(max(float(frac), 0.0), 1.0)
            t_cruzamento = t0 + frac * dt
            if acima0:
                tempo_acima += max(0.0, t_cruzamento - t0)
            else:
                tempo_acima += max(0.0, t1 - t_cruzamento)

    t_mic = max(0.0, min(100.0, tempo_acima / float(intervalo_h) * 100.0))
    return round(float(t_mic), 1)


def calcular_auc(
    tempos_h: list,
    concentracoes: list,
    intervalo_h: float,
) -> float:
    """
    Calcula a área sob a curva concentração-tempo (AUC)
    no último intervalo (estado estacionário).

    Retorna:
    --------
    AUC em mg.h/L
    """

    tempos = np.array(tempos_h)
    concs = np.array(concentracoes)

    tempo_inicio = tempos[-1] - intervalo_h
    mascara = tempos >= tempo_inicio

    auc = np.trapz(concs[mascara], tempos[mascara])

    return round(float(auc), 1)


def calcular_auc_mic(auc_mg_h_L: float, mic_mg_L: float) -> float:
    """
    Calcula a razão AUC/MIC.

    Retorna:
    --------
    razão AUC/MIC em h
    """

    if mic_mg_L <= 0:
        raise ValueError("MIC deve ser maior que zero.")

    return round(auc_mg_h_L / mic_mg_L, 1)


def calcular_pico_mic(pico_mg_L: float, mic_mg_L: float) -> float:
    """
    Calcula a razão Pico/MIC.

    Retorna:
    --------
    razão Pico/MIC
    """

    if mic_mg_L <= 0:
        raise ValueError("MIC deve ser maior que zero.")

    return round(pico_mg_L / mic_mg_L, 1)


def avaliar_alvo_fd(atb: str, resultado_fd: dict) -> dict:
    """
    Avalia se os índices farmacodinâmicos atingiram o alvo
    definido no config.py para o antimicrobiano.

    Para beta-lactâmicos em infusão intermitente, o T>MIC isolado pode
    classificar como adequado um regime cujo vale final caiu abaixo da MIC.
    Por isso, quando vale_mg_L e mic_mg_L são informados, o vale >= MIC entra
    como critério adicional de adequação farmacodinâmica.

    Retorna:
    --------
    dicionário com avaliação de cada índice
    """

    config = ATB_CONFIG.get(atb)
    if config is None:
        raise ValueError(f"Antimicrobiano '{atb}' não encontrado no config.")

    alvo = config.get("alvo_fd", {})
    avaliacao = {}

    # AUC/MIC (vancomicina, daptomicina, linezolida)
    if "AUC/MIC" in alvo.get("indice", ""):
        auc_mic = resultado_fd.get("auc_mic")
        if auc_mic is not None:
            atingiu = alvo["alvo_minimo"] <= auc_mic <= alvo["alvo_maximo"]
            avaliacao["auc_mic"] = {
                "valor"         : auc_mic,
                "alvo"          : f"{alvo['alvo_minimo']} - {alvo['alvo_maximo']} {alvo['unidade']}",
                "atingiu_alvo"  : atingiu,
                "interpretacao" : "dentro do alvo" if atingiu else
                                  "subdose" if auc_mic < alvo["alvo_minimo"] else "sobredose",
            }

    # T>MIC (beta-lactâmicos)
    if "T>MIC" in str(alvo):
        alvo_beta = alvo if "indice" in alvo else alvo.get("ceftazidima", alvo)
        t_mic = resultado_fd.get("t_mic_percentual")
        vale = resultado_fd.get("vale_mg_L")
        mic = resultado_fd.get("mic_mg_L")

        if t_mic is not None:
            alvo_min = (
                alvo_beta.get("alvo_minimo_percentual")
                or alvo_beta.get("alvo_percentual")
                or 40
            )
            alvo_otimo = alvo_beta.get("alvo_otimo_percentual")
            atingiu_min = t_mic >= alvo_min
            atingiu_otimo = t_mic >= alvo_otimo if alvo_otimo is not None else None

            if not atingiu_min:
                interpretacao = "subdose"
            elif atingiu_otimo is False:
                interpretacao = "atingiu o alvo mínimo, abaixo do alvo ótimo"
            elif atingiu_otimo is True:
                interpretacao = "dentro do alvo ótimo"
            else:
                interpretacao = "dentro do alvo"

            alvo_txt = f"≥{alvo_min}% do intervalo"
            if alvo_otimo is not None and alvo_otimo != alvo_min:
                alvo_txt += f"; ótimo: {alvo_otimo}%"

            avaliacao["t_mic"] = {
                "valor"         : f"{t_mic}%",
                "alvo"          : alvo_txt,
                "atingiu_alvo"  : atingiu_min,
                "atingiu_otimo" : atingiu_otimo,
                "interpretacao" : interpretacao,
            }

        # Critério adicional para evitar falso "no alvo" quando o vale final
        # fica abaixo da MIC, apesar de o T>MIC mínimo ter sido atingido.
        if vale is not None and mic is not None:
            try:
                vale_num = float(vale)
                mic_num = float(mic)
                atingiu_vale = vale_num >= mic_num
                avaliacao["vale_mic"] = {
                    "valor"         : f"{round(vale_num, 2)} mg/L",
                    "alvo"          : f"≥ MIC ({round(mic_num, 2)} mg/L)",
                    "atingiu_alvo"  : atingiu_vale,
                    "interpretacao" : "vale acima da MIC" if atingiu_vale else "vale abaixo da MIC",
                }
            except Exception:
                pass

    return avaliacao


# =============================================================
# INFUSÃO CONTÍNUA
# =============================================================

def calcular_css_infusao_continua(
    dose_mg_dia: float,
    clearance_L_h: float,
) -> float:
    """
    Calcula a concentração no estado estacionário (Css)
    durante infusão contínua.

    Css = Taxa de infusão / Clearance
    Taxa de infusão = dose_mg_dia / 24h

    Parâmetros:
    -----------
    dose_mg_dia     : dose diária total em mg
    clearance_L_h   : clearance do fármaco em L/h

    Retorna:
    --------
    Css em mg/L
    """

    if clearance_L_h <= 0:
        raise ValueError("Clearance deve ser maior que zero.")

    taxa_mg_h = dose_mg_dia / 24
    css = taxa_mg_h / clearance_L_h

    return round(css, 2)


def gerar_curva_infusao_continua(
    dose_mg_dia: float,
    vd_L: float,
    clearance_L_h: float,
    duracao_total_h: float = 72,
    resolucao_h: float = 0.5,
) -> dict:
    """
    Gera a curva de concentração sérica durante infusão contínua.
    Modelo monocompartimental com infusão de taxa constante.

    Parâmetros:
    -----------
    dose_mg_dia     : dose diária total em mg
    vd_L            : volume de distribuição em litros
    clearance_L_h   : clearance total em L/h
    duracao_total_h : duração total da simulação em horas
    resolucao_h     : resolução temporal em horas

    Retorna:
    --------
    dicionário com:
        tempos_h        : lista de tempos em horas
        concentracoes   : lista de concentrações em mg/L
        css_mg_L        : concentração no estado estacionário
        tempo_css_h     : tempo para atingir 90% do Css
    """

    ke          = clearance_L_h / vd_L
    taxa_mg_h   = dose_mg_dia / 24
    css         = taxa_mg_h / clearance_L_h

    tempos = np.arange(0, duracao_total_h + resolucao_h, resolucao_h)
    concentracoes = css * (1 - np.exp(-ke * tempos))

    # Tempo para atingir 90% do Css
    tempo_css_90 = -np.log(0.1) / ke

    return {
        "tempos_h"      : list(tempos),
        "concentracoes" : list(concentracoes),
        "css_mg_L"      : round(float(css), 2),
        "tempo_css_h"   : round(float(tempo_css_90), 1),
        "pico_mg_L"     : round(float(css), 2),
        "vale_mg_L"     : round(float(css), 2),
    }


def avaliar_css_vs_mic(
    css_mg_L: float,
    mic_mg_L: float,
    multiplo_alvo: float = 4.0,
) -> dict:
    """
    Avalia se o Css atinge o alvo farmacodinâmico
    para infusão contínua de beta-lactâmicos.

    Alvo recomendado: Css ≥ 4-6x MIC (100% T>MIC)

    Parâmetros:
    -----------
    css_mg_L        : concentração no estado estacionário em mg/L
    mic_mg_L        : concentração inibitória mínima em mg/L
    multiplo_alvo   : múltiplo do MIC desejado (padrão 4x)

    Retorna:
    --------
    dicionário com avaliação
    """

    razao = round(css_mg_L / mic_mg_L, 1) if mic_mg_L > 0 else None
    alvo  = mic_mg_L * multiplo_alvo
    atingiu = css_mg_L >= alvo if razao is not None else None

    return {
        "css_mg_L"          : css_mg_L,
        "mic_mg_L"          : mic_mg_L,
        "razao_css_mic"     : razao,
        "alvo_mg_L"         : round(alvo, 2),
        "multiplo_alvo"     : multiplo_alvo,
        "atingiu_alvo"      : atingiu,
        "interpretacao"     : "dentro do alvo" if atingiu else "subdose" if atingiu is not None else "indefinido",
    }


# =============================================================
# AVALIAÇÃO DE TOXICIDADE
# =============================================================

def avaliar_toxicidade(
    atb: str,
    pico_mg_L: float = None,
    vale_mg_L: float = None,
    auc_mg_h_L: float = None,
) -> dict:
    """
    Avalia o risco de toxicidade conforme os limiares definidos
    no config.py para o antimicrobiano.

    Parâmetros:
    -----------
    atb         : chave do antimicrobiano no config.py
    pico_mg_L   : concentração máxima estimada
    vale_mg_L   : concentração mínima estimada antes da próxima dose
    auc_mg_h_L  : área sob a curva no último intervalo

    Retorna:
    --------
    dicionário com:
        nivel         : "seguro" / "atencao" / "risco"
        cor           : cor sugerida para o gráfico/alerta
        mensagem      : texto explicativo
        tipo          : tipo de toxicidade (nefro/neuro)
    """

    config = ATB_CONFIG.get(atb, {})
    tox    = config.get("toxicidade", {})

    if not tox:
        return {"nivel": "indefinido", "cor": "#9E9E9E",
                "mensagem": "Limiar de toxicidade não definido para este ATB.",
                "tipo": None}

    tipo        = tox.get("tipo", "toxicidade")
    limiar      = tox.get("concentracao_limiar_mg_L")
    alerta      = tox.get("concentracao_alerta_mg_L")
    auc_limiar  = tox.get("auc_limiar_mg_h_L")
    fonte       = tox.get("fonte", "")

    nivel    = "seguro"
    cor      = "#4CAF50"
    mensagem = f"Dentro da faixa segura ({tipo})."

    # Avaliação por concentração.
    # Para beta-lactâmicos, a segurança não deve depender apenas do pico,
    # porque o acúmulo em insuficiência renal aparece principalmente no vale.
    concentracoes_avaliadas = []
    if pico_mg_L is not None:
        concentracoes_avaliadas.append(("pico", pico_mg_L))
    if vale_mg_L is not None:
        concentracoes_avaliadas.append(("vale", vale_mg_L))

    if concentracoes_avaliadas and limiar is not None:
        marcador, maior_conc = max(concentracoes_avaliadas, key=lambda x: x[1])
        if alerta and maior_conc >= alerta:
            nivel = "risco"
            cor = "#D32F2F"
            mensagem = (f"RISCO ALTO de {tipo}: {marcador} {maior_conc} mg/L "
                        f"≥ limiar crítico de {alerta} mg/L. Fonte: {fonte}.")
        elif maior_conc >= limiar:
            nivel = "atencao"
            cor = "#FB8C00"
            mensagem = (f"ATENÇÃO — risco de {tipo}: {marcador} {maior_conc} mg/L "
                        f"≥ limiar de {limiar} mg/L. Fonte: {fonte}.")

    # Avaliação por AUC (vancomicina)
    if auc_mg_h_L is not None and auc_limiar is not None:
        if auc_mg_h_L >= auc_limiar:
            if nivel != "risco":
                nivel = "atencao" if nivel == "seguro" else nivel
                cor = "#FB8C00" if nivel == "atencao" else cor
            mensagem += (f" AUC {auc_mg_h_L} mg·h/L ≥ limiar de {auc_limiar} mg·h/L "
                         f"(risco de {tipo}).")

    return {
        "nivel"     : nivel,
        "cor"       : cor,
        "mensagem"  : mensagem,
        "tipo"      : tipo,
        "limiar_mg_L"      : limiar,
        "alerta_mg_L"      : alerta,
        "auc_limiar"       : auc_limiar,
        "pico_mg_L"        : pico_mg_L,
        "vale_mg_L"        : vale_mg_L,
    }

# =============================================================
# MODELOS ESTRUTURAIS BI E TRICOMPARTIMENTAL + SELEÇÃO POR AIC/BIC
# =============================================================
# A estrutura PK (mono/bi/tri) define a FORMA da curva. O monocompartimental
# já é a curva clássica (gerar_curva_concentracao). Aqui entram a bi e a tri,
# e a função que ajusta as três a pontos observados e escolhe a de menor ruído
# penalizado pelo número de parâmetros (AIC/BIC) — o critério da popPK.
# Requer scipy (já é dependência do scikit-learn).

from scipy.integrate import solve_ivp as _solve_ivp
from scipy.optimize import least_squares as _least_squares


def _simular_ncompartimental(parametros: dict, estrutura: str,
                             dose_mg, intervalo_h, tempo_infusao_h,
                             numero_doses, tempos_eval):
    """
    Simula a concentração central de um modelo de 1, 2 ou 3 compartimentos
    para um regime de infusões repetidas. Integra o sistema de EDOs.

    parametros (positivos):
      mono : CL, V1
      bi   : CL, V1, Q2, V2
      tri  : CL, V1, Q2, V2, Q3, V3
    """
    tempos_eval = np.asarray(tempos_eval, dtype=float)
    CL = parametros["CL"]; V1 = parametros["V1"]
    rin = dose_mg / tempo_infusao_h

    def taxa_entrada(t):
        r = 0.0
        for n in range(numero_doses):
            ini = n * intervalo_h
            if ini <= t < ini + tempo_infusao_h:
                r += rin
        return r

    k10 = CL / V1
    if estrutura == "mono":
        def ode(t, A):
            return [taxa_entrada(t) - k10 * A[0]]
        y0 = [0.0]
    elif estrutura == "bi":
        Q2 = parametros["Q2"]; V2 = parametros["V2"]
        k12 = Q2 / V1; k21 = Q2 / V2
        def ode(t, A):
            A1, A2 = A
            return [taxa_entrada(t) - (k10 + k12) * A1 + k21 * A2,
                    k12 * A1 - k21 * A2]
        y0 = [0.0, 0.0]
    elif estrutura == "tri":
        Q2 = parametros["Q2"]; V2 = parametros["V2"]
        Q3 = parametros["Q3"]; V3 = parametros["V3"]
        k12 = Q2 / V1; k21 = Q2 / V2
        k13 = Q3 / V1; k31 = Q3 / V3
        def ode(t, A):
            A1, A2, A3 = A
            return [taxa_entrada(t) - (k10 + k12 + k13) * A1 + k21 * A2 + k31 * A3,
                    k12 * A1 - k21 * A2,
                    k13 * A1 - k31 * A3]
        y0 = [0.0, 0.0, 0.0]
    else:
        raise ValueError(f"Estrutura desconhecida: {estrutura}")

    t_fim = float(max(tempos_eval[-1], 1e-6))
    sol = _solve_ivp(ode, (0.0, t_fim), y0, t_eval=tempos_eval,
                     method="LSODA", max_step=0.25, rtol=1e-6, atol=1e-8)
    C1 = np.maximum(sol.y[0] / V1, 0.0)
    return C1


_PARAMS_ESTRUTURA = {
    "mono": ["CL", "V1"],
    "bi":   ["CL", "V1", "Q2", "V2"],
    "tri":  ["CL", "V1", "Q2", "V2", "Q3", "V3"],
}


def gerar_curva_bicompartimental(dose_mg, V1_L, CL_L_h, Q2_L_h, V2_L,
                                 intervalo_h, tempo_infusao_h=0.5,
                                 numero_doses=5, resolucao_h=0.5):
    """Curva de 2 compartimentos no mesmo formato de gerar_curva_concentracao."""
    tempos = np.arange(0, numero_doses * intervalo_h + resolucao_h, resolucao_h)
    parametros = {"CL": CL_L_h, "V1": V1_L, "Q2": Q2_L_h, "V2": V2_L}
    concs = _simular_ncompartimental(parametros, "bi", dose_mg, intervalo_h,
                                     tempo_infusao_h, numero_doses, tempos)
    return {
        "tempos_h"     : list(tempos),
        "concentracoes": [round(float(c), 2) for c in concs],
        "pico_mg_L"    : round(float(np.max(concs)), 2),
        "vale_mg_L"    : round(float(concs[-1]), 2),
        "tempo_pico_h" : round(float(tempos[np.argmax(concs)]), 1),
        "estrutura"    : "bi",
    }


def gerar_curva_tricompartimental(dose_mg, V1_L, CL_L_h, Q2_L_h, V2_L, Q3_L_h, V3_L,
                                  intervalo_h, tempo_infusao_h=0.5,
                                  numero_doses=5, resolucao_h=0.5):
    """Curva de 3 compartimentos no mesmo formato de gerar_curva_concentracao."""
    tempos = np.arange(0, numero_doses * intervalo_h + resolucao_h, resolucao_h)
    parametros = {"CL": CL_L_h, "V1": V1_L, "Q2": Q2_L_h, "V2": V2_L, "Q3": Q3_L_h, "V3": V3_L}
    concs = _simular_ncompartimental(parametros, "tri", dose_mg, intervalo_h,
                                     tempo_infusao_h, numero_doses, tempos)
    return {
        "tempos_h"     : list(tempos),
        "concentracoes": [round(float(c), 2) for c in concs],
        "pico_mg_L"    : round(float(np.max(concs)), 2),
        "vale_mg_L"    : round(float(concs[-1]), 2),
        "tempo_pico_h" : round(float(tempos[np.argmax(concs)]), 1),
        "estrutura"    : "tri",
    }


def _ajustar_estrutura(estrutura, tempos_obs, concs_obs,
                       dose_mg, intervalo_h, tempo_infusao_h, numero_doses,
                       chute):
    """Ajusta uma estrutura aos pontos observados por mínimos quadrados.
    Otimiza no log dos parâmetros para garantir positividade."""
    nomes = _PARAMS_ESTRUTURA[estrutura]
    p0 = np.log(np.array([max(chute[n], 1e-3) for n in nomes], dtype=float))

    def residuo(log_p):
        params = {n: float(np.exp(v)) for n, v in zip(nomes, log_p)}
        cpred = _simular_ncompartimental(params, estrutura, dose_mg, intervalo_h,
                                         tempo_infusao_h, numero_doses, tempos_obs)
        return cpred - concs_obs

    sol = _least_squares(residuo, p0, method="lm", max_nfev=4000)
    params = {n: float(np.exp(v)) for n, v in zip(nomes, sol.x)}
    rss = float(np.sum(sol.fun ** 2))
    return params, rss


def selecionar_estrutura_aic_bic(tempos_obs, concs_obs,
                                 dose_mg, intervalo_h, tempo_infusao_h,
                                 numero_doses,
                                 CL_chute, V_chute):
    """
    Ajusta mono, bi e tri aos pontos observados e escolhe a estrutura de
    menor ruído PENALIZADO pelo número de parâmetros (AIC/BIC).

    Critério (literatura popPK):
      AIC = n*ln(RSS/n) + 2k
      BIC = n*ln(RSS/n) + k*ln(n)
    onde n = nº de pontos, k = nº de parâmetros, RSS = soma dos resíduos².
    Menor AIC/BIC = melhor ajuste descontada a complexidade.
    """
    tempos_obs = np.asarray(tempos_obs, dtype=float)
    concs_obs = np.asarray(concs_obs, dtype=float)
    n = len(concs_obs)

    chute = {"CL": CL_chute, "V1": V_chute, "Q2": CL_chute, "V2": V_chute,
             "Q3": CL_chute * 0.5, "V3": V_chute * 0.5}

    resultados = []
    for estrutura in ["mono", "bi", "tri"]:
        try:
            params, rss = _ajustar_estrutura(
                estrutura, tempos_obs, concs_obs,
                dose_mg, intervalo_h, tempo_infusao_h, numero_doses, chute)
            k = len(_PARAMS_ESTRUTURA[estrutura])
            rss = max(rss, 1e-9)
            rmse = float(np.sqrt(rss / n))
            aic = n * np.log(rss / n) + 2 * k
            bic = n * np.log(rss / n) + k * np.log(n)
            resultados.append({
                "estrutura"    : estrutura,
                "n_parametros" : k,
                "rmse_residual": round(rmse, 4),
                "aic"          : round(float(aic), 2),
                "bic"          : round(float(bic), 2),
                "parametros"   : {kk: round(vv, 4) for kk, vv in params.items()},
                "ajustou"      : True,
            })
        except Exception as e:
            resultados.append({
                "estrutura": estrutura, "ajustou": False, "erro": str(e)[:120],
                "aic": float("inf"), "bic": float("inf"),
            })

    validos = [r for r in resultados if r.get("ajustou")]
    melhor = min(validos, key=lambda r: r["aic"])["estrutura"] if validos else None
    return {"tabela": resultados, "melhor_estrutura": melhor, "criterio": "menor AIC"}


# =============================================================
# CALIBRAÇÃO INDIVIDUAL COM DOSEAMENTO SÉRICO REAL
# =============================================================
# Quando o farmacêutico dispõe de uma concentração sérica medida,
# é possível estimar os parâmetros individuais (CL e Vd) por ajuste
# da curva monocompartimental ao ponto real. Essa estimativa é usada
# para redesenhar a curva e recalcular os índices PK/PD com parâmetros
# do paciente, não populacionais.
#
# Com apenas um ponto de concentração, o sistema de equações é
# subdeterminado (2 parâmetros, 1 equação). A solução adotada é:
#   - Manter o Vd populacional como âncora (menos sensível a erros)
#   - Estimar apenas o CL individual que faz a curva passar pelo ponto
# Isso é equivalente ao método de substituição direta usado em TDM
# convencional, porém usando a equação completa de múltiplas doses
# em vez da equação de dose única.
#
# Com dois ou mais pontos, o ajuste de CL e Vd é feito por mínimos
# quadrados, aumentando a confiabilidade.

def calibrar_com_doseamento(
    conc_medida_mg_L: float,
    tempo_coleta_h: float,
    dose_mg: float,
    intervalo_h: float,
    tempo_infusao_h: float,
    numero_doses: int,
    vd_inicial_L: float,
    cl_inicial_L_h: float,
    pontos_extras: list = None,
    resolucao_h: float = 0.25,
) -> dict:
    """
    Calibra CL (e Vd quando há mais de um ponto) para que a curva
    monocompartimental passe pelo(s) ponto(s) medido(s).

    pontos_extras: lista de dicts [{"tempo_h": float, "conc_mg_L": float}]
    para ajuste com múltiplos pontos.

    Retorna dict com:
        cl_individual_L_h   : CL estimado
        vd_individual_L     : Vd usado (estimado ou populacional)
        curva               : dict igual ao retorno de gerar_curva_concentracao
        erro_residual_mg_L  : |predito - medido| no ponto de calibração
        metodo              : "1-ponto" ou "n-pontos"
    """
    from scipy.optimize import minimize_scalar, minimize

    intervalo_h    = max(intervalo_h, 0.1)
    tempo_infusao_h = max(tempo_infusao_h, 0.05)
    numero_doses    = max(numero_doses, 1)

    # Monta todos os pontos observados
    tempos_obs = [tempo_coleta_h]
    concs_obs  = [conc_medida_mg_L]
    if pontos_extras:
        for pt in pontos_extras:
            tempos_obs.append(float(pt["tempo_h"]))
            concs_obs.append(float(pt["conc_mg_L"]))
    tempos_obs = np.array(tempos_obs, dtype=float)
    concs_obs  = np.array(concs_obs,  dtype=float)

    def _curva_em_tempos(cl, vd, tempos):
        ke = max(cl / vd, 1e-6)
        rin = dose_mg / tempo_infusao_h
        concs = np.zeros_like(tempos)
        for n in range(numero_doses):
            ini = n * intervalo_h
            tr  = tempos - ini
            m   = tr >= 0
            dur = m & (tr <= tempo_infusao_h)
            pos = m & (tr >  tempo_infusao_h)
            concs[dur] += (rin / cl) * (1 - np.exp(-ke * tr[dur]))
            c_fim = (rin / cl) * (1 - np.exp(-ke * tempo_infusao_h))
            concs[pos] += c_fim * np.exp(-ke * (tr[pos] - tempo_infusao_h))
        return np.maximum(concs, 0.0)

    if len(tempos_obs) == 1:
        # 1 ponto: estima só CL mantendo Vd populacional
        vd_cal = vd_inicial_L

        def residuo_cl(log_cl):
            cl = float(np.exp(log_cl))
            pred = _curva_em_tempos(cl, vd_cal, tempos_obs)
            return float((pred[0] - concs_obs[0]) ** 2)

        res = minimize_scalar(residuo_cl,
                              bounds=(np.log(0.1), np.log(cl_inicial_L_h * 10)),
                              method="bounded")
        cl_cal = float(np.exp(res.x))
        metodo = "1-ponto (Vd fixo populacional)"
    else:
        # N pontos: estima CL e Vd
        def residuo_2p(log_params):
            cl  = float(np.exp(log_params[0]))
            vd  = float(np.exp(log_params[1]))
            pred = _curva_em_tempos(cl, vd, tempos_obs)
            return float(np.sum((pred - concs_obs) ** 2))

        x0  = [np.log(cl_inicial_L_h), np.log(vd_inicial_L)]
        res = minimize(residuo_2p, x0, method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-8})
        cl_cal = float(np.exp(res.x[0]))
        vd_cal = float(np.exp(res.x[1]))
        metodo = f"{len(tempos_obs)}-pontos (CL e Vd estimados)"

    cl_cal = max(cl_cal, 0.05)
    vd_cal = max(vd_cal, 0.5)

    # Gera curva completa com parâmetros individuais
    curva = gerar_curva_concentracao(
        dose_mg         = dose_mg,
        vd_L            = vd_cal,
        clearance_L_h   = cl_cal,
        intervalo_h     = intervalo_h,
        tempo_infusao_h = tempo_infusao_h,
        numero_doses    = numero_doses,
        resolucao_h     = resolucao_h,
    )

    # Erro residual no ponto de calibração
    pred_cal = _curva_em_tempos(cl_cal, vd_cal, np.array([tempo_coleta_h]))[0]
    erro     = round(abs(pred_cal - conc_medida_mg_L), 3)

    return {
        "cl_individual_L_h"  : round(cl_cal, 4),
        "vd_individual_L"    : round(vd_cal, 2),
        "curva"              : curva,
        "erro_residual_mg_L" : erro,
        "metodo"             : metodo,
        "tempos_obs"         : list(tempos_obs),
        "concs_obs"          : list(concs_obs),
    }