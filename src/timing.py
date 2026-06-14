# =============================================================
# ATBSPACE - Módulo de Timing
# =============================================================
# Responsável por:
# → Calcular horário seguro da primeira dose do novo regime
# → Detectar doses perdidas
# → Detectar intervalo inseguro entre regimes
# → Gerar alertas de subdose e sobredose por timing
# =============================================================

from datetime import datetime, timedelta


# -------------------------------------------------------------
# FUNÇÃO PRINCIPAL
# -------------------------------------------------------------

def calcular_timing(
    horario_ultima_dose: str,
    dose_foi_administrada: bool,
    intervalo_anterior_h: float,
    novo_intervalo_h: float,
    horario_dose_perdida: str = None,
):
    """
    Calcula o horário seguro para iniciar o novo regime.

    Parâmetros:
    -----------
    horario_ultima_dose       : horário da última dose administrada (formato "HH:MM")
    dose_foi_administrada     : True se a dose foi dada, False se foi perdida
    intervalo_anterior_h      : intervalo do regime anterior em horas
    novo_intervalo_h          : intervalo do novo regime em horas
    horario_dose_perdida      : horário da dose que NÃO foi administrada (formato "HH:MM")
                                obrigatório se dose_foi_administrada=False

    Retorna:
    --------
    dicionário com:
        horario_proxima_dose  : horário seguro para primeira dose do novo regime
        intervalo_real_h      : intervalo real entre última dose e nova dose
        seguro                : True se o intervalo é seguro
        alerta                : mensagem de alerta se houver problema
        detalhes              : explicação do cálculo
    """

    formato = "%H:%M"
    hoje = datetime.today().replace(second=0, microsecond=0)

    ultima = datetime.strptime(horario_ultima_dose, formato).replace(
        year=hoje.year, month=hoje.month, day=hoje.day
    )

    # ---------------------------------------------------------
    # CENÁRIO 1: dose foi administrada normalmente
    # ---------------------------------------------------------
    if dose_foi_administrada:
        proxima_dose = ultima + timedelta(hours=intervalo_anterior_h)
        intervalo_real = intervalo_anterior_h
        detalhes = (
            f"Última dose administrada às {horario_ultima_dose}. "
            f"Respeitando o intervalo anterior de {intervalo_anterior_h}h, "
            f"a primeira dose do novo regime deve ser às {proxima_dose.strftime(formato)}."
        )

    # ---------------------------------------------------------
    # CENÁRIO 2: dose NÃO foi administrada (dose perdida)
    # ---------------------------------------------------------
    else:
        if horario_dose_perdida is None:
            raise ValueError(
                "Informe o horário da dose perdida quando dose_foi_administrada=False."
            )

        dose_perdida = datetime.strptime(horario_dose_perdida, formato).replace(
            year=hoje.year, month=hoje.month, day=hoje.day
        )

        # Referência: última dose REAL administrada
        intervalo_desde_ultima = (dose_perdida - ultima).total_seconds() / 3600
        proxima_dose = ultima + timedelta(hours=intervalo_anterior_h)
        intervalo_real = intervalo_anterior_h

        detalhes = (
            f"Última dose REAL administrada às {horario_ultima_dose}. "
            f"Dose das {horario_dose_perdida} NÃO foi administrada. "
            f"O nível sérico está em queda desde {horario_ultima_dose}. "
            f"Primeira dose do novo regime às {proxima_dose.strftime(formato)} "
            f"(referência: última dose real + {intervalo_anterior_h}h)."
        )

    # ---------------------------------------------------------
    # VERIFICAÇÃO DE SEGURANÇA
    # ---------------------------------------------------------
    intervalo_entre_regimes = (proxima_dose - ultima).total_seconds() / 3600
    seguro = intervalo_entre_regimes >= intervalo_anterior_h

    alerta = None

    if not seguro:
        alerta = (
            f"ALERTA: Intervalo entre regimes de {intervalo_entre_regimes:.1f}h "
            f"é inferior ao intervalo anterior de {intervalo_anterior_h}h. "
            f"Risco de acúmulo e sobredose."
        )
    elif intervalo_entre_regimes > novo_intervalo_h * 1.5:
        alerta = (
            f"ATENÇÃO: Intervalo entre regimes de {intervalo_entre_regimes:.1f}h "
            f"é superior a 1,5x o novo intervalo de {novo_intervalo_h}h. "
            f"Risco de subdose e perda de cobertura."
        )

    return {
        "horario_ultima_dose"       : horario_ultima_dose,
        "dose_foi_administrada"     : dose_foi_administrada,
        "intervalo_anterior_h"      : intervalo_anterior_h,
        "novo_intervalo_h"          : novo_intervalo_h,
        "horario_proxima_dose"      : proxima_dose.strftime(formato),
        "intervalo_entre_regimes_h" : round(intervalo_entre_regimes, 1),
        "seguro"                    : seguro,
        "alerta"                    : alerta,
        "detalhes"                  : detalhes,
    }


# -------------------------------------------------------------
# FUNÇÃO DE HISTÓRICO DE DOSES
# -------------------------------------------------------------

def analisar_historico(historico: list) -> dict:
    """
    Analisa o histórico completo de administração de um paciente.

    Parâmetros:
    -----------
    historico : lista de dicionários com:
        [
            {"horario": "08:00", "dose_mg": 1250, "administrada": True},
            {"horario": "16:00", "dose_mg": 1250, "administrada": False},
            {"horario": "00:00", "dose_mg": 940,  "administrada": True},
        ]

    Retorna:
    --------
    dicionário com:
        doses_perdidas         : lista de horários com doses não administradas
        intervalos_reais_h     : intervalos reais entre doses administradas
        menor_intervalo_h      : menor intervalo detectado
        alertas                : lista de alertas identificados
    """

    doses_administradas = [d for d in historico if d["administrada"]]
    doses_perdidas = [d["horario"] for d in historico if not d["administrada"]]

    formato = "%H:%M"
    hoje = datetime.today()

    intervalos = []
    alertas = []

    for i in range(1, len(doses_administradas)):
        t1 = datetime.strptime(doses_administradas[i - 1]["horario"], formato).replace(
            year=hoje.year, month=hoje.month, day=hoje.day
        )
        t2 = datetime.strptime(doses_administradas[i]["horario"], formato).replace(
            year=hoje.year, month=hoje.month, day=hoje.day
        )

        # Ajuste para virada de meia-noite
        if t2 < t1:
            t2 += timedelta(days=1)

        intervalo = (t2 - t1).total_seconds() / 3600
        intervalos.append(round(intervalo, 1))

    if intervalos:
        menor = min(intervalos)
        if menor < 6:
            alertas.append(
                f"Intervalo mínimo de {menor}h detectado entre doses administradas. "
                f"Risco elevado de acúmulo."
            )

    if doses_perdidas:
        alertas.append(
            f"{len(doses_perdidas)} dose(s) não administrada(s): {', '.join(doses_perdidas)}. "
            f"Possível perda de cobertura terapêutica."
        )

    return {
        "total_doses_previstas"     : len(historico),
        "total_doses_administradas" : len(doses_administradas),
        "doses_perdidas"            : doses_perdidas,
        "intervalos_reais_h"        : intervalos,
        "menor_intervalo_h"         : min(intervalos) if intervalos else None,
        "alertas"                   : alertas,
    }
