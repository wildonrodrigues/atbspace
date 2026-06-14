# =============================================================
# ATBSPACE - Interface Streamlit streamlit run app/app.py
# =============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from config import ATB_CONFIG, MODELOS_DISPONIVEIS
from pk_engine import (
    calcular_clcr_cockcroft_gault,
    calcular_peso_ideal,
    calcular_peso_ajustado,
    selecionar_ajuste_renal,
    gerar_curva_concentracao,
    gerar_curva_infusao_continua,
    avaliar_css_vs_mic,
    calcular_auc,
    calcular_t_mic,
    calcular_auc_mic,
    calcular_pico_mic,
    avaliar_alvo_fd,
    avaliar_toxicidade,
)
from timing import calcular_timing, analisar_historico
from scenarios import criar_cenario, simular_cenario, comparar_cenarios, incorporar_doseamento_real
from plots import (
    plot_curva,
    plot_comparacao_modelos,
    plot_comparacao_cenarios,
    plot_predito_vs_observado,
)

# =============================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================

st.set_page_config(
    page_title  = "ATBSPACE",
    page_icon   = "💊",
    layout      = "wide",
)

st.title("💊 ATBSPACE")
st.caption("Calculadora farmacocinética para antimicrobianos intravenosos")

# =============================================================
# SIDEBAR — DADOS DO PACIENTE
# =============================================================

with st.sidebar:
    st.header("Dados do Paciente")

    idade       = st.number_input("Idade (anos)", min_value=18, max_value=100, value=55)
    peso_real   = st.number_input("Peso real (kg)", min_value=30.0, max_value=200.0, value=72.0)
    altura      = st.number_input("Altura (cm)", min_value=100, max_value=220, value=170)
    sexo        = st.selectbox("Sexo", ["masculino", "feminino"])
    creatinina  = st.number_input("Creatinina sérica (mg/dL)", min_value=0.1, max_value=20.0, value=1.2, step=0.1)

    trs = st.selectbox(
        "Terapia renal substitutiva",
        ["nao", "hemofiltração continua", "hemodiálise continua", "hemodiálise", "diálise peritoneal"]
    )
    modalidade_trs = None if trs == "nao" else trs

    st.divider()
    st.header("Antimicrobiano")
    atb_selecionado = st.selectbox("Selecione", list(ATB_CONFIG.keys()))
    mic_mg_L = st.number_input("MIC (mg/L)", min_value=0.1, max_value=64.0, value=1.0, step=0.1)

    st.divider()
    st.header("Modelo")
    estrutura_sel = st.selectbox(
        "Estrutura PK",
        ["mono", "bi", "tri"],
        help="Forma da curva-verdade usada no treino dos modelos. mono = curva clássica (antigo pkpd).",
    )
    modelo_selecionado = st.selectbox(
        "Modelo preditivo",
        ["pkpd"] + [m for m in MODELOS_DISPONIVEIS if m != "pkpd"],
        help="pkpd = mostrar só a curva de referência. Os demais usam o modelo treinado na estrutura escolhida.",
    )
    comparar_todos = st.checkbox("Comparar todos os modelos")

# =============================================================
# CÁLCULOS BASE
# =============================================================

peso_ideal    = calcular_peso_ideal(altura, sexo)
peso_ajustado = calcular_peso_ajustado(peso_real, peso_ideal)
peso_calc     = peso_ajustado

config_atb  = ATB_CONFIG[atb_selecionado]
pk          = config_atb.get("pk_populacional", {})
fator_corr  = pk.get("fator_correcao_clcr", 1.0)

clcr = calcular_clcr_cockcroft_gault(
    idade=idade,
    peso_kg=peso_calc,
    sexo=sexo,
    creatinina_mg_dL=creatinina,
    fator_correcao=fator_corr,
)

ajuste      = selecionar_ajuste_renal(atb_selecionado, clcr, modalidade_trs)
dose_mg     = ajuste.get("dose_mg") or (ajuste.get("dose_mg_kg", 15.0) * peso_calc)
intervalo_h = ajuste["intervalo_h"]
admin       = config_atb.get("administracao", {})
tempo_inf   = admin.get("tempo_infusao_h", 1.0)
vd_L_kg     = pk.get("vd_L_kg", 0.82)
vd_total    = vd_L_kg * peso_calc

if pk.get("equacao_clearance") and "3.66" in str(pk.get("equacao_clearance", "")):
    # Vancomicina — equacao de Matzke (proporcional ao ClCr)
    cl_L_h = (3.66 + 0.689 * clcr) / 1000 * 60
elif pk.get("clearance_L_h"):
    # Meropenem, CAZ-AVI — clearance proporcional ao ClCr do paciente
    pop      = config_atb.get("populacao_padrao", {})
    clcr_ref = pop.get("clcr_medio_mL_min", 60.0)
    cl_ref   = pk.get("clearance_L_h")
    cl_L_h   = cl_ref * (clcr / clcr_ref)
    cl_L_h   = round(max(cl_L_h, 0.1), 3)
elif pk.get("meia_vida_h"):
    cl_L_h = vd_total * 0.693 / pk["meia_vida_h"]
else:
    cl_L_h = vd_total * 0.693 / 6.0

# =============================================================
# RECOMENDAÇÃO RENAL E PRESCRIÇÃO ATUAL
# =============================================================
# dose_recomendada_mg e intervalo_recomendado_h são apenas referência.
# A curva, a toxicidade e os cenários usam a prescrição atual informada
# pelo usuário. Esses campos não devem ser sobrescritos quando a creatinina
# muda, senão o aplicativo esconde sobredose por ajuste automático.

dose_recomendada_mg       = dose_mg
intervalo_recomendado_h   = intervalo_h
tempo_inf_recomendado_h   = tempo_inf

if st.session_state.get("_prescricao_atb") != atb_selecionado:
    st.session_state["_prescricao_atb"] = atb_selecionado
    st.session_state["dose_prescrita_mg"] = float(round(dose_recomendada_mg))
    st.session_state["intervalo_prescrito_h"] = float(intervalo_recomendado_h)
    st.session_state["tempo_infusao_prescrito_h"] = float(tempo_inf_recomendado_h)

# Valores usados antes da renderização dos widgets da aba 1.
# Depois dos widgets, essas variáveis são atualizadas novamente.
dose_mg     = st.session_state.get("dose_prescrita_mg", float(round(dose_recomendada_mg)))
intervalo_h = st.session_state.get("intervalo_prescrito_h", float(intervalo_recomendado_h))
tempo_inf   = st.session_state.get("tempo_infusao_prescrito_h", float(tempo_inf_recomendado_h))


def montar_dados_base_modelo(dose_base_mg: float, intervalo_base_h: float, tempo_infusao_base_h: float) -> dict:
    """Monta as features usadas pelos modelos treinados para o paciente atual."""
    return {
        "idade"               : idade,
        "peso_kg"             : peso_calc,
        "sexo_num"            : 1 if sexo == "masculino" else 0,
        "creat_basal"         : creatinina,
        "creat_atual"         : creatinina,
        "vd_L_kg"             : vd_L_kg,
        "vd_total_L"          : vd_total,
        "dose_mg_kg"          : dose_base_mg / peso_calc if peso_calc else 0,
        "dose_mg"             : dose_base_mg,
        "intervalo_h"         : intervalo_base_h,
        "tempo_infusao_h"     : tempo_infusao_base_h,
        "clearance_atual_L_h" : cl_L_h,
    }


def gerar_curva_por_modelo(nome_modelo: str, dose_base_mg: float, intervalo_base_h: float,
                           tempo_infusao_base_h: float, numero_doses: int,
                           estrutura: str = "mono") -> dict:
    """Gera a curva pelo modelo selecionado: curva PK de referência ou modelo treinado."""
    if nome_modelo == "pkpd":
        curva_pk = gerar_curva_concentracao(
            dose_mg         = dose_base_mg,
            vd_L            = vd_total,
            clearance_L_h   = cl_L_h,
            intervalo_h     = intervalo_base_h,
            tempo_infusao_h = tempo_infusao_base_h,
            numero_doses    = numero_doses,
        )
        curva_pk["modelo"] = "pkpd"
        return curva_pk

    from models import predizer_curva_modelo
    return predizer_curva_modelo(
        atb             = atb_selecionado,
        nome_modelo     = nome_modelo,
        dados_base      = montar_dados_base_modelo(dose_base_mg, intervalo_base_h, tempo_infusao_base_h),
        duracao_total_h = numero_doses * intervalo_base_h,
        resolucao_h     = 0.5,
        estrutura       = estrutura,
    )

# =============================================================
# ABAS PRINCIPAIS
# =============================================================

aba1, aba2, aba3, aba4, aba5 = st.tabs([
    "📋 Dose e Timing",
    "📈 Curva Preditiva",
    "🔁 Comparação de Cenários",
    "🔬 Comparação de Modelos",
    "📄 Relatório",
])

# =============================================================
# ABA 1 — DOSE E TIMING
# =============================================================

with aba1:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("ClCr estimado", f"{clcr} mL/min")
        st.metric("Peso utilizado", f"{peso_calc} kg")
        st.caption(f"Peso ideal: {peso_ideal} kg | Peso ajustado: {peso_ajustado} kg")

    with col2:
        st.metric("Dose recomendada", f"{round(dose_recomendada_mg, 0)} mg")
        st.metric("Intervalo recomendado", f"q{int(intervalo_recomendado_h)}h")
        st.caption(f"Fonte: {ajuste['fonte']}")

    with col3:
        st.metric("Tempo de infusão recomendado", f"{tempo_inf_recomendado_h}h")
        st.metric("Volume de distribuição", f"{round(vd_total, 1)} L")

    # ----------------------------------------------------------
    # DADOS DA PRESCRIÇÃO ATUAL
    # ----------------------------------------------------------
    st.divider()
    st.subheader("📝 Dados da prescrição atual")
    st.caption(
        "Esses campos alimentam a curva preditiva, a toxicidade e os cenários. "
        "Eles não são alterados automaticamente quando a creatinina muda."
    )

    if st.button("Copiar recomendação renal para a prescrição atual"):
        st.session_state["dose_prescrita_mg"] = float(round(dose_recomendada_mg))
        st.session_state["intervalo_prescrito_h"] = float(intervalo_recomendado_h)
        st.session_state["tempo_infusao_prescrito_h"] = float(tempo_inf_recomendado_h)
        st.rerun()

    col_p1, col_p2, col_p3 = st.columns(3)
    dose_sim_mg = col_p1.number_input(
        "Dose prescrita atual (mg)",
        min_value=0.0,
        step=100.0,
        key="dose_prescrita_mg",
    )
    intervalo_sim_h = col_p2.number_input(
        "Intervalo prescrito atual (h)",
        min_value=1.0,
        max_value=48.0,
        step=1.0,
        key="intervalo_prescrito_h",
    )
    tempo_inf_sim_h = col_p3.number_input(
        "Tempo de infusão prescrito (h)",
        min_value=0.1,
        max_value=24.0,
        step=0.5,
        key="tempo_infusao_prescrito_h",
    )

    dose_mg     = dose_sim_mg
    intervalo_h = intervalo_sim_h
    tempo_inf   = tempo_inf_sim_h

    st.info(
        f"Regime usado nas simulações: {round(dose_mg, 0)} mg "
        f"q{int(intervalo_h)}h, infusão de {tempo_inf}h."
    )

    # ----------------------------------------------------------
    # TAXA DE INFUSÃO
    # ----------------------------------------------------------
    st.divider()
    st.subheader("💉 Taxa de Infusão")
    st.caption("Cálculo operacional da administração a partir da dose prescrita atual.")

    col_inf1, col_inf2, col_inf3, col_inf4 = st.columns(4)

    with col_inf1:
        dose_inf_mg = dose_mg
        st.metric("Dose usada no cálculo", f"{round(dose_inf_mg, 0)} mg")

    with col_inf2:
        tem_volume = st.checkbox("Volume final disponível")
        volume_mL  = None
        if tem_volume:
            volume_mL = st.number_input("Volume final (mL)", min_value=1.0, value=100.0, key="vol_inf")

    with col_inf3:
        tem_taxa = st.checkbox("Taxa de infusão disponível")
        taxa_unidade = st.selectbox("Unidade", ["mL/h", "mL/min"], key="taxa_unid") if tem_taxa else "mL/h"
        taxa_val = None
        if tem_taxa:
            taxa_val = st.number_input(f"Taxa ({taxa_unidade})", min_value=0.1, value=50.0, key="taxa_inf")

    with col_inf4:
        tem_tempo_inf = st.checkbox("Tempo de infusão disponível")
        tempo_unidade = st.selectbox("Unidade ", ["horas", "minutos"], key="tempo_unid") if tem_tempo_inf else "horas"
        tempo_inf_val = None
        if tem_tempo_inf:
            tempo_inf_val = st.number_input(f"Tempo ({tempo_unidade})", min_value=0.1, value=float(tempo_inf), key="tempo_inf_val")

    if dose_inf_mg > 0:
        if taxa_val and taxa_unidade == "mL/min":
            taxa_val_h = taxa_val * 60
        else:
            taxa_val_h = taxa_val

        if tempo_inf_val and tempo_unidade == "minutos":
            tempo_inf_h = tempo_inf_val / 60
        else:
            tempo_inf_h = tempo_inf_val

        resultado_inf = {}

        if volume_mL and taxa_val_h and not tempo_inf_h:
            tempo_inf_h = volume_mL / taxa_val_h
            resultado_inf["Tempo de infusão calculado"] = f"{round(tempo_inf_h, 2)}h ({round(tempo_inf_h*60,0)} min)"

        elif volume_mL and tempo_inf_h and not taxa_val_h:
            taxa_val_h = volume_mL / tempo_inf_h
            resultado_inf["Taxa calculada"] = f"{round(taxa_val_h, 1)} mL/h ({round(taxa_val_h/60, 2)} mL/min)"

        elif taxa_val_h and tempo_inf_h and not volume_mL:
            volume_mL = taxa_val_h * tempo_inf_h
            resultado_inf["Volume calculado"] = f"{round(volume_mL, 1)} mL"

        if volume_mL and dose_inf_mg:
            conc_solucao = dose_inf_mg / volume_mL
            resultado_inf["Concentração da solução"] = f"{round(conc_solucao, 2)} mg/mL"

        # Expõe o tempo de infusão real calculado para as outras abas
        if tempo_inf_h:
            st.session_state["tempo_inf_real_h"] = round(float(tempo_inf_h), 4)
        else:
            st.session_state.pop("tempo_inf_real_h", None)

        if resultado_inf:
            for campo, valor in resultado_inf.items():
                st.info(f"**{campo}:** {valor}")

    # ----------------------------------------------------------
    # MÓDULO DE TIMING
    # ----------------------------------------------------------
    st.divider()
    st.subheader("⏱ Módulo de Timing")

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        horario_ultima    = st.text_input("Horário da última dose (HH:MM)", value="08:00")
        dose_administrada = st.radio("Dose foi administrada?", ["sim", "não"]) == "sim"
        intervalo_anterior = st.number_input("Intervalo anterior (horas)", min_value=1.0, max_value=48.0, value=8.0)
        novo_intervalo    = st.number_input("Novo intervalo (horas)", min_value=1.0, max_value=48.0, value=float(intervalo_h))

    with col_t2:
        horario_perdida = None
        if not dose_administrada:
            horario_perdida = st.text_input("Horário da dose NÃO administrada (HH:MM)", value="16:00")

        if st.button("Calcular timing"):
            try:
                resultado_timing = calcular_timing(
                    horario_ultima_dose    = horario_ultima,
                    dose_foi_administrada  = dose_administrada,
                    intervalo_anterior_h   = intervalo_anterior,
                    novo_intervalo_h       = novo_intervalo,
                    horario_dose_perdida   = horario_perdida,
                )
                st.success(f"Primeira dose do novo regime: **{resultado_timing['horario_proxima_dose']}**")
                st.info(f"Intervalo entre regimes: {resultado_timing['intervalo_entre_regimes_h']}h")
                if resultado_timing["alerta"]:
                    st.warning(resultado_timing["alerta"])
                st.caption(resultado_timing["detalhes"])
            except Exception as e:
                st.error(str(e))

    # ----------------------------------------------------------
    # AUDITORIA DE HISTÓRICO
    # ----------------------------------------------------------
    st.divider()
    st.subheader("📋 Auditoria de Histórico")

    n_doses  = st.number_input("Número de doses para auditar", min_value=1, max_value=20, value=3)
    historico = []

    for i in range(int(n_doses)):
        cols    = st.columns(3)
        horario = cols[0].text_input(f"Horário dose {i+1}", value="08:00", key=f"h_{i}")
        dose_val = cols[1].number_input(f"Dose {i+1} (mg)", value=float(round(dose_mg)), key=f"d_{i}")
        adm     = cols[2].selectbox(f"Administrada?", ["sim", "não"], key=f"a_{i}") == "sim"
        historico.append({"horario": horario, "dose_mg": dose_val, "administrada": adm})

    if st.button("Auditar histórico"):
        resultado_hist = analisar_historico(historico)

        col_a1, col_a2 = st.columns(2)
        col_a1.metric("Doses previstas", resultado_hist["total_doses_previstas"])
        col_a2.metric("Doses administradas", resultado_hist["total_doses_administradas"])

        st.divider()
        st.markdown("**Linha do tempo:**")
        for i, dose in enumerate(historico):
            status = "✅ administrada" if dose["administrada"] else "❌ NÃO administrada"
            intervalo_txt = ""
            if i > 0 and resultado_hist["intervalos_reais_h"]:
                idx_intervalo = i - 1
                if idx_intervalo < len(resultado_hist["intervalos_reais_h"]):
                    intervalo_real = resultado_hist["intervalos_reais_h"][idx_intervalo]
                    intervalo_txt = f" | intervalo desde dose anterior: **{intervalo_real}h**"
            st.markdown(f"- **{dose['horario']}** — {dose['dose_mg']} mg — {status}{intervalo_txt}")

        if resultado_hist["doses_perdidas"]:
            st.warning(f"⚠ {len(resultado_hist['doses_perdidas'])} dose(s) não administrada(s): {', '.join(resultado_hist['doses_perdidas'])}")

        if resultado_hist["menor_intervalo_h"] is not None:
            esperado = intervalo_h
            if resultado_hist["menor_intervalo_h"] < esperado * 0.8:
                st.error(f"🚨 Intervalo mínimo detectado: {resultado_hist['menor_intervalo_h']}h — abaixo do esperado ({esperado}h). Risco de acúmulo.")
            else:
                st.success(f"✅ Intervalos dentro do esperado. Menor intervalo: {resultado_hist['menor_intervalo_h']}h")

        if resultado_hist["alertas"]:
            for alerta in resultado_hist["alertas"]:
                st.error(alerta)

# =============================================================
# ABA 2 — CURVA PREDITIVA
# =============================================================

with aba2:
    st.subheader("Curva de Concentração Sérica Predita")

    col_c1, col_c2 = st.columns([2, 1])

    with col_c2:
        modo_infusao = st.radio(
            "Modo de infusão",
            ["intermitente", "contínua"],
            help="Contínua: dose diária total administrada em 24h sem interrupção"
        )

        if modo_infusao == "intermitente":
            n_doses_curva = st.slider("Número de doses", min_value=1, max_value=10, value=5, key="n_doses_curva")
        else:
            duracao_continua = st.slider("Duração da simulação (horas)", min_value=24, max_value=168, value=72, step=24)
            dose_dia_mg = st.number_input("Dose diária total (mg)", min_value=100.0, value=float(round(dose_mg * (24 / intervalo_h))), step=100.0)

        st.caption(
            "A curva PK/PD é a referência clínica usada para pico, vale, AUC/MIC, T>MIC e toxicidade. "
            "Os modelos treinados entram apenas como sobreposição exploratória."
        )
        tem_doseamento = st.checkbox("Tenho doseamento sérico real")
        conc_real      = None
        horario_coleta = None

        if tem_doseamento:
            conc_real      = st.number_input("Concentração real (mg/L)", min_value=0.1, value=15.0, step=0.1)
            horario_coleta = st.text_input("Horário da coleta (HH:MM)", value="07:00")
            horario_ult    = st.text_input("Horário da última dose (HH:MM)", value="20:00")

    with col_c1:
        alvo     = config_atb.get("concentracao_alvo", {})
        alvo_min = alvo.get("minimo_mg_L")
        alvo_max = alvo.get("maximo_mg_L")

        alvo_fd_cfg = config_atb.get("alvo_fd", {})
        conc_alvo_pk = None
        if "T>MIC" in str(alvo_fd_cfg):
            multiplo_mic = alvo_fd_cfg.get("multiplo_mic_alvo")
            if multiplo_mic:
                conc_alvo_pk = mic_mg_L * multiplo_mic
            else:
                conc_alvo_pk = alvo_fd_cfg.get("concentracao_alvo_mg_L")
        else:
            conc_alvo_pk = alvo_fd_cfg.get("concentracao_alvo_mg_L")

        horario_coleta_h = None
        curva_calibrada  = None
        info_calibracao  = None

        # Tempo de infusão real (de volume/taxa na Aba 1) ou prescrito
        tempo_inf_pk = st.session_state.get("tempo_inf_real_h", tempo_inf)
        if tempo_inf_pk != tempo_inf:
            st.caption(f"⚡ Usando tempo de infusão real calculado: **{tempo_inf_pk}h** "
                       f"(prescrito: {tempo_inf}h)")

        if tem_doseamento and horario_coleta and horario_ult:
            fmt  = "%H:%M"
            hoje = datetime.today()
            t_col = datetime.strptime(horario_coleta, fmt).replace(year=hoje.year, month=hoje.month, day=hoje.day)
            t_ult = datetime.strptime(horario_ult, fmt).replace(year=hoje.year, month=hoje.month, day=hoje.day)
            if t_col < t_ult:
                t_col += timedelta(days=1)
            horario_coleta_h = (t_col - t_ult).total_seconds() / 3600

            # Calibração individual: ajusta CL (e Vd) para passar pelo ponto real
            if conc_real and horario_coleta_h and horario_coleta_h > 0:
                try:
                    from pk_engine import calibrar_com_doseamento
                    cal = calibrar_com_doseamento(
                        conc_medida_mg_L    = float(conc_real),
                        tempo_coleta_h      = float(horario_coleta_h),
                        dose_mg             = float(dose_mg),
                        intervalo_h         = float(intervalo_h),
                        tempo_infusao_h     = float(tempo_inf_pk),
                        numero_doses        = int(n_doses_curva),
                        vd_inicial_L        = float(vd_total),
                        cl_inicial_L_h      = float(cl_L_h),
                    )
                    curva_calibrada = cal["curva"]
                    curva_calibrada["modelo"] = "calibrado"
                    info_calibracao = cal
                except Exception as e_cal:
                    st.warning(f"Calibração não convergiu: {e_cal}")

        modelo_usado = "pkpd"
        curva_ml_selecionada = None

        if modo_infusao == "intermitente":
            curva_pk_fd = gerar_curva_por_modelo(
                nome_modelo          = "pkpd",
                dose_base_mg         = dose_mg,
                intervalo_base_h     = intervalo_h,
                tempo_infusao_base_h = tempo_inf_pk,
                numero_doses         = n_doses_curva,
            )
            curva = curva_pk_fd

            if modelo_selecionado != "pkpd":
                try:
                    curva_ml_selecionada = gerar_curva_por_modelo(
                        nome_modelo            = modelo_selecionado,
                        dose_base_mg           = dose_mg,
                        intervalo_base_h       = intervalo_h,
                        tempo_infusao_base_h   = tempo_inf_pk,
                        numero_doses           = n_doses_curva,
                        estrutura              = estrutura_sel,
                    )
                    modelo_usado = modelo_selecionado
                except FileNotFoundError as e:
                    st.warning(str(e))
                    st.info("Mostrando apenas a curva PK/PD. Treine ou reative o modelo na aba 'Comparação de Modelos'.")
                except Exception as e:
                    st.warning(f"Não foi possível calcular a curva do modelo {modelo_selecionado}: {e}")
        else:
            if modelo_selecionado != "pkpd":
                st.warning("Infusão contínua permanece no motor PK/PD. Os modelos treinados atuais foram estruturados para curva intermitente.")
            curva = gerar_curva_infusao_continua(
                dose_mg_dia     = dose_dia_mg,
                vd_L            = vd_total,
                clearance_L_h   = cl_L_h,
                duracao_total_h = duracao_continua,
            )
            curva["modelo"] = "pkpd"

        fig, ax = plt.subplots(figsize=(12, 6))

        titulo_curva = "Curva PK/PD usada nos indicadores"
        if curva_calibrada is not None:
            titulo_curva = "Curva calibrada com doseamento real (usada nos indicadores)"

        plot_curva(
            tempos_h                = curva["tempos_h"],
            concentracoes           = curva["concentracoes"],
            alvo_min_mg_L           = alvo_min,
            alvo_max_mg_L           = alvo_max,
            mic_mg_L                = mic_mg_L,
            concentracao_alvo_mg_L  = conc_alvo_pk,
            tox_limiar_mg_L         = config_atb.get("toxicidade", {}).get("concentracao_limiar_mg_L"),
            titulo                  = titulo_curva,
            nome_modelo             = "pkpd",
            concentracao_real_mg_L  = conc_real,
            horario_coleta_h        = horario_coleta_h,
            ax                      = ax,
        )

        # Curva calibrada — sobreposta em verde escuro quando disponível
        if curva_calibrada is not None:
            ax.plot(
                curva_calibrada["tempos_h"], curva_calibrada["concentracoes"],
                color="#1B5E20", linewidth=2.2, linestyle="-",
                alpha=0.90, label="calibrado (doseamento real)",
            )

        if modo_infusao == "intermitente":
            CORES_ML = {
                "bayesianridge"    : "#9C27B0",
                "lightgbm"         : "#4CAF50",
                "gradientboosting" : "#FF9800",
                "randomforest"     : "#F44336",
            }

            if curva_ml_selecionada is not None:
                cor_ml = CORES_ML.get(modelo_selecionado, "#607D8B")
                ax.plot(
                    curva_ml_selecionada["tempos_h"], curva_ml_selecionada["concentracoes"],
                    color=cor_ml, linewidth=1.7, linestyle="--", alpha=0.85,
                    label=f"{modelo_selecionado} (exploratório)",
                )
                if curva_ml_selecionada.get("ic_inferior") and curva_ml_selecionada.get("ic_superior"):
                    ax.fill_between(
                        curva_ml_selecionada["tempos_h"],
                        curva_ml_selecionada["ic_inferior"],
                        curva_ml_selecionada["ic_superior"],
                        color=cor_ml, alpha=0.10, label=f"IC 95% {modelo_selecionado}",
                    )

            if comparar_todos:
                try:
                    from models import listar_modelos_ativos as _listar
                    modelos_ativos = [m for m in _listar(atb_selecionado, estrutura_sel) if m != modelo_selecionado]
                    for nome_m in modelos_ativos:
                        try:
                            curva_m = gerar_curva_por_modelo(
                                nome_modelo          = nome_m,
                                dose_base_mg         = dose_mg,
                                intervalo_base_h     = intervalo_h,
                                tempo_infusao_base_h = tempo_inf,
                                numero_doses         = n_doses_curva,
                                estrutura            = estrutura_sel,
                            )
                            ax.plot(
                                curva_m["tempos_h"], curva_m["concentracoes"],
                                color=CORES_ML.get(nome_m, "#607D8B"), linewidth=1.2,
                                linestyle=":", alpha=0.75, label=f"{nome_m} (exploratório)",
                            )
                        except FileNotFoundError:
                            continue
                        except Exception:
                            continue
                except Exception as e:
                    st.warning(f"Não foi possível comparar todos os modelos: {e}")

            ax.legend(fontsize=8, loc="upper right")

        if modelo_usado == "pkpd":
            st.caption("Indicadores da curva PK/PD de referência (convenção monocompartimental).")
        else:
            st.caption(f"Esquerda: PK/PD de referência (convenção). Direita: modelo {modelo_usado} "
                       f"treinado na estrutura '{estrutura_sel}'.")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    if modo_infusao == "intermitente":
        def _indices_curva(curva_d):
            auc_ = calcular_auc(curva_d["tempos_h"], curva_d["concentracoes"], intervalo_h)
            return {
                "pico"   : curva_d["pico_mg_L"],
                "vale"   : curva_d["vale_mg_L"],
                "auc_mic": calcular_auc_mic(auc_, mic_mg_L),
                "t_mic"  : calcular_t_mic(curva_d["tempos_h"], curva_d["concentracoes"], mic_mg_L, intervalo_h),
            }

        ind_ref = _indices_curva(curva_pk_fd)

        # Quando há calibração individual, os índices vêm da curva calibrada
        if curva_calibrada is not None:
            curva_fd_indices = curva_calibrada
            ind_mod = _indices_curva(curva_calibrada)
            col_ref, col_mod = st.columns(2)
            with col_ref:
                st.markdown("**PK/PD (referência populacional)**")
                st.metric("Pico", f"{ind_ref['pico']} mg/L")
                st.metric("Vale", f"{ind_ref['vale']} mg/L")
                st.metric("AUC/MIC", f"{ind_ref['auc_mic']}")
                st.metric("T>MIC", f"{ind_ref['t_mic']}%")
            with col_mod:
                st.markdown("**Calibrado com doseamento real**")
                st.metric("Pico", f"{ind_mod['pico']} mg/L")
                st.metric("Vale", f"{ind_mod['vale']} mg/L")
                st.metric("AUC/MIC", f"{ind_mod['auc_mic']}")
                st.metric("T>MIC", f"{ind_mod['t_mic']}%")
            if info_calibracao:
                st.caption(
                    f"CL individual: **{info_calibracao['cl_individual_L_h']} L/h** "
                    f"(populacional: {round(cl_L_h,3)} L/h) · "
                    f"Vd: **{info_calibracao['vd_individual_L']} L** · "
                    f"Erro residual no ponto: {info_calibracao['erro_residual_mg_L']} mg/L · "
                    f"Método: {info_calibracao['metodo']}"
                )
            curva_fd    = curva_calibrada
            auc_mic_val = ind_mod["auc_mic"]
            t_mic_val   = ind_mod["t_mic"]

        elif curva_ml_selecionada is not None:
            # duas colunas: referência (convenção) vs modelo (estrutura)
            ind_mod = _indices_curva(curva_ml_selecionada)
            col_ref, col_mod = st.columns(2)
            with col_ref:
                st.markdown("**PK/PD (referência)**")
                st.metric("Pico", f"{ind_ref['pico']} mg/L")
                st.metric("Vale", f"{ind_ref['vale']} mg/L")
                st.metric("AUC/MIC", f"{ind_ref['auc_mic']}")
                st.metric("T>MIC", f"{ind_ref['t_mic']}%")
            with col_mod:
                st.markdown(f"**{modelo_usado} · estrutura {estrutura_sel}**")
                st.metric("Pico", f"{ind_mod['pico']} mg/L")
                st.metric("Vale", f"{ind_mod['vale']} mg/L")
                st.metric("AUC/MIC", f"{ind_mod['auc_mic']}")
                st.metric("T>MIC", f"{ind_mod['t_mic']}%")
            # a avaliação de alvo segue o modelo selecionado
            curva_fd = curva_ml_selecionada
            auc_mic_val = ind_mod["auc_mic"]; t_mic_val = ind_mod["t_mic"]
        else:
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Pico (PK)", f"{ind_ref['pico']} mg/L")
            col_m2.metric("Vale (PK)", f"{ind_ref['vale']} mg/L")
            col_m3.metric("AUC/MIC (PK)", f"{ind_ref['auc_mic']}")
            col_m4.metric("T>MIC (PK)", f"{ind_ref['t_mic']}%")
            curva_fd = curva_pk_fd
            auc_mic_val = ind_ref["auc_mic"]; t_mic_val = ind_ref["t_mic"]

        avaliacao = avaliar_alvo_fd(atb_selecionado, {
            "t_mic_percentual" : t_mic_val,
            "auc_mic"          : auc_mic_val,
            "pico_mic"         : calcular_pico_mic(curva_fd["pico_mg_L"], mic_mg_L),
            "vale_mg_L"        : curva_fd["vale_mg_L"],
            "mic_mg_L"         : mic_mg_L,
        })
    else:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Css (estado estacionário)", f"{curva['css_mg_L']} mg/L")
        col_m2.metric("Tempo para Css (90%)", f"{curva['tempo_css_h']}h")
        aval_css = avaliar_css_vs_mic(curva["css_mg_L"], mic_mg_L)
        col_m3.metric("Css/MIC", f"{aval_css['razao_css_mic']}")
        col_m4.metric("Alvo (Css ≥ 4x MIC)", "✅ sim" if aval_css["atingiu_alvo"] else "❌ não")
        avaliacao = {}

    if avaliacao:
        st.divider()
        st.subheader("Avaliação Farmacodinâmica")
        for indice, dados in avaliacao.items():
            if dados["atingiu_alvo"]:
                st.success(f"**{indice}**: {dados['valor']} — {dados['interpretacao']} (alvo: {dados['alvo']})")
            else:
                st.error(f"**{indice}**: {dados['valor']} — {dados['interpretacao']} (alvo: {dados['alvo']})")

    if modo_infusao == "contínua":
        aval_css = avaliar_css_vs_mic(curva["css_mg_L"], mic_mg_L)
        st.divider()
        st.subheader("Avaliação Farmacodinâmica — Infusão Contínua")
        if aval_css["atingiu_alvo"]:
            st.success(f"Css {aval_css['css_mg_L']} mg/L ≥ {aval_css['multiplo_alvo']}x MIC ({aval_css['alvo_mg_L']} mg/L) — dentro do alvo")
        else:
            st.error(f"Css {aval_css['css_mg_L']} mg/L < {aval_css['multiplo_alvo']}x MIC ({aval_css['alvo_mg_L']} mg/L) — subdose")

# =============================================================
# ABA 3 — COMPARAÇÃO DE CENÁRIOS
# =============================================================

with aba3:
    st.subheader("Comparação de Cenários")
    st.caption("Compare diferentes regimes de dosagem para o mesmo paciente.")

    n_cenarios    = st.number_input("Número de cenários", min_value=2, max_value=6, value=2)
    cenarios_input = []
    dados_paciente_modelo = {
        "idade"       : idade,
        "peso_kg"     : peso_calc,
        "sexo_num"    : 1 if sexo == "masculino" else 0,
        "creat_basal" : creatinina,
        "creat_atual" : creatinina,
        "vd_L_kg"     : vd_L_kg,
    }
    if modelo_selecionado == "pkpd":
        st.caption("Cenários calculados pela curva PK/PD de referência.")
    else:
        st.caption(f"Cenários calculados pela curva do modelo {modelo_selecionado}. Se o modelo não estiver treinado/ativo, cada cenário cai de volta na curva PK/PD.")

    for i in range(int(n_cenarios)):
        with st.expander(f"Cenário {i+1}", expanded=(i < 2)):

            col_nome, col_modo = st.columns([3, 1])
            nome_cen = col_nome.text_input("Nome", value=f"Cenário {i+1}", key=f"cn_{i}")
            modo_cen = col_modo.selectbox("Modo", ["intermitente", "contínua"], key=f"cm_{i}")

            if modo_cen == "intermitente":
                cols       = st.columns(5)
                dose_cen   = cols[0].number_input("Dose unitária (mg)", value=float(round(dose_mg)), key=f"cd_{i}")
                interv_cen = cols[1].number_input("Intervalo (h)", value=float(intervalo_h), key=f"ci_{i}")
                tempo_cen  = cols[2].number_input("Tempo infusão (h)", min_value=0.1, max_value=24.0, value=float(tempo_inf), step=0.5, key=f"ct_{i}")
                n_doses_cen = cols[3].number_input("Nº doses", min_value=1, max_value=10, value=int(st.session_state.get("n_doses_curva", 5)), key=f"nd_{i}")
                inicio_cen = cols[4].text_input("Horário 1ª dose", value="08:00", key=f"ch_{i}")

                cenarios_input.append(criar_cenario(
                    nome                  = nome_cen,
                    dose_mg               = dose_cen,
                    intervalo_h           = interv_cen,
                    vd_total_L            = vd_total,
                    clearance_L_h         = cl_L_h,
                    tempo_infusao_h       = tempo_cen,
                    horario_primeira_dose = inicio_cen,
                    numero_doses          = int(n_doses_cen),
                    atb                   = atb_selecionado,
                    mic_mg_L              = mic_mg_L,
                    modelo                = modelo_selecionado,
                    dados_paciente        = dados_paciente_modelo,
                    estrutura             = estrutura_sel,
                ))

            else:
                cols         = st.columns(2)
                dose_dia_cen = cols[0].number_input(
                    "Dose diária total (mg/dia)",
                    value=float(round(dose_mg * (24 / intervalo_h))),
                    step=100.0,
                    key=f"cd_{i}"
                )
                duracao_cen  = cols[1].number_input(
                    "Duração simulação (h)",
                    min_value=24, max_value=168,
                    value=72,
                    key=f"ci_{i}"
                )

                # Para comparação, converte infusão contínua em
                # cenário equivalente com intervalo=24h e dose=dose_dia
                cenarios_input.append(criar_cenario(
                    nome                  = nome_cen,
                    dose_mg               = dose_dia_cen,
                    intervalo_h           = 24,
                    vd_total_L            = vd_total,
                    clearance_L_h         = cl_L_h,
                    tempo_infusao_h       = 24,
                    horario_primeira_dose = "08:00",
                    numero_doses          = max(1, int(duracao_cen / 24)),
                    atb                   = atb_selecionado,
                    mic_mg_L              = mic_mg_L,
                    modelo                = modelo_selecionado,
                    dados_paciente        = dados_paciente_modelo,
                    estrutura             = estrutura_sel,
                ))

    if st.button("Comparar cenários"):
        try:
            comparacao = comparar_cenarios(cenarios_input)
        except FileNotFoundError as e:
            st.error(str(e))
            st.info("Treine o modelo em 'Comparação de Modelos' antes de usá-lo nos cenários.")
            st.stop()

        # guarda o resultado para o relatório usar os "dados atuais do cenário"
        try:
            st.session_state["rel_cenarios"] = {
                "tabela"  : comparacao["tabela_resumo"].to_dict("records"),
                "colunas" : list(comparacao["tabela_resumo"].columns),
                "melhor"  : comparacao.get("melhor_cenario"),
                "modelo"  : modelo_selecionado,
                "atb"     : atb_selecionado,
            }
        except Exception:
            pass

        alvo    = config_atb.get("concentracao_alvo", {})
        tox_cfg = config_atb.get("toxicidade", {})

        # Se "comparar todos", adiciona pontos ML por cenário nos resultados
        if comparar_todos:
            try:
                from models import listar_modelos_ativos as _listar
                from models import predizer_serie_temporal as _pred_st
                modelos_cmp = [m for m in _listar(atb_selecionado, estrutura_sel) if m != modelo_selecionado]
                for res in comparacao["resultados"]:
                    pontos_extras = {}
                    db = montar_dados_base_modelo(res["dose_mg"], res["intervalo_h"], tempo_inf)
                    for nome_m in modelos_cmp:
                        try:
                            s = _pred_st(atb_selecionado, nome_m, db, estrutura=estrutura_sel)
                            pontos_extras[nome_m] = {"tempos_h": s["tempos_h"], "concentracoes": s["concentracoes"]}
                        except Exception:
                            continue
                    res["pontos_extras_modelos"] = pontos_extras
            except Exception:
                pass

        fig = plot_comparacao_cenarios(
            resultados_cenarios = comparacao["resultados"],
            alvo_min_mg_L       = alvo.get("minimo_mg_L"),
            alvo_max_mg_L       = alvo.get("maximo_mg_L"),
            tox_limiar_mg_L     = tox_cfg.get("concentracao_limiar_mg_L"),
            titulo              = f"Comparação de Cenários — {ATB_CONFIG[atb_selecionado]['nome_completo']} — PK/PD + {modelo_selecionado}",
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.dataframe(comparacao["tabela_resumo"], use_container_width=True)

        if comparacao["melhor_cenario"]:
            st.success(f"Melhor cenário: **{comparacao['melhor_cenario']}**")
        else:
            st.warning(comparacao.get("mensagem_melhor", "Nenhum cenário seguro foi selecionado."))

# =============================================================
# ABA 4 — COMPARAÇÃO DE MODELOS
# =============================================================

with aba4:
    st.subheader("Comparação de Modelos")
    st.caption("Treine, compare, ative/desative modelos e exporte relatórios.")

    usar_real = st.checkbox("Usar dataset real (se disponível)")

    # ── ETAPA 1: Treinar ─────────────────────────────────────
    if st.button("Treinar e comparar modelos"):
        try:
            from models import comparar_modelos
            with st.spinner(f"Treinando todos os modelos na estrutura '{estrutura_sel}'..."):
                tabela, detalhes = comparar_modelos(atb_selecionado, usar_real=usar_real,
                                                    retornar_detalhes=True, estrutura=estrutura_sel)
            # Guarda por estrutura para não perder resultados de treinos anteriores
            if "tab4_por_estrutura" not in st.session_state:
                st.session_state["tab4_por_estrutura"] = {}
            st.session_state["tab4_por_estrutura"][estrutura_sel] = {
                "tabela"  : tabela.to_dict("records"),
                "colunas" : list(tabela.columns),
                "detalhes": detalhes,
                "atb"     : atb_selecionado,
            }
            # Mantém compatibilidade com o restante do app
            st.session_state["tab4_tabela"]    = tabela.to_dict("records")
            st.session_state["tab4_colunas"]   = list(tabela.columns)
            st.session_state["tab4_atb"]       = atb_selecionado
            st.session_state["tab4_detalhes"]  = detalhes
            st.session_state["tab4_estrutura"] = estrutura_sel
        except FileNotFoundError as e:
            st.error(str(e))
            st.info("Execute simulate.py para gerar o dataset simulado antes de treinar.")
        except Exception as e:
            st.error(f"Erro no treino: {e}")

    # ── ETAPA 2: Resultados ──────────────────────────────────
    # Tenta carregar do disco se session_state estiver vazio (após restart)
    if st.session_state.get("tab4_atb") != atb_selecionado or "tab4_tabela" not in st.session_state:
        try:
            from models import carregar_detalhes_diagnostico, PASTA_MODELOS
            caminho_reg = os.path.join(PASTA_MODELOS, "registro.csv")
            reg = pd.read_csv(caminho_reg)
            reg_atb = reg[(reg["atb"] == atb_selecionado) & (reg["estrutura"] == estrutura_sel) & (reg["ativo"] == "sim")]
            if not reg_atb.empty:
                linhas_tab = []
                for _, row in reg_atb.iterrows():
                    linhas_tab.append({
                        "modelo"       : row["modelo"],
                        "r2"           : row["r2"],
                        "mae_mg_L"     : row["mae_mg_L"],
                        "rmse_mg_L"    : row["rmse_mg_L"],
                        "ic_disponivel": row["modelo"] == "bayesianridge",
                    })
                st.session_state["tab4_tabela"]    = linhas_tab
                st.session_state["tab4_colunas"]   = ["modelo","r2","mae_mg_L","rmse_mg_L","ic_disponivel"]
                st.session_state["tab4_atb"]       = atb_selecionado
                st.session_state["tab4_estrutura"] = estrutura_sel
                det = carregar_detalhes_diagnostico(atb_selecionado, estrutura_sel)
                if det:
                    st.session_state["tab4_detalhes"] = det
                    if "tab4_por_estrutura" not in st.session_state:
                        st.session_state["tab4_por_estrutura"] = {}
                    st.session_state["tab4_por_estrutura"][estrutura_sel] = {
                        "tabela"  : linhas_tab,
                        "colunas" : ["modelo","r2","mae_mg_L","rmse_mg_L","ic_disponivel"],
                        "detalhes": det,
                        "atb"     : atb_selecionado,
                    }
        except Exception:
            pass

    if st.session_state.get("tab4_atb") == atb_selecionado and "tab4_tabela" in st.session_state:
        tabela = pd.DataFrame(
            st.session_state["tab4_tabela"],
            columns=st.session_state["tab4_colunas"],
        )

        est_treino = st.session_state.get("tab4_estrutura", "mono")
        st.divider()
        st.markdown(f"**Desempenho — Conjunto de Teste (estrutura {est_treino})**")
        st.dataframe(tabela, use_container_width=True)
        modelos_reais = [m for m in tabela["modelo"] if m != "PK_simples_baseline"]
        melhor = next((m for m in tabela["modelo"] if m != "PK_simples_baseline"), tabela.iloc[0]["modelo"])
        st.success(f"Melhor modelo: **{melhor}** (menor MAE). A linha PK_simples_baseline é a convenção: "
                   f"se um modelo tem MAE menor que ela, o ML é eficiente nesta estrutura.")

        # ── ETAPA 3: Ativar / Desativar ──────────────────────
        st.divider()
        col_at1, col_at2 = st.columns(2)

        with col_at1:
            st.markdown("**Ativar modelos**")
            lista_m = modelos_reais
            ativos_atuais = []
            try:
                from models import listar_modelos_ativos as _listar
                ativos_atuais = _listar(atb_selecionado, est_treino)
            except Exception:
                pass

            modelos_ativar = st.multiselect(
                "Selecione para ativar",
                options=lista_m,
                default=ativos_atuais if ativos_atuais else [melhor],
                key="ms_ativar",
            )
            if st.button("✅ Confirmar ativação"):
                if modelos_ativar:
                    from models import ativar_modelos
                    ativar_modelos(atb_selecionado, modelos_ativar, est_treino)
                    st.session_state["tab4_ativados"] = modelos_ativar
                else:
                    st.warning("Selecione ao menos um modelo.")

        with col_at2:
            st.markdown("**Desativar modelos**")
            modelos_desativar = st.multiselect(
                "Selecione para desativar",
                options=lista_m,
                default=[],
                key="ms_desativar",
            )
            if st.button("❌ Confirmar desativação"):
                if modelos_desativar:
                    try:
                        from models import listar_modelos_ativos as _listar2
                        ativos = _listar2(atb_selecionado, est_treino)
                        novos_ativos = [m for m in ativos if m not in modelos_desativar]
                        from models import ativar_modelos as _ativar
                        _ativar(atb_selecionado, novos_ativos, est_treino)
                        st.session_state["tab4_ativados"] = novos_ativos
                        st.success(f"Desativados: {', '.join(modelos_desativar)}")
                    except Exception as e:
                        st.error(str(e))

        if st.session_state.get("tab4_ativados"):
            st.success(f"✅ Ativos: **{', '.join(st.session_state['tab4_ativados'])}** — selecione na barra lateral.")
        elif ativos_atuais:
            st.caption(f"Ativos no registro: {', '.join(ativos_atuais)}")

        # ── ETAPA 4: Gráfico 5 pontos + Scatter ─────────────
        st.divider()
        st.markdown("**Comparação visual dos modelos**")

        try:
            resultados_curvas = []
            from models import predizer_curva_modelo as _pred_curva

            # pkpd como referência
            curva_ref = gerar_curva_por_modelo(
                nome_modelo          = "pkpd",
                dose_base_mg         = dose_mg,
                intervalo_base_h     = intervalo_h,
                tempo_infusao_base_h = tempo_inf,
                numero_doses         = int(st.session_state.get("n_doses_curva", 5)),
            )
            resultados_curvas.append({
                "nome_modelo"       : "pkpd",
                "tempos_h"          : curva_ref["tempos_h"],
                "concentracoes"     : curva_ref["concentracoes"],
                "pontos_treino_h"   : None,
                "pontos_treino_conc": None,
                "r2"                : "-",
                "mae_mg_L"          : "-",
                "rmse_mg_L"         : "-",
                "ic_disponivel"     : False,
            })

            dados_base_tab4 = montar_dados_base_modelo(dose_mg, intervalo_h, tempo_inf)
            duracao_tab4    = int(st.session_state.get("n_doses_curva", 5)) * intervalo_h

            for _, linha in tabela.iterrows():
                nome_m = linha["modelo"]
                if nome_m == "PK_simples_baseline":
                    continue
                try:
                    curva_m = _pred_curva(
                        atb             = atb_selecionado,
                        nome_modelo     = nome_m,
                        dados_base      = dados_base_tab4,
                        duracao_total_h = duracao_tab4,
                        estrutura       = est_treino,
                    )
                    resultados_curvas.append({
                        "nome_modelo"       : nome_m,
                        "estrutura"         : est_treino,
                        "tempos_h"          : curva_m["tempos_h"],
                        "concentracoes"     : curva_m["concentracoes"],
                        "pontos_treino_h"   : curva_m.get("pontos_treino_h"),
                        "pontos_treino_conc": curva_m.get("pontos_treino_conc"),
                        "r2"                : linha.get("r2", "-"),
                        "mae_mg_L"          : linha.get("mae_mg_L", "-"),
                        "rmse_mg_L"         : linha.get("rmse_mg_L", "-"),
                        "ic_disponivel"     : linha.get("ic_disponivel", False),
                        "ic_inferior"       : curva_m.get("ic_inferior"),
                        "ic_superior"       : curva_m.get("ic_superior"),
                    })
                except FileNotFoundError:
                    continue

            alvo = config_atb.get("concentracao_alvo", {})
            fig_mod = plot_comparacao_modelos(
                resultados_modelos = resultados_curvas,
                alvo_min_mg_L      = alvo.get("minimo_mg_L"),
                alvo_max_mg_L      = alvo.get("maximo_mg_L"),
                titulo             = f"Comparação de Modelos — {ATB_CONFIG[atb_selecionado]['nome_completo']}",
            )
            st.pyplot(fig_mod, use_container_width=True)
            plt.close(fig_mod)

        except Exception as e:
            st.warning(f"Não foi possível gerar o gráfico comparativo: {e}")

        # ── ETAPA 4b: Seleção estrutural por AIC/BIC ─────────
        st.divider()
        st.markdown("**Seleção de estrutura PK por AIC/BIC**")
        st.caption("Ajusta mono, bi e tri à curva do paciente atual e indica qual tem menor resíduo "
                   "penalizado pelo número de parâmetros. Menor AIC/BIC = melhor ajuste sem sobreajuste.")
        if st.button("🔬 Calcular melhor estrutura (AIC/BIC)", key="btn_aicbic"):
            try:
                from pk_engine import selecionar_estrutura_aic_bic, gerar_curva_concentracao as _gcurva
                n_doses_aicbic = int(st.session_state.get("n_doses_curva", 5))
                curva_ref_aicbic = _gcurva(
                    dose_mg=dose_mg, vd_L=vd_total, clearance_L_h=cl_L_h,
                    intervalo_h=intervalo_h, tempo_infusao_h=tempo_inf,
                    numero_doses=n_doses_aicbic, resolucao_h=0.5,
                )
                t_obs = np.array(curva_ref_aicbic["tempos_h"])
                c_obs = np.array(curva_ref_aicbic["concentracoes"])
                res_aicbic = selecionar_estrutura_aic_bic(
                    tempos_obs=t_obs, concs_obs=c_obs,
                    dose_mg=dose_mg, intervalo_h=intervalo_h,
                    tempo_infusao_h=tempo_inf, numero_doses=n_doses_aicbic,
                    CL_chute=cl_L_h, V_chute=vd_total,
                )
                df_aicbic = pd.DataFrame([
                    {k: v for k, v in r.items() if k not in ("parametros","ajustou","erro")}
                    for r in res_aicbic["tabela"] if r.get("ajustou")
                ])
                st.dataframe(df_aicbic, use_container_width=True)
                melhor_est = res_aicbic["melhor_estrutura"]
                st.success(f"Melhor estrutura para este paciente: **{melhor_est}** (menor AIC). "
                           "⚠ Em dados simulados, este resultado é indicativo — a seleção definitiva "
                           "requer dados reais de concentração medida.")
            except Exception as e:
                st.error(f"Erro no cálculo AIC/BIC: {e}")

        # ── ETAPA 5: Relatório e Log ─────────────────────────
        st.divider()
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            if st.button("📄 Gerar relatório"):
                import io, datetime as dt2
                linhas_rel = [
                    f"ATBSPACE — Relatório de Modelos",
                    f"ATB: {atb_selecionado}",
                    f"Data/hora: {dt2.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    f"Dataset: {'real' if usar_real else 'simulado'}",
                    "",
                    "Desempenho dos modelos (conjunto de teste):",
                    tabela.to_string(index=False),
                    "",
                ]
                try:
                    from models import listar_modelos_ativos as _listar3
                    ativos_rel = _listar3(atb_selecionado)
                    linhas_rel.append(f"Modelos ativos: {', '.join(ativos_rel)}")
                except Exception:
                    pass
                linhas_rel += [
                    "",
                    "Dados do paciente usado nas predições:",
                    f"  Idade: {idade} anos | Peso: {peso_calc} kg | Sexo: {sexo}",
                    f"  Creatinina: {creatinina} mg/dL | ClCr: {clcr} mL/min",
                    f"  Prescrição: {dose_mg} mg q{intervalo_h}h | Infusão: {tempo_inf}h",
                ]
                relatorio_txt = "\n".join(linhas_rel)
                buf = io.BytesIO(relatorio_txt.encode("utf-8"))
                st.download_button(
                    "⬇ Baixar relatório (.txt)",
                    data=buf,
                    file_name=f"relatorio_{atb_selecionado}_{dt2.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                )

        with col_r2:
            if st.button("🪵 Gerar log de execução"):
                import io, datetime as dt3
                linhas_log = [
                    f"ATBSPACE — Log de Execução",
                    f"ATB: {atb_selecionado}",
                    f"Data/hora: {dt3.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    "",
                    "Parâmetros PK usados:",
                    f"  Vd total: {round(vd_total, 1)} L | CL: {round(cl_L_h, 3)} L/h",
                    f"  ClCr ref: {config_atb.get('populacao_padrao', {}).get('clcr_medio_mL_min', '-')} mL/min",
                    "",
                    "Registro de modelos (registro.csv):",
                ]
                try:
                    import os
                    from models import REGISTRO_CSV
                    if os.path.exists(REGISTRO_CSV):
                        df_reg = pd.read_csv(REGISTRO_CSV)
                        linhas_log.append(df_reg.to_string(index=False))
                    else:
                        linhas_log.append("  registro.csv não encontrado.")
                except Exception as e:
                    linhas_log.append(f"  Erro ao ler registro: {e}")

                log_txt = "\n".join(linhas_log)
                buf2 = io.BytesIO(log_txt.encode("utf-8"))
                st.download_button(
                    "⬇ Baixar log (.txt)",
                    data=buf2,
                    file_name=f"log_{atb_selecionado}_{dt3.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                )


# =============================================================
# ABA 5 — RELATÓRIO (independente do treino)
# =============================================================

with aba5:
    st.subheader("Relatório completo (HTML)")
    st.caption("Usa os dados atuais do paciente (barra lateral), a estrutura e o modelo selecionados. "
               "Os diagnósticos de teste (predito vs observado, resíduos) requerem treino prévio.")

    # Lê dados de treino da estrutura atual (se disponível)
    dados_estrutura = (st.session_state.get("tab4_por_estrutura") or {}).get(estrutura_sel)
    tem_treino_estrutura = (dados_estrutura is not None and
                            dados_estrutura.get("atb") == atb_selecionado)
    # fallback para compatibilidade com sessões antigas
    if not tem_treino_estrutura:
        tem_treino_estrutura = ("tab4_tabela" in st.session_state and
                                st.session_state.get("tab4_atb") == atb_selecionado and
                                st.session_state.get("tab4_estrutura") == estrutura_sel)
        if tem_treino_estrutura:
            dados_estrutura = {
                "tabela"  : st.session_state["tab4_tabela"],
                "colunas" : st.session_state["tab4_colunas"],
                "detalhes": st.session_state.get("tab4_detalhes", {}),
                "atb"     : atb_selecionado,
            }

    est_treino_salvo = estrutura_sel if tem_treino_estrutura else st.session_state.get("tab4_estrutura", "mono")

    if not tem_treino_estrutura:
        st.info(f"Nenhum treino disponível para a estrutura '{estrutura_sel}'. "
                f"Treine em 'Comparação de Modelos' para incluir desempenho e diagnósticos.")

    tabela_modelos_rel = dados_estrutura["tabela"] if tem_treino_estrutura else []
    est_rel = estrutura_sel

    from models import listar_modelos_ativos as _listar_rel
    modelos_ativos_rel = _listar_rel(atb_selecionado, estrutura_sel)
    if not modelos_ativos_rel and tabela_modelos_rel:
        modelos_ativos_rel = [r["modelo"] for r in tabela_modelos_rel
                              if r["modelo"] != "PK_simples_baseline"]

    sel_rel = []
    if modelos_ativos_rel:
        st.markdown(f"**Modelos a incluir** (ativos na estrutura {estrutura_sel})")
        cols_chk = st.columns(min(4, len(modelos_ativos_rel)))
        for i, nome_m in enumerate(modelos_ativos_rel):
            if cols_chk[i % len(cols_chk)].checkbox(nome_m, value=True, key=f"rel5_chk_{nome_m}"):
                sel_rel.append(nome_m)
    else:
        st.info(f"Nenhum modelo ativo na estrutura '{estrutura_sel}'. Treine em 'Comparação de Modelos'.")

    incluir_cenarios = st.checkbox("Incluir comparação de cenários (gerada agora para o paciente atual)",
                                   value=True, key="rel5_cenarios")

    if st.button("📄 Gerar relatório", key="rel5_btn"):
        try:
            import datetime as _dtr5
            from report import gerar_relatorio_html

            n_doses_rel = int(st.session_state.get("n_doses_curva", 5))

            def _indices_rel5(curva_d):
                auc_ = calcular_auc(curva_d["tempos_h"], curva_d["concentracoes"], intervalo_h)
                return {
                    "pico"    : curva_d.get("pico_mg_L"),
                    "vale"    : curva_d.get("vale_mg_L"),
                    "auc"     : auc_,
                    "auc_mic" : calcular_auc_mic(auc_, mic_mg_L),
                    "t_mic"   : calcular_t_mic(curva_d["tempos_h"], curva_d["concentracoes"],
                                               mic_mg_L, intervalo_h),
                }

            # ── Curva PK de referência (sempre disponível) ──
            curva_pkpd_rel = gerar_curva_por_modelo(
                nome_modelo="pkpd", dose_base_mg=dose_mg, intervalo_base_h=intervalo_h,
                tempo_infusao_base_h=tempo_inf, numero_doses=n_doses_rel,
            )
            curvas_rel = {
                "pkpd": {
                    "tempos_h"     : curva_pkpd_rel["tempos_h"],
                    "concentracoes": curva_pkpd_rel["concentracoes"],
                    "pk_base"      : None,
                    "indices"      : _indices_rel5(curva_pkpd_rel),
                    "carregou"     : True,
                }
            }

            # ── Curvas de todos os modelos selecionados na estrutura atual ──
            for nome_m in sel_rel:
                if nome_m == "pkpd":
                    continue
                try:
                    curva_m = gerar_curva_por_modelo(
                        nome_modelo=nome_m, dose_base_mg=dose_mg,
                        intervalo_base_h=intervalo_h, tempo_infusao_base_h=tempo_inf,
                        numero_doses=n_doses_rel, estrutura=est_rel,
                    )
                    carregou = True
                except Exception:
                    curva_m = curva_pkpd_rel
                    carregou = False
                curvas_rel[nome_m] = {
                    "tempos_h"     : curva_m["tempos_h"],
                    "concentracoes": curva_m["concentracoes"],
                    "pk_base"      : curva_m.get("pk_base"),
                    "indices"      : _indices_rel5(curva_m),
                    "carregou"     : carregou,
                }

            # ── Cenários: gerados agora para todos os modelos ativos ──
            cenarios_rel = None
            if incluir_cenarios and sel_rel:
                try:
                    from scenarios import criar_cenario, comparar_cenarios
                    dados_pac_cen = montar_dados_base_modelo(dose_mg, intervalo_h, tempo_inf)
                    cens_rel = []
                    for nome_m in sel_rel:
                        cens_rel.append(criar_cenario(
                            nome=f"{nome_m}",
                            dose_mg=dose_mg, intervalo_h=intervalo_h,
                            vd_total_L=vd_total, clearance_L_h=cl_L_h,
                            tempo_infusao_h=tempo_inf, numero_doses=n_doses_rel,
                            atb=atb_selecionado, mic_mg_L=mic_mg_L,
                            modelo=nome_m, dados_paciente=dados_pac_cen,
                            estrutura=est_rel,
                        ))
                    comp_rel = comparar_cenarios(cens_rel)
                    cenarios_rel = {
                        "tabela" : comp_rel["tabela_resumo"].to_dict("records"),
                        "colunas": list(comp_rel["tabela_resumo"].columns),
                        "melhor" : comp_rel.get("melhor_cenario"),
                        "modelo" : f"todos ({est_rel})",
                    }
                except Exception as e_cen:
                    st.warning(f"Não foi possível gerar cenários: {e_cen}")

            paciente_rel = {
                "Antimicrobiano"         : ATB_CONFIG[atb_selecionado]["nome_completo"],
                "Estrutura PK"           : est_rel,
                "Idade"                  : f"{idade} anos",
                "Peso (cálculo)"         : f"{peso_calc} kg",
                "Sexo"                   : sexo,
                "Creatinina"             : f"{creatinina} mg/dL",
                "ClCr (Cockcroft-Gault)" : f"{clcr} mL/min",
                "Vd total"               : f"{round(vd_total, 1)} L",
                "Clearance do fármaco"   : f"{round(cl_L_h, 3)} L/h",
                "Dose"                   : f"{dose_mg} mg",
                "Intervalo"              : f"{intervalo_h} h",
                "Tempo de infusão prescrito" : f"{tempo_inf} h",
                "Tempo de infusão real"  : f"{tempo_inf_pk} h" if tempo_inf_pk != tempo_inf else f"{tempo_inf} h",
                "MIC"                    : f"{mic_mg_L} mg/L",
            }

            # Monta dados de calibração para o relatório quando disponível
            calibracao_rel = None
            if info_calibracao and curva_calibrada:
                ind_cal = _indices_rel5(curva_calibrada)
                curva_pk_rel = gerar_curva_por_modelo(
                    nome_modelo="pkpd", dose_base_mg=dose_mg,
                    intervalo_base_h=intervalo_h,
                    tempo_infusao_base_h=tempo_inf_pk,
                    numero_doses=n_doses_rel,
                )
                ind_pop = _indices_rel5(curva_pk_rel)

                # Avaliação FD da curva calibrada
                aval_cal = avaliar_alvo_fd(atb_selecionado, {
                    "t_mic_percentual" : ind_cal["t_mic"],
                    "auc_mic"          : ind_cal["auc_mic"],
                    "pico_mic"         : calcular_pico_mic(curva_calibrada["pico_mg_L"], mic_mg_L),
                    "vale_mg_L"        : curva_calibrada["vale_mg_L"],
                    "mic_mg_L"         : mic_mg_L,
                })
                aval_lista = []
                for indice, dados in (aval_cal or {}).items():
                    aval_lista.append({
                        "indice"       : indice,
                        "valor"        : dados.get("valor", "-"),
                        "atingiu"      : dados.get("atingiu_alvo", False),
                        "interpretacao": dados.get("interpretacao", "-"),
                        "alvo"         : dados.get("alvo", "-"),
                    })

                calibracao_rel = {
                    "conc_medida"          : conc_real,
                    "tempo_coleta_h"       : horario_coleta_h,
                    "horario_coleta"       : horario_coleta if tem_doseamento else "-",
                    "horario_ultima_dose"  : horario_ult if tem_doseamento else "-",
                    "cl_individual"        : info_calibracao["cl_individual_L_h"],
                    "cl_populacional"      : round(cl_L_h, 3),
                    "vd_individual"        : info_calibracao["vd_individual_L"],
                    "erro_residual"        : info_calibracao["erro_residual_mg_L"],
                    "metodo"               : info_calibracao["metodo"],
                    "indices"              : ind_cal,
                    "pico_pop"             : ind_pop["pico"],
                    "vale_pop"             : ind_pop["vale"],
                    "auc_pop"              : ind_pop["auc"],
                    "auc_mic_pop"          : ind_pop["auc_mic"],
                    "t_mic_pop"            : ind_pop["t_mic"],
                    "avaliacao"            : aval_lista,
                    "curva"                : {"tempos_h": curva_calibrada["tempos_h"],
                                              "concentracoes": curva_calibrada["concentracoes"]},
                    "curva_pk_concentracoes": None,  # não passa PK aqui — já aparece no gráfico principal
                }

            # diagnósticos: usa dados do treino da estrutura atual
            detalhes_rel = dados_estrutura["detalhes"] if tem_treino_estrutura else {}

            alvo_rel = config_atb.get("concentracao_alvo", {})
            html_rel = gerar_relatorio_html(
                meta={
                    "atb"           : atb_selecionado,
                    "nome_completo" : ATB_CONFIG[atb_selecionado]["nome_completo"],
                    "dataset"       : f"simulado — estrutura {est_rel}",
                },
                paciente=paciente_rel,
                tabela_modelos=tabela_modelos_rel,
                selecionados=sel_rel,
                detalhes=detalhes_rel,
                curvas=curvas_rel,
                alvo_min=alvo_rel.get("minimo_mg_L"),
                alvo_max=alvo_rel.get("maximo_mg_L"),
                mic=mic_mg_L,
                cenarios=cenarios_rel,
                calibracao=calibracao_rel,
            )
            st.download_button(
                "⬇ Baixar relatório (.html)",
                data=html_rel.encode("utf-8"),
                file_name=f"relatorio_{atb_selecionado}_{est_rel}_{_dtr5.datetime.now().strftime('%Y%m%d_%H%M')}.html",
                mime="text/html",
                key="rel5_dl",
            )
            st.success("Relatório gerado. Baixe e abra no navegador (dá pra imprimir em PDF).")
        except Exception as e:
            st.error(f"Erro ao gerar relatório: {e}")