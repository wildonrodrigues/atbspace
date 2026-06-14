# =============================================================
# ATBSPACE - Gerador de Relatório (HTML autocontido)
# =============================================================
# Monta um relatório único com:
#   - Dados do paciente
#   - Tabela de desempenho dos modelos (R2, MAE, RMSE)
#   - Por modelo selecionado: predito vs observado + resíduos (conjunto de teste)
#   - Por modelo selecionado: curva do paciente (modelo vs PK base) + índices FD
#   - Comparação das curvas dos modelos selecionados em um único gráfico
#
# As figuras são embutidas em base64, então o HTML é um arquivo único.
# Nenhuma função aqui usa np.trapz: os índices FD chegam prontos do app.
# =============================================================

import io
import base64
import datetime as _dt

import numpy as np
import matplotlib
try:
    matplotlib.use("Agg", force=False)
except Exception:
    pass
import matplotlib.pyplot as plt


# Cores por modelo (mesma paleta usada na Aba 2)
_CORES = {
    "pkpd"             : "#1565C0",
    "bayesianridge"    : "#9C27B0",
    "lightgbm"         : "#4CAF50",
    "gradientboosting" : "#FF9800",
    "randomforest"     : "#F44336",
    "xgboost"          : "#00897B",
    "linearregression" : "#6D4C41",
    "svr"              : "#546E7A",
    "mlp"              : "#C2185B",
}


def _cor(modelo: str) -> str:
    return _CORES.get(str(modelo).lower(), "#607D8B")


def _fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _img_tag(b64: str, largura: str = "100%") -> str:
    return f'<img style="width:{largura};max-width:760px;display:block;margin:8px 0;" src="data:image/png;base64,{b64}"/>'


# -------------------------------------------------------------
# Figuras
# -------------------------------------------------------------

def _fig_scatter(modelo, y_test, y_pred, r2, mae, rmse):
    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    ax.scatter(y_test, y_pred, s=12, alpha=0.5, color=_cor(modelo), edgecolors="none")
    lim_min = float(min(y_test.min(), y_pred.min())) if len(y_test) else 0.0
    lim_max = float(max(y_test.max(), y_pred.max())) if len(y_test) else 1.0
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "--", color="#D32F2F", linewidth=1.2)
    ax.set_xlabel("Observado (mg/L)")
    ax.set_ylabel("Predito (mg/L)")
    ax.set_title(f"{modelo} — predito vs observado\nR²={r2}  MAE={mae}  RMSE={rmse}", fontsize=10)
    ax.grid(alpha=0.25)
    return fig


def _fig_residuos(modelo, y_test, y_pred):
    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuos = y_pred - y_test
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    ax.hist(residuos, bins=30, color=_cor(modelo), alpha=0.75, edgecolor="white")
    ax.axvline(0, color="#D32F2F", linestyle="--", linewidth=1.2)
    media = float(np.mean(residuos)) if len(residuos) else 0.0
    dp = float(np.std(residuos)) if len(residuos) else 0.0
    ax.set_xlabel("Resíduo (predito − observado) mg/L")
    ax.set_ylabel("Frequência")
    ax.set_title(f"{modelo} — distribuição de resíduos\nmédia={media:.2f}  DP={dp:.2f}", fontsize=10)
    ax.grid(alpha=0.25)
    return fig


def _fig_curva_paciente(modelo, tempos, conc_modelo, conc_pk, alvo_min, alvo_max, mic):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    if conc_pk is not None:
        ax.plot(tempos, conc_pk, color="#90A4AE", linewidth=1.6, label="PK/PD (referência)")
    ax.plot(tempos, conc_modelo, color=_cor(modelo), linewidth=2.0,
            label=f"{modelo}" if modelo != "pkpd" else "PK/PD")
    if alvo_min is not None and alvo_max is not None:
        ax.axhspan(alvo_min, alvo_max, color="#4CAF50", alpha=0.10, label="alvo")
    if mic:
        ax.axhline(mic, color="#FB8C00", linestyle=":", linewidth=1.2, label="MIC")
    ax.set_xlabel("Tempo (h)")
    ax.set_ylabel("Concentração (mg/L)")
    ax.set_title(f"Curva do paciente — {modelo}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return fig


def _fig_comparacao(curvas: dict, ordem: list, alvo_min, alvo_max, mic):
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    if alvo_min is not None and alvo_max is not None:
        ax.axhspan(alvo_min, alvo_max, color="#4CAF50", alpha=0.08, label="alvo")
    if mic:
        ax.axhline(mic, color="#FB8C00", linestyle=":", linewidth=1.2, label="MIC")
    for modelo in ordem:
        c = curvas.get(modelo)
        if not c:
            continue
        estilo = "-" if modelo == "pkpd" else "--"
        largura = 2.2 if modelo == "pkpd" else 1.6
        ax.plot(c["tempos_h"], c["concentracoes"], estilo, color=_cor(modelo),
                linewidth=largura, alpha=0.9, label=modelo)
    ax.set_xlabel("Tempo (h)")
    ax.set_ylabel("Concentração (mg/L)")
    ax.set_title("Comparação das curvas — mesmo paciente", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    return fig


# -------------------------------------------------------------
# Tabelas (HTML)
# -------------------------------------------------------------

def _tabela_html(cabecalhos: list, linhas: list, destacar=None) -> str:
    th = "".join(f"<th>{h}</th>" for h in cabecalhos)
    trs = []
    for linha in linhas:
        marca = (destacar and str(linha[0]).lower() == str(destacar).lower())
        estilo = ' style="background:#E8F5E9;font-weight:600;"' if marca else ""
        tds = "".join(f"<td>{c}</td>" for c in linha)
        trs.append(f"<tr{estilo}>{tds}</tr>")
    return f'<table><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table>'


# -------------------------------------------------------------
# Relatório
# -------------------------------------------------------------

def gerar_relatorio_html(
    meta: dict,
    paciente: dict,
    tabela_modelos: list,
    selecionados: list,
    detalhes: dict,
    curvas: dict,
    alvo_min=None,
    alvo_max=None,
    mic=None,
    cenarios=None,
    calibracao: dict = None,
) -> str:
    """
    meta           : {"atb", "nome_completo", "dataset"}
    paciente       : dict label -> valor (linhas da tabela de paciente)
    tabela_modelos : lista de dicts com modelo, r2, mae_mg_L, rmse_mg_L (todos treinados)
    selecionados   : lista de nomes de modelos a detalhar
    detalhes       : {modelo: {"y_test":[...], "y_pred":[...]}}
    curvas         : {modelo: {"tempos_h", "concentracoes", "pk_base"(opt), "indices":{...}}}
    calibracao     : dict com dados da calibração individual (opcional)
                     {"conc_medida", "tempo_coleta_h", "horario_coleta", "horario_ultima_dose",
                      "cl_individual", "cl_populacional", "vd_individual", "erro_residual",
                      "metodo", "indices": {"pico","vale","auc","auc_mic","t_mic"},
                      "avaliacao": [{"indice","valor","atingiu","alvo"}]}
    """
    agora = _dt.datetime.now().strftime("%d/%m/%Y %H:%M")
    nome_completo = meta.get("nome_completo", meta.get("atb", ""))
    melhor = tabela_modelos[0]["modelo"] if tabela_modelos else None

    partes = []

    # ---- Cabeçalho
    partes.append(f"""
    <h1>ATBSPACE — Relatório de Análise</h1>
    <p class="sub">{nome_completo} &nbsp;|&nbsp; Dataset: {meta.get('dataset','simulado')} &nbsp;|&nbsp; Gerado em {agora}</p>
    """)

    # ---- Paciente
    linhas_pac = [[k, v] for k, v in paciente.items()]
    partes.append("<h2>1. Dados do paciente</h2>")
    partes.append(_tabela_html(["Parâmetro", "Valor"], linhas_pac))

    # ---- Calibração individual (quando disponível)
    if calibracao:
        partes.append("<h2>2. Calibração individual com doseamento sérico real</h2>")
        partes.append(
            "<p>Quando um doseamento sérico real está disponível, o sistema estima os parâmetros "
            "farmacocinéticos individuais do paciente (CL e Vd) ajustando a curva ao ponto medido. "
            "Os índices farmacodinâmicos abaixo são calculados a partir da curva calibrada, "
            "não dos parâmetros populacionais.</p>"
        )

        # Dados do doseamento
        partes.append("<h3>2.1 Dados do doseamento</h3>")
        partes.append(_tabela_html(["Parâmetro", "Valor"], [
            ["Concentração medida", f"{calibracao.get('conc_medida', '-')} mg/L"],
            ["Horário da coleta", calibracao.get("horario_coleta", "-")],
            ["Horário da última dose", calibracao.get("horario_ultima_dose", "-")],
            ["Tempo desde a última dose", f"{round(float(calibracao.get('tempo_coleta_h', 0)), 1)} h"],
        ]))

        # Parâmetros individuais estimados
        partes.append("<h3>2.2 Parâmetros individuais estimados</h3>")
        cl_ind = calibracao.get("cl_individual", "-")
        cl_pop = calibracao.get("cl_populacional", "-")
        delta_cl = ""
        try:
            pct = (float(cl_ind) - float(cl_pop)) / float(cl_pop) * 100
            delta_cl = f" ({'+' if pct >= 0 else ''}{pct:.1f}% em relação ao populacional)"
        except Exception:
            pass
        partes.append(_tabela_html(["Parâmetro", "Valor", "Referência populacional"], [
            ["CL individual (L/h)", f"{cl_ind}{delta_cl}", f"{cl_pop} L/h"],
            ["Vd individual (L)", f"{calibracao.get('vd_individual', '-')} L", "-"],
            ["Erro residual no ponto", f"{calibracao.get('erro_residual', '-')} mg/L", "—"],
            ["Método de estimação", calibracao.get("metodo", "-"), "—"],
        ]))

        # Índices calibrados
        partes.append("<h3>2.3 Índices farmacodinâmicos (curva calibrada)</h3>")
        ind = calibracao.get("indices", {})
        partes.append(_tabela_html(
            ["Índice", "Valor calibrado", "Valor populacional (PK/PD)"],
            [
                ["Pico (mg/L)", ind.get("pico", "-"), calibracao.get("pico_pop", "-")],
                ["Vale (mg/L)", ind.get("vale", "-"), calibracao.get("vale_pop", "-")],
                ["AUC (mg·h/L)", ind.get("auc", "-"), calibracao.get("auc_pop", "-")],
                ["AUC/MIC", ind.get("auc_mic", "-"), calibracao.get("auc_mic_pop", "-")],
                ["T>MIC (%)", ind.get("t_mic", "-"), calibracao.get("t_mic_pop", "-")],
            ]
        ))

        # Avaliação farmacodinâmica calibrada
        aval = calibracao.get("avaliacao", [])
        if aval:
            partes.append("<h3>2.4 Avaliação farmacodinâmica (curva calibrada)</h3>")
            for item in aval:
                cor = "#E8F5E9" if item.get("atingiu") else "#FFEBEE"
                icone = "✅" if item.get("atingiu") else "❌"
                partes.append(
                    f'<p style="background:{cor};padding:6px 10px;border-radius:4px;margin:4px 0;">'
                    f'{icone} <b>{item.get("indice","")}</b>: {item.get("valor","")} '
                    f'— {item.get("interpretacao","")} (alvo: {item.get("alvo","")})</p>'
                )

        # Curva calibrada
        c = calibracao.get("curva")
        if c:
            fig_cal = _fig_curva_paciente(
                "calibrado", c["tempos_h"], c["concentracoes"],
                calibracao.get("curva_pk_concentracoes"),
                alvo_min, alvo_max, mic
            )
            # Adiciona ponto do doseamento
            try:
                import matplotlib.pyplot as _plt2
                import io as _io2, base64 as _b64_2
                buf2 = _io2.BytesIO()
                fig_cal.savefig(buf2, format="png", dpi=110, bbox_inches="tight")
                _plt2.close(fig_cal)
                buf2.seek(0)
                b64 = _b64_2.b64encode(buf2.read()).decode("ascii")
            except Exception:
                b64 = None
            if b64:
                partes.append(_img_tag(b64))

        secao_desempenho = "3."
        secao_curvas     = "4."
        secao_cenarios   = "5."
        secao_modelos    = "6."
    else:
        secao_desempenho = "2."
        secao_curvas     = "3."
        secao_cenarios   = "4."
        secao_modelos    = "5."

    # ---- Desempenho dos modelos
    partes.append(f"<h2>{secao_desempenho} Desempenho dos modelos (conjunto de teste)</h2>")
    linhas_tab = [
        [r.get("modelo"), r.get("r2", "-"), r.get("mae_mg_L", "-"), r.get("rmse_mg_L", "-")]
        for r in tabela_modelos
    ]
    partes.append(_tabela_html(["Modelo", "R²", "MAE (mg/L)", "RMSE (mg/L)"], linhas_tab, destacar=melhor))
    if melhor:
        partes.append(f'<p class="nota">Melhor desempenho (menor MAE): <b>{melhor}</b>. Linha destacada em verde.</p>')

    # ---- Comparação das curvas (mesmo paciente)
    ordem = (["pkpd"] if "pkpd" in curvas else []) + [m for m in selecionados if m != "pkpd"]
    if curvas:
        partes.append(f"<h2>{secao_curvas} Comparação das curvas no paciente atual</h2>")
        partes.append("<p>Curvas geradas para o mesmo paciente e regime, variando o modelo. "
                      "Se uma curva coincide com a PK/PD, aquele modelo reproduziu a curva clássica.</p>")
        fig = _fig_comparacao(curvas, ordem, alvo_min, alvo_max, mic)
        partes.append(_img_tag(_fig_b64(fig)))

        # tabela de índices FD por modelo
        linhas_idx = []
        for m in ordem:
            ind = (curvas.get(m, {}) or {}).get("indices", {}) or {}
            linhas_idx.append([
                m,
                ind.get("pico", "-"),
                ind.get("vale", "-"),
                ind.get("auc", "-"),
                ind.get("auc_mic", "-"),
                ind.get("t_mic", "-"),
            ])
        partes.append(_tabela_html(
            ["Modelo", "Pico (mg/L)", "Vale (mg/L)", "AUC (mg·h/L)", "AUC/MIC", "T>MIC (%)"],
            linhas_idx, destacar=None
        ))

    # ---- Comparação de cenários (dados atuais do cenário)
    if cenarios and cenarios.get("tabela"):
        partes.append(f"<h2>{secao_cenarios} Comparação de cenários (prescrição atual)</h2>")
        modelo_cen = cenarios.get("modelo")
        if modelo_cen:
            partes.append(f'<p class="nota">Cenários calculados pela curva do modelo <b>{modelo_cen}</b>.</p>')
        colunas = cenarios.get("colunas") or list(cenarios["tabela"][0].keys())
        linhas_cen = [[r.get(c, "-") for c in colunas] for r in cenarios["tabela"]]
        partes.append(_tabela_html(colunas, linhas_cen, destacar=cenarios.get("melhor")))
        if cenarios.get("melhor"):
            partes.append(f'<p class="nota">Melhor cenário (atinge o alvo com menor exposição segura): '
                          f'<b>{cenarios["melhor"]}</b>.</p>')
        secao_modelos = "5. Análise por modelo"
    else:
        secao_modelos = "4. Análise por modelo"

    # ---- Detalhe por modelo selecionado
    partes.append(f"<h2>{secao_modelos} Análise por modelo</h2>")
    for m in selecionados:
        partes.append(f'<h3 style="color:{_cor(m)};">{m}</h3>')

        # diagnósticos do teste
        d = detalhes.get(m)
        if d and d.get("y_test") and d.get("y_pred"):
            rr = next((r for r in tabela_modelos if r.get("modelo") == m), {})
            fig_s = _fig_scatter(m, d["y_test"], d["y_pred"],
                                 rr.get("r2", "-"), rr.get("mae_mg_L", "-"), rr.get("rmse_mg_L", "-"))
            fig_r = _fig_residuos(m, d["y_test"], d["y_pred"])
            partes.append('<div class="par">')
            partes.append(_img_tag(_fig_b64(fig_s), largura="48%"))
            partes.append(_img_tag(_fig_b64(fig_r), largura="48%"))
            partes.append("</div>")
        else:
            partes.append('<p class="nota">Sem dados de teste (y_test/y_pred) para este modelo.</p>')

        # curva do paciente: modelo vs PK base
        c = curvas.get(m)
        if c:
            pk_base = c.get("pk_base")
            fig_c = _fig_curva_paciente(m, c["tempos_h"], c["concentracoes"], pk_base,
                                        alvo_min, alvo_max, mic)
            partes.append(_img_tag(_fig_b64(fig_c)))
            ind = c.get("indices", {}) or {}
            partes.append(_tabela_html(
                ["Pico (mg/L)", "Vale (mg/L)", "AUC (mg·h/L)", "AUC/MIC", "T>MIC (%)"],
                [[ind.get("pico", "-"), ind.get("vale", "-"), ind.get("auc", "-"),
                  ind.get("auc_mic", "-"), ind.get("t_mic", "-")]]
            ))
            if c.get("carregou") is False:
                partes.append('<p class="alerta">⚠ Este modelo não pôde ser carregado e caiu de volta na curva PK/PD. '
                              'Por isso os números coincidem com o PK/PD.</p>')

    corpo = "\n".join(partes)

    html = f"""<!DOCTYPE html>
<html lang="pt-br"><head><meta charset="utf-8"/>
<title>ATBSPACE — Relatório {meta.get('atb','')}</title>
<style>
  body {{ font-family: Arial, Helvetica, sans-serif; color:#212121; max-width:820px;
          margin:24px auto; padding:0 16px; line-height:1.45; }}
  h1 {{ font-size:22px; margin-bottom:2px; }}
  h2 {{ font-size:17px; margin-top:28px; border-bottom:2px solid #1565C0; padding-bottom:4px; }}
  h3 {{ font-size:15px; margin-top:20px; margin-bottom:4px; }}
  p.sub {{ color:#555; margin-top:0; }}
  p.nota {{ color:#555; font-size:13px; }}
  p.alerta {{ color:#B71C1C; font-size:13px; font-weight:600; }}
  table {{ border-collapse:collapse; width:100%; margin:10px 0; font-size:13px; }}
  th {{ background:#1565C0; color:#fff; text-align:left; padding:6px 8px; }}
  td {{ border-bottom:1px solid #E0E0E0; padding:5px 8px; }}
  .par {{ display:flex; gap:2%; flex-wrap:wrap; }}
  footer {{ margin-top:36px; color:#888; font-size:11px; border-top:1px solid #ddd; padding-top:8px; }}
</style></head>
<body>
{corpo}
<footer>ATBSPACE — relatório gerado automaticamente. Indicadores derivados da curva de cada modelo;
PK/PD é a curva monocompartimental de referência. Documento de validação computacional, não destinado a uso clínico direto.</footer>
</body></html>"""
    return html