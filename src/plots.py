# =============================================================
# ATBSPACE - Visualizações
# =============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta


CORES_MODELOS = {
    "pkpd"                  : "#2196F3",
    "bayesianridge"         : "#9C27B0",
    "lightgbm"              : "#4CAF50",
    "gradientboosting"      : "#FF9800",
    "randomforest"          : "#F44336",
    "PK_simples_baseline"   : "#90A4AE",
}

CORES_CENARIOS = [
    "#E53935", "#1E88E5", "#43A047",
    "#FB8C00", "#8E24AA", "#00ACC1",
]

ESTILOS_ML_CENARIO = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]


# =============================================================
# CURVA DE CONCENTRAÇÃO — UM CENÁRIO / UM MODELO
# =============================================================

def plot_curva(
    tempos_h: list,
    concentracoes: list,
    alvo_min_mg_L: float = None,
    alvo_max_mg_L: float = None,
    mic_mg_L: float = None,
    concentracao_alvo_mg_L: float = None,
    tox_limiar_mg_L: float = None,
    titulo: str = "Concentração Sérica Predita",
    nome_modelo: str = None,
    estrutura: str = None,
    concentracao_real_mg_L: float = None,
    horario_coleta_h: float = None,
    ic_inferior: list = None,
    ic_superior: list = None,
    ax=None,
) -> plt.Axes:

    proprio_ax = ax is None
    if proprio_ax:
        _, ax = plt.subplots(figsize=(10, 5))

    cor   = CORES_MODELOS.get(nome_modelo, "#2196F3")
    label = nome_modelo or "predito"
    if estrutura and nome_modelo and nome_modelo != "pkpd":
        label = f"{nome_modelo} ({estrutura})"

    concs = list(concentracoes) if concentracoes else []
    tempos = list(tempos_h) if tempos_h else []

    if not concs or not tempos:
        ax.text(0.5, 0.5, "Sem dados para exibir", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="#888")
        return ax

    ax.plot(tempos, concs, color=cor, linewidth=2, label=label)

    # IC 95% — só para BayesianRidge
    if ic_inferior and ic_superior and len(ic_inferior) == len(tempos):
        ax.fill_between(tempos, ic_inferior, ic_superior,
                        color=cor, alpha=0.15, label="IC 95%")

    # Zona terapêutica
    if alvo_min_mg_L is not None and alvo_max_mg_L is not None:
        ax.axhspan(alvo_min_mg_L, alvo_max_mg_L,
                   color="#A5D6A7", alpha=0.25, label="janela terapêutica")
        ax.axhline(alvo_min_mg_L, color="#388E3C", linewidth=1.0,
                   linestyle="--", label=f"alvo mín: {alvo_min_mg_L} mg/L")
        ax.axhline(alvo_max_mg_L, color="#388E3C", linewidth=1.0,
                   linestyle="--", label=f"alvo máx: {alvo_max_mg_L} mg/L")

    # MIC
    if mic_mg_L:
        ax.axhline(mic_mg_L, color="#E53935", linewidth=1.5,
                   linestyle="-.", label=f"MIC: {mic_mg_L} mg/L")

    # Zona de toxicidade
    if tox_limiar_mg_L:
        topo = max(max(concs) * 1.1 if concs else tox_limiar_mg_L * 1.2,
                   tox_limiar_mg_L * 1.2)
        ax.axhspan(tox_limiar_mg_L, topo, color="#D32F2F", alpha=0.10,
                   label=f"zona tóxica (>{tox_limiar_mg_L} mg/L)")
        ax.axhline(tox_limiar_mg_L, color="#D32F2F", linewidth=1.2, linestyle=":")

    # Alvo pontual (quando não há faixa min-max)
    if concentracao_alvo_mg_L and not (alvo_min_mg_L is not None and alvo_max_mg_L is not None):
        ax.axhline(concentracao_alvo_mg_L, color="#FB8C00", linewidth=1.5,
                   linestyle="--", label=f"concentração alvo: {concentracao_alvo_mg_L} mg/L")

    # Doseamento real
    if concentracao_real_mg_L is not None and horario_coleta_h is not None:
        ax.scatter(horario_coleta_h, concentracao_real_mg_L,
                   color="#D32F2F", zorder=5, s=80,
                   label=f"doseamento real: {concentracao_real_mg_L} mg/L")

    ax.set_xlabel("Tempo (horas)", fontsize=11)
    ax.set_ylabel("Concentração (mg/L)", fontsize=11)
    ax.set_title(titulo, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if proprio_ax:
        plt.tight_layout()

    return ax


# =============================================================
# COMPARAÇÃO DE MODELOS — MÚLTIPLAS CURVAS
# =============================================================

def plot_comparacao_modelos(
    resultados_modelos: list,
    alvo_min_mg_L: float = None,
    alvo_max_mg_L: float = None,
    titulo: str = "Comparação de Modelos",
    y_real: list = None,
    y_pred_por_modelo: dict = None,
) -> plt.Figure:

    ESTILOS = {
        "pkpd"             : "-",
        "bayesianridge"    : "--",
        "lightgbm"         : "-.",
        "gradientboosting" : (0, (5, 1)),
        "randomforest"     : ":",
        "PK_simples_baseline": (0, (3, 1, 1, 1)),
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax_curva   = axes[0]
    ax_direito = axes[1]

    if alvo_min_mg_L is not None and alvo_max_mg_L is not None:
        ax_curva.axhspan(alvo_min_mg_L, alvo_max_mg_L, color="#A5D6A7", alpha=0.20)
        ax_curva.axhline(alvo_min_mg_L, color="#388E3C", linewidth=0.8, linestyle="--")
        ax_curva.axhline(alvo_max_mg_L, color="#388E3C", linewidth=0.8, linestyle="--")

    linhas_tabela = []

    for r in resultados_modelos:
        nome  = r["nome_modelo"]
        cor   = CORES_MODELOS.get(nome, "#607D8B")
        estilo = ESTILOS.get(nome, "--")
        est   = r.get("estrutura")
        label = f"{nome} ({est})" if est and nome not in ("pkpd", "PK_simples_baseline") else nome

        tempos = r.get("tempos_h") or []
        concs  = r.get("concentracoes") or []
        if tempos and concs:
            ax_curva.plot(tempos, concs, color=cor, linewidth=1.8,
                          linestyle=estilo, alpha=0.85, label=label)

        pts_t = r.get("pontos_treino_h")
        pts_c = r.get("pontos_treino_conc")
        if pts_t and pts_c and len(pts_t) > 0:
            ax_curva.scatter(pts_t, pts_c, color=cor, s=60,
                             zorder=6, edgecolors="black", linewidths=0.5)

        ic_inf = r.get("ic_inferior")
        ic_sup = r.get("ic_superior")
        if ic_inf and ic_sup and tempos and len(ic_inf) == len(tempos):
            ax_curva.fill_between(tempos, ic_inf, ic_sup, color=cor, alpha=0.10)

        linhas_tabela.append([
            nome,
            str(r.get("r2", "-")),
            str(r.get("mae_mg_L", "-")),
            str(r.get("rmse_mg_L", "-")),
            "sim" if r.get("ic_disponivel") else "não",
        ])

    ax_curva.set_xlabel("Tempo (horas)", fontsize=11)
    ax_curva.set_ylabel("Concentração (mg/L)", fontsize=11)
    ax_curva.set_title("Curvas de predição por modelo", fontsize=12, fontweight="bold")
    ax_curva.legend(fontsize=9, loc="upper right")
    ax_curva.grid(True, alpha=0.3)

    # Painel direito: scatter predito vs observado ou tabela de desempenho
    todos_pred = [v for vals in (y_pred_por_modelo or {}).values() for v in vals]
    if y_real and todos_pred:
        lim_min = min(min(y_real), min(todos_pred)) * 0.9
        lim_max = max(max(y_real), max(todos_pred)) * 1.1
        ax_direito.plot([lim_min, lim_max], [lim_min, lim_max],
                        "k--", linewidth=1, label="predição perfeita")
        for nome_m, y_p in y_pred_por_modelo.items():
            cor = CORES_MODELOS.get(nome_m, "#607D8B")
            ax_direito.scatter(y_real, y_p, color=cor, alpha=0.4, s=20, label=nome_m)
        ax_direito.set_xlabel("Observado (mg/L)", fontsize=11)
        ax_direito.set_ylabel("Predito (mg/L)", fontsize=11)
        ax_direito.set_title("Predito vs Observado", fontsize=12, fontweight="bold")
        ax_direito.legend(fontsize=9)
        ax_direito.grid(True, alpha=0.3)
    else:
        ax_direito.axis("off")
        if linhas_tabela:
            tabela = ax_direito.table(
                cellText  = linhas_tabela,
                colLabels = ["Modelo", "R²", "MAE (mg/L)", "RMSE (mg/L)", "IC 95%"],
                cellLoc   = "center",
                loc       = "center",
            )
            tabela.auto_set_font_size(False)
            tabela.set_fontsize(10)
            tabela.scale(1.2, 1.8)
            for (row, col), cell in tabela.get_celld().items():
                if row == 0:
                    cell.set_facecolor("#1F4E79")
                    cell.set_text_props(color="white", fontweight="bold")
                elif row % 2 == 0:
                    cell.set_facecolor("#EBF3FB")
        ax_direito.set_title("Desempenho comparativo", fontsize=12, fontweight="bold")

    fig.suptitle(titulo, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# =============================================================
# COMPARAÇÃO DE CENÁRIOS
# =============================================================

def plot_comparacao_cenarios(
    resultados_cenarios: list,
    alvo_min_mg_L: float = None,
    alvo_max_mg_L: float = None,
    tox_limiar_mg_L: float = None,
    titulo: str = "Comparação de Cenários",
) -> plt.Figure:

    CORES_ML_CEN = {
        "pkpd"             : "#2196F3",
        "bayesianridge"    : "#9C27B0",
        "lightgbm"         : "#4CAF50",
        "gradientboosting" : "#FF9800",
        "randomforest"     : "#F44336",
    }
    # Estilos de linha ML por cenário: varia para distinguir cenário 1 do cenário 2
    # mesmo quando o modelo é o mesmo (mesma cor)
    ESTILOS_ML = [":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax_curva  = axes[0]
    ax_tabela = axes[1]

    if alvo_min_mg_L is not None and alvo_max_mg_L is not None:
        ax_curva.axhspan(alvo_min_mg_L, alvo_max_mg_L, color="#A5D6A7", alpha=0.20)
        ax_curva.axhline(alvo_min_mg_L, color="#388E3C", linewidth=0.8, linestyle="--")
        ax_curva.axhline(alvo_max_mg_L, color="#388E3C", linewidth=0.8, linestyle="--")

    if tox_limiar_mg_L:
        concs_todos = [c for r in resultados_cenarios
                       for c in (r.get("concentracoes") or [])]
        y_max = max(concs_todos) if concs_todos else tox_limiar_mg_L * 1.2
        topo  = max(y_max * 1.1, tox_limiar_mg_L * 1.2)
        ax_curva.axhspan(tox_limiar_mg_L, topo, color="#D32F2F", alpha=0.08,
                         label=f"zona tóxica (>{tox_limiar_mg_L} mg/L)")
        ax_curva.axhline(tox_limiar_mg_L, color="#D32F2F", linewidth=1.2, linestyle=":")

    ESTILOS_PK = ["-", "--", "-.", (0, (5, 2)), (0, (3, 2, 1, 2))]
    linhas_tabela = []
    modelos_vistos_ml = {}   # modelo -> lista de índices de cenário já plotados

    for i, r in enumerate(resultados_cenarios):
        cor_pk = CORES_CENARIOS[i % len(CORES_CENARIOS)]
        est_pk = ESTILOS_PK[i % len(ESTILOS_PK)]

        # Curva PK/PD do cenário
        tempos = r.get("tempos_h") or []
        concs  = r.get("concentracoes") or []
        if tempos and concs:
            ax_curva.plot(tempos, concs, color=cor_pk, linewidth=2.2,
                          linestyle=est_pk, alpha=0.9,
                          label=f"{r['nome']} (PK)", zorder=10 + i)

        # Curva do modelo ML — cor por modelo, estilo varia por cenário
        curva_ml = r.get("curva_ml")
        if curva_ml and curva_ml.get("tempos_h") and curva_ml.get("concentracoes"):
            modelo_nome = r.get("modelo", "ML")
            cor_ml = CORES_ML_CEN.get(modelo_nome, "#607D8B")

            # Conta quantas vezes esse modelo já apareceu para variar o estilo
            idx_ml = len(modelos_vistos_ml.get(modelo_nome, []))
            estilo_ml = ESTILOS_ML[idx_ml % len(ESTILOS_ML)]
            modelos_vistos_ml.setdefault(modelo_nome, []).append(i)

            ax_curva.plot(
                curva_ml["tempos_h"], curva_ml["concentracoes"],
                color=cor_ml, linewidth=2.0, linestyle=estilo_ml,
                alpha=0.90, label=f"{r['nome']} ({modelo_nome})", zorder=15 + i,
            )

        # Pontos extras de outros modelos (modo "comparar todos")
        pontos_extras = r.get("pontos_extras_modelos", {})
        for nome_m, pts in pontos_extras.items():
            cor_m = CORES_ML_CEN.get(nome_m, "#607D8B")
            ax_curva.scatter(
                pts.get("tempos_h", []), pts.get("concentracoes", []),
                color=cor_m, marker="o", s=30, alpha=0.6, zorder=8,
                edgecolors="none",
                label=f"{nome_m} ({r['nome']})" if i == 0 else "_nolegend_",
            )

        # Linha da tabela
        avaliacao = r.get("avaliacao_fd") or {}
        if avaliacao:
            atingiu = "sim" if all(v["atingiu_alvo"] for v in avaliacao.values()) else "não"
        else:
            atingiu = "-"
        tox_txt = r.get("toxicidade_nivel") or r.get("tox_nivel") or "-"

        linhas_tabela.append([
            r.get("nome", f"C{i+1}"),
            r.get("modelo", "-"),
            f"{r.get('dose_mg', '-')} mg",
            f"q{int(r['intervalo_h'])}h" if r.get("intervalo_h") else "-",
            f"{r.get('pico_mg_L', '-')} mg/L",
            f"{r.get('vale_mg_L', '-')} mg/L",
            f"{r.get('t_mic_percentual', '-')}%",
            atingiu,
            tox_txt,
        ])

    ax_curva.set_xlabel("Tempo (horas)", fontsize=11)
    ax_curva.set_ylabel("Concentração (mg/L)", fontsize=11)
    ax_curva.set_title("Curvas por cenário", fontsize=12, fontweight="bold")
    # Legenda em duas colunas para não sufocar o gráfico
    ax_curva.legend(fontsize=8, loc="upper right", ncol=2)
    ax_curva.grid(True, alpha=0.3)

    # Tabela de resumo
    ax_tabela.axis("off")
    if linhas_tabela:
        tabela = ax_tabela.table(
            cellText  = linhas_tabela,
            colLabels = ["Cenário", "Modelo", "Dose", "Intervalo",
                         "Pico", "Vale", "T>MIC", "Alvo FD", "Toxicidade"],
            cellLoc   = "center",
            loc       = "center",
        )
        tabela.auto_set_font_size(False)
        tabela.set_fontsize(8)
        tabela.scale(1.1, 1.6)
        for (row, col), cell in tabela.get_celld().items():
            if row == 0:
                cell.set_facecolor("#1F4E79")
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#EBF3FB")

    ax_tabela.set_title("Resumo por cenário", fontsize=12, fontweight="bold")
    fig.suptitle(titulo, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# =============================================================
# PREDITO VS OBSERVADO
# =============================================================

def plot_predito_vs_observado(
    y_real: list,
    y_pred: list,
    nome_modelo: str,
    r2: float = None,
    mae: float = None,
) -> plt.Figure:

    fig, ax = plt.subplots(figsize=(6, 6))
    cor = CORES_MODELOS.get(nome_modelo, "#2196F3")

    if not y_real or not y_pred:
        ax.text(0.5, 0.5, "Sem dados para exibir", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="#888")
        return fig

    ax.scatter(y_real, y_pred, color=cor, alpha=0.5, s=30, label=nome_modelo)

    lim_min = min(min(y_real), min(y_pred)) * 0.9
    lim_max = max(max(y_real), max(y_pred)) * 1.1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1,
            label="predição perfeita")

    metricas = ""
    if r2  is not None: metricas += f"R² = {r2}\n"
    if mae is not None: metricas += f"MAE = {mae} mg/L"
    if metricas:
        ax.text(0.05, 0.92, metricas.strip(), transform=ax.transAxes,
                fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    ax.set_xlabel("Observado (mg/L)", fontsize=11)
    ax.set_ylabel("Predito (mg/L)", fontsize=11)
    ax.set_title(f"Predito vs Observado — {nome_modelo}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


# =============================================================
# SALVAR FIGURA
# =============================================================

def salvar_figura(fig: plt.Figure, nome_arquivo: str, pasta: str = None):
    import os
    from config import OUTPUT_DIR
    pasta = pasta or os.path.join(OUTPUT_DIR, "graficos")
    os.makedirs(pasta, exist_ok=True)
    caminho = os.path.join(pasta, nome_arquivo)
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return caminho