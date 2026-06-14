# 💊 ATBSPACE

Calculadora farmacocinética/farmacodinâmica (PK/PD) para antimicrobianos intravenosos, com simulação de curvas de concentração, avaliação de alvos terapêuticos e comparação com modelos de machine learning.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://atbspace.streamlit.app)

## Sobre o projeto

O ATBSPACE permite simular o perfil farmacocinético de antimicrobianos (Vancomicina, Meropenem, Ceftazidima/Avibactam) a partir dos dados do paciente, avaliando:

- Curvas de concentração ao longo do tempo (infusão intermitente ou contínua)
- Atingimento de alvos PK/PD (Css/MIC, T>MIC, AUC/MIC, Pico/MIC)
- Risco de toxicidade
- Comparação entre o modelo PK/PD clássico e modelos de machine learning treinados (Random Forest, LightGBM, Gradient Boosting, Bayesian Ridge)

> ⚠️ Projeto de caráter educacional/pesquisa. Os dados de pacientes utilizados são **simulados**, não representam pacientes reais.

## Estrutura do projeto

```
app/         interface Streamlit (app/app.py)
src/         motor de cálculo PK/PD, modelos de ML, simulação e relatórios
data/        dados simulados utilizados para treino/teste
notebooks/   documentação técnica e operacional
outputs/     modelos treinados (.pkl) e gráficos gerados
Literatura/  referências e estratégias de busca da literatura
```

## Executando localmente

```bash
pip install -r requirements.txt
streamlit run app/app.py
```
