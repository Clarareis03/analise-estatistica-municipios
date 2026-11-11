"""Análise estatística inferencial (itens 3, 4 e 5 da atividade).

  Item 3: intervalos de confiança (95%) para a proporção de pessoas com
          rendimento de até 1/2 salário mínimo e teste de diferença entre
          Paraíba e Rio Grande do Norte;
  Item 4: comparação das médias do IDEB (anos iniciais);
  Item 5: comparação das médias do salário médio mensal.

Todos os testes usam as amostras aleatórias simples (30 municípios por
estado), nível de significância de 5%.

Observação sobre a variável do item 3: no banco de dados, cada município
possui um PERCENTUAL de pessoas com rendimento de até 1/2 salário mínimo.
Por isso, a abordagem principal trata os 30 percentuais de cada amostra como
observações e usa a distribuição t de Student para a média. A abordagem
clássica de proporção (z, com n = 30) é calculada como complemento, para
comparação, mas ela considera cada município como se fosse um indivíduo.

Execução: python -m src.analysis.inferencia
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# =====================================================
# CONFIGURAÇÕES
# =====================================================

PASTA_PROJETO = Path(__file__).resolve().parents[2]

PASTA_PROCESSADOS = PASTA_PROJETO / "data" / "processed"
PASTA_TABELAS = PASTA_PROJETO / "outputs" / "tabelas"
PASTA_TESTES = PASTA_PROJETO / "outputs" / "resultados_testes"

ALFA = 0.05
CONFIANCA = 1 - ALFA

NOMES = {"PB": "Paraíba", "RN": "Rio Grande do Norte"}


# =====================================================
# CARREGAMENTO DAS AMOSTRAS
# =====================================================

def carregar_amostras():
    """Lê os CSVs das amostras selecionadas (30 municípios por estado)."""

    pb = pd.read_csv(
        PASTA_PROCESSADOS / "amostra_pb.csv", encoding="utf-8-sig"
    )
    rn = pd.read_csv(
        PASTA_PROCESSADOS / "amostra_rn.csv", encoding="utf-8-sig"
    )

    return {"PB": pb, "RN": rn}


# =====================================================
# FUNÇÕES AUXILIARES
# =====================================================

def formatar_p(p):
    """Formata o valor-p para exibição."""

    return "< 0,001" if p < 0.001 else f"{p:.4f}".replace(".", ",")


def decisao(p):
    """Decisão do teste ao nível ALFA."""

    return "Rejeita H0" if p < ALFA else "Não rejeita H0"


# =====================================================
# ITEM 3 - INTERVALOS DE CONFIANÇA
# =====================================================

def ic_media_t(valores):
    """IC de 95% para a média (t de Student), em pontos percentuais."""

    n = len(valores)
    media = valores.mean()
    erro = stats.sem(valores)
    limite_inf, limite_sup = stats.t.interval(
        CONFIANCA, df=n - 1, loc=media, scale=erro
    )

    return n, media, erro, limite_inf, limite_sup


def ic_proporcao_z(proporcao, n):
    """IC de 95% para uma proporção (Wald, aproximação normal)."""

    z = stats.norm.ppf(1 - ALFA / 2)
    erro = np.sqrt(proporcao * (1 - proporcao) / n)

    return proporcao - z * erro, proporcao + z * erro


def intervalos_confianca(amostras):
    """Tabela com os ICs de 95% das duas amostras, pelas duas abordagens."""

    linhas = []

    for uf, dados in amostras.items():
        valores = dados["percentual_baixa_renda"].dropna()

        # Abordagem principal: média dos percentuais municipais (t)
        n, media, erro, inf, sup = ic_media_t(valores)
        linhas.append({
            "estado": uf,
            "abordagem": "Média dos percentuais (t de Student)",
            "n": n,
            "estimativa_%": media,
            "erro_padrao_%": erro,
            "limite_inferior_%": inf,
            "limite_superior_%": sup,
            "amplitude_%": sup - inf,
        })

        # Abordagem complementar: proporção clássica (z), n = 30
        p = media / 100
        inf_z, sup_z = ic_proporcao_z(p, n)
        linhas.append({
            "estado": uf,
            "abordagem": "Proporção clássica (z, Wald, n = 30)",
            "n": n,
            "estimativa_%": p * 100,
            "erro_padrao_%": np.sqrt(p * (1 - p) / n) * 100,
            "limite_inferior_%": inf_z * 100,
            "limite_superior_%": sup_z * 100,
            "amplitude_%": (sup_z - inf_z) * 100,
        })

    return pd.DataFrame(linhas)


# =====================================================
# ITEM 3 - TESTE PARA A DIFERENÇA ENTRE AS PROPORÇÕES
# =====================================================

def teste_diferenca_proporcoes(amostras):
    """Testa H0: proporção_PB = proporção_RN (bilateral, alfa = 5%)."""

    pb = amostras["PB"]["percentual_baixa_renda"].dropna()
    rn = amostras["RN"]["percentual_baixa_renda"].dropna()

    h0 = "As proporções (médias dos percentuais) de PB e RN são iguais"
    h1 = "As proporções (médias dos percentuais) de PB e RN são diferentes"

    linhas = []

    # Abordagem principal: t de Welch sobre os percentuais municipais
    resultado = stats.ttest_ind(pb, rn, equal_var=False)
    ic = resultado.confidence_interval(CONFIANCA)
    linhas.append({
        "item": 3,
        "variavel": "Rendimento de até 1/2 salário mínimo (%)",
        "teste": "t de Welch (percentuais municipais)",
        "H0": h0,
        "H1": h1,
        "estatistica": resultado.statistic,
        "graus_liberdade": resultado.df,
        "p_valor": resultado.pvalue,
        "alfa": ALFA,
        "diferenca_PB_menos_RN": pb.mean() - rn.mean(),
        "ic95_inf": ic.low,
        "ic95_sup": ic.high,
        "decisao": decisao(resultado.pvalue),
    })

    # Abordagem complementar: z para duas proporções (n = 30 por grupo)
    n1, n2 = len(pb), len(rn)
    p1, p2 = pb.mean() / 100, rn.mean() / 100
    p_comum = (p1 * n1 + p2 * n2) / (n1 + n2)
    erro = np.sqrt(p_comum * (1 - p_comum) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / erro
    p_valor = 2 * stats.norm.sf(abs(z))
    erro_ic = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z_crit = stats.norm.ppf(1 - ALFA / 2)
    linhas.append({
        "item": 3,
        "variavel": "Rendimento de até 1/2 salário mínimo (%)",
        "teste": "z para duas proporções (n = 30 por grupo)",
        "H0": h0,
        "H1": h1,
        "estatistica": z,
        "graus_liberdade": np.nan,
        "p_valor": p_valor,
        "alfa": ALFA,
        "diferenca_PB_menos_RN": (p1 - p2) * 100,
        "ic95_inf": (p1 - p2 - z_crit * erro_ic) * 100,
        "ic95_sup": (p1 - p2 + z_crit * erro_ic) * 100,
        "decisao": decisao(p_valor),
    })

    return linhas


# =====================================================
# ITENS 4 E 5 - COMPARAÇÃO DE MÉDIAS
# =====================================================

def testar_normalidade(valores):
    """Teste de Shapiro-Wilk (H0: os dados vêm de uma normal)."""

    estatistica, p = stats.shapiro(valores)

    return estatistica, p


def testar_variancias(a, b):
    """Testes de igualdade de variâncias entre dois grupos.

    - Teste F: razão das variâncias amostrais (exige normalidade).
    - Levene (centrado na mediana): robusto à não normalidade.
    """

    f = a.var(ddof=1) / b.var(ddof=1)
    gl1, gl2 = len(a) - 1, len(b) - 1
    p_f = 2 * min(stats.f.cdf(f, gl1, gl2), stats.f.sf(f, gl1, gl2))

    estatistica_lev, p_lev = stats.levene(a, b, center="median")

    return {
        "F": f, "gl1": gl1, "gl2": gl2, "p_F": min(p_f, 1.0),
        "W_Levene": estatistica_lev, "p_Levene": p_lev,
    }


def comparar_medias(amostras, coluna, nome, item):
    """Compara as médias de PB e RN.

    Fluxo: Shapiro-Wilk -> teste F (e Levene) para as variâncias ->
    t de Student (variâncias iguais) ou t de Welch (variâncias diferentes).
    Mann-Whitney é calculado como verificação de robustez.
    """

    pb = amostras["PB"][coluna].dropna()
    rn = amostras["RN"][coluna].dropna()

    # Pressupostos
    _, p_sw_pb = testar_normalidade(pb)
    _, p_sw_rn = testar_normalidade(rn)
    var = testar_variancias(pb, rn)

    # Escolha do teste pela igualdade de variâncias (teste F, alfa = 5%)
    variancias_iguais = var["p_F"] >= ALFA
    nome_teste = (
        "t de Student (variâncias iguais)"
        if variancias_iguais else "t de Welch (variâncias diferentes)"
    )

    resultado = stats.ttest_ind(pb, rn, equal_var=variancias_iguais)
    ic = resultado.confidence_interval(CONFIANCA)

    # Verificação de robustez (não depende de normalidade)
    u, p_mw = stats.mannwhitneyu(pb, rn, alternative="two-sided")

    teste = {
        "item": item,
        "variavel": nome,
        "teste": nome_teste,
        "H0": "As médias de PB e RN são iguais (μ_PB = μ_RN)",
        "H1": "As médias de PB e RN são diferentes (μ_PB ≠ μ_RN)",
        "estatistica": resultado.statistic,
        "graus_liberdade": resultado.df,
        "p_valor": resultado.pvalue,
        "alfa": ALFA,
        "diferenca_PB_menos_RN": pb.mean() - rn.mean(),
        "ic95_inf": ic.low,
        "ic95_sup": ic.high,
        "decisao": decisao(resultado.pvalue),
    }

    pressupostos = {
        "item": item,
        "variavel": nome,
        "n_PB": len(pb),
        "n_RN": len(rn),
        "media_PB": pb.mean(),
        "media_RN": rn.mean(),
        "variancia_PB": pb.var(ddof=1),
        "variancia_RN": rn.var(ddof=1),
        "shapiro_p_PB": p_sw_pb,
        "shapiro_p_RN": p_sw_rn,
        "F": var["F"],
        "gl_F": f"{var['gl1']};{var['gl2']}",
        "p_F": var["p_F"],
        "p_Levene": var["p_Levene"],
        "variancias": "Iguais" if variancias_iguais else "Diferentes",
        "teste_escolhido": nome_teste,
        "mann_whitney_U": u,
        "mann_whitney_p": p_mw,
    }

    return teste, pressupostos


# =====================================================
# TEXTO INTERPRETATIVO
# =====================================================

def montar_resumo(ics, testes, pressupostos):
    """Gera um texto com hipóteses, resultados e interpretação."""

    linhas = []
    linhas.append("RESULTADOS DA INFERÊNCIA ESTATÍSTICA")
    linhas.append(f"Nível de significância: {ALFA:.0%} | "
                  f"Confiança dos intervalos: {CONFIANCA:.0%}")
    linhas.append("Amostras: 30 municípios por estado (PB e RN)")

    # Item 3 - intervalos de confiança
    linhas.append("\n" + "=" * 70)
    linhas.append("ITEM 3 - INTERVALOS DE CONFIANÇA (rendimento até 1/2 SM)")
    linhas.append("=" * 70)

    for _, r in ics.iterrows():
        linhas.append(
            f"{NOMES[r['estado']]} - {r['abordagem']}: "
            f"{r['estimativa_%']:.2f}% "
            f"(IC 95%: {r['limite_inferior_%']:.2f}% a "
            f"{r['limite_superior_%']:.2f}%)"
        )

    # Testes
    for item, titulo in [(3, "ITEM 3 - TESTE PARA A DIFERENÇA ENTRE PB E RN"),
                         (4, "ITEM 4 - IDEB (ANOS INICIAIS)"),
                         (5, "ITEM 5 - SALÁRIO MÉDIO MENSAL")]:
        linhas.append("\n" + "=" * 70)
        linhas.append(titulo)
        linhas.append("=" * 70)

        if item in (4, 5):
            p = pressupostos[pressupostos["item"] == item].iloc[0]
            linhas.append(
                f"Normalidade (Shapiro-Wilk): PB p = "
                f"{formatar_p(p['shapiro_p_PB'])} | RN p = "
                f"{formatar_p(p['shapiro_p_RN'])}"
            )
            linhas.append(
                f"Variâncias: F = {p['F']:.3f} "
                f"(gl = {p['gl_F']}), p = {formatar_p(p['p_F'])} "
                f"[Levene p = {formatar_p(p['p_Levene'])}] -> "
                f"variâncias {p['variancias'].lower()}"
            )

        for _, t in testes[testes["item"] == item].iterrows():
            gl = "" if pd.isna(t["graus_liberdade"]) else (
                f", gl = {t['graus_liberdade']:.1f}"
            )
            linhas.append(f"\nTeste: {t['teste']}")
            linhas.append(f"H0: {t['H0']}")
            linhas.append(f"H1: {t['H1']}")
            linhas.append(
                f"Estatística = {t['estatistica']:.3f}{gl}, "
                f"p = {formatar_p(t['p_valor'])}, α = {t['alfa']:.2f}"
            )
            linhas.append(
                f"Diferença PB - RN = {t['diferenca_PB_menos_RN']:.3f} "
                f"(IC 95%: {t['ic95_inf']:.3f} a {t['ic95_sup']:.3f})"
            )
            linhas.append(f"Decisão: {t['decisao']}")

        if item in (4, 5):
            p = pressupostos[pressupostos["item"] == item].iloc[0]
            linhas.append(
                f"Verificação (Mann-Whitney): p = "
                f"{formatar_p(p['mann_whitney_p'])}"
            )

    return "\n".join(linhas)


# =====================================================
# EXECUÇÃO DA ANÁLISE
# =====================================================

def executar_inferencia():

    amostras = carregar_amostras()

    # Item 3
    ics = intervalos_confianca(amostras)
    testes = teste_diferenca_proporcoes(amostras)

    # Itens 4 e 5
    pressupostos = []

    for coluna, nome, item in [
        ("ideb", "IDEB (anos iniciais)", 4),
        ("salario_medio", "Salário médio mensal (salários mínimos)", 5),
    ]:
        teste, pressuposto = comparar_medias(amostras, coluna, nome, item)
        testes.append(teste)
        pressupostos.append(pressuposto)

    testes = pd.DataFrame(testes)
    pressupostos = pd.DataFrame(pressupostos)

    # Salvamento
    PASTA_TABELAS.mkdir(parents=True, exist_ok=True)
    PASTA_TESTES.mkdir(parents=True, exist_ok=True)

    ics.to_csv(
        PASTA_TABELAS / "intervalos_confianca.csv",
        index=False, encoding="utf-8-sig"
    )
    testes.to_csv(
        PASTA_TESTES / "testes_hipoteses.csv",
        index=False, encoding="utf-8-sig"
    )
    pressupostos.to_csv(
        PASTA_TESTES / "pressupostos_testes.csv",
        index=False, encoding="utf-8-sig"
    )

    resumo = montar_resumo(ics, testes, pressupostos)
    (PASTA_TESTES / "resumo_inferencia.txt").write_text(
        resumo, encoding="utf-8"
    )

    print(resumo)
    print(f"\nResultados salvos em: {PASTA_TABELAS} e {PASTA_TESTES}")

    return ics, testes, pressupostos


# =====================================================
# EXECUÇÃO DO MÓDULO
# =====================================================

if __name__ == "__main__":

    executar_inferencia()

    print("\nANÁLISE INFERENCIAL CONCLUÍDA COM SUCESSO!")
