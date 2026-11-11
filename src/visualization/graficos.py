"""Gráficos do salário médio mensal dos trabalhadores formais (PB e RN).
  1. Histogramas (um por estado, mesma escala para comparação);
  2. Boxplots;
  3. Tabela com os valores atípicos (critério 1,5 x IQR).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PASTA_PROJETO = Path(__file__).resolve().parents[2]
PASTA_PROCESSADOS = PASTA_PROJETO / "data" / "processed"
PASTA_GRAFICOS = PASTA_PROJETO / "outputs" / "graficos"
PASTA_TABELAS = PASTA_PROJETO / "outputs" / "tabelas"

COLUNA = "salario_medio"
ROTULO = "Salário médio mensal (salários mínimos)"
CORES = {"PB": "#2a9d8f", "RN": "#e76f51"}
NOMES = {"PB": "Paraíba", "RN": "Rio Grande do Norte"}

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})



def carregar_amostras():
    pb = pd.read_csv(PASTA_PROCESSADOS / "amostra_pb.csv", encoding="utf-8-sig")
    rn = pd.read_csv(PASTA_PROCESSADOS / "amostra_rn.csv", encoding="utf-8-sig")
    return {"PB": pb, "RN": rn}

# HISTOGRAMAS

def gerar_histogramas(amostras):
    """Dois histogramas com mesmos intervalos (bins) e mesmo eixo y."""
    todos = pd.concat([a[COLUNA] for a in amostras.values()]).dropna()
    bins = np.arange(
        np.floor(todos.min() * 5) / 5,
        np.ceil(todos.max() * 5) / 5 + 0.2,
        0.2,
    )

    fig, eixos = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    for eixo, (uf, dados) in zip(eixos, amostras.items()):
        v = dados[COLUNA].dropna()
        eixo.hist(v, bins=bins, color=CORES[uf], edgecolor="white")
        eixo.axvline(v.mean(), color="black", linestyle="--", linewidth=1.5,
                     label=f"Média = {v.mean():.2f}")
        eixo.axvline(v.median(), color="black", linestyle=":", linewidth=1.5,
                     label=f"Mediana = {v.median():.2f}")
        eixo.set_title(f"{NOMES[uf]} (n = {len(v)})")
        eixo.set_xlabel(ROTULO)
        eixo.legend(frameon=False)

    eixos[0].set_ylabel("Número de municípios")
    fig.suptitle("Distribuição do salário médio mensal dos trabalhadores formais",
                 fontweight="bold")
    fig.tight_layout()

    caminho = PASTA_GRAFICOS / "histogramas_salario_medio.png"
    fig.savefig(caminho, dpi=200)
    plt.close(fig)
    print(f"Salvo: {caminho}")



# BOXPLOTS

def gerar_boxplots(amostras):
    fig, eixo = plt.subplots(figsize=(7, 5.5))

    dados = [amostras[uf][COLUNA].dropna() for uf in amostras]
    caixas = eixo.boxplot(
        dados,
        tick_labels=[NOMES[uf] for uf in amostras],
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "black", "linewidth": 2},
        flierprops={"marker": "o", "markerfacecolor": "white",
                    "markeredgecolor": "black", "markersize": 7},
    )
    for caixa, uf in zip(caixas["boxes"], amostras):
        caixa.set_facecolor(CORES[uf])
        caixa.set_alpha(0.85)

    # médias (losango)
    eixo.scatter(
        [1, 2], [d.mean() for d in dados],
        marker="D", color="black", zorder=3, label="Média",
    )

    eixo.set_ylabel(ROTULO)
    eixo.set_title("Boxplot do salário médio mensal (amostras)",
                   fontweight="bold")
    eixo.legend(frameon=False)
    fig.tight_layout()

    caminho = PASTA_GRAFICOS / "boxplots_salario_medio.png"
    fig.savefig(caminho, dpi=200)
    plt.close(fig)
    print(f"Salvo: {caminho}")


# VALORES ATÍPICOS (1,5 x IQR) E RESUMO

def identificar_atipicos(amostras):
    linhas = []
    for uf, df in amostras.items():
        v = df[COLUNA].dropna()
        q1, q3 = v.quantile(0.25), v.quantile(0.75)
        iqr = q3 - q1
        inf, sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        fora = df[(df[COLUNA] < inf) | (df[COLUNA] > sup)]
        print(f"\n{uf}: Q1={q1:.3f} Q3={q3:.3f} IQR={iqr:.3f} "
              f"limites=[{inf:.3f}; {sup:.3f}]")
        if fora.empty:
            print("  Sem valores atípicos.")
        for _, r in fora.iterrows():
            print(f"  Atípico: {r['municipio']} = {r[COLUNA]}")
            linhas.append({"estado": uf, "municipio": r["municipio"],
                           "salario_medio": r[COLUNA],
                           "limite_inferior": inf, "limite_superior": sup})

    PASTA_TABELAS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(linhas).to_csv(PASTA_TABELAS / "valores_atipicos_salario.csv",
                                index=False, encoding="utf-8-sig")




def executar_graficos():
    PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)
    amostras = carregar_amostras()
    gerar_histogramas(amostras)
    gerar_boxplots(amostras)
    identificar_atipicos(amostras)


if __name__ == "__main__":
    executar_graficos()
