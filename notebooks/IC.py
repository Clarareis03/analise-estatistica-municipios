"""
IC de 95% para a proporção de pessoas com rendimento per capita de até 1/2 salário mínimo
(PB e RN) e teste de hipóteses para a diferença entre as duas proporções.

Uso:  python analise.py
Requer: pandas, numpy, scipy
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA = Path(__file__).parent / "data"
ALPHA = 0.05


def carregar(nome: str) -> pd.Series:
    df = pd.read_csv(DATA / nome, encoding="utf-8-sig")
    return df["percentual_baixa_renda"] / 100  # % -> proporção


def ic_proporcao(p: float, n: int, alpha: float = ALPHA):
    z = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(p * (1 - p) / n)
    return p, se, p - z * se, p + z * se


def main():
    pb = carregar("amostra_pb.csv")
    rn = carregar("amostra_rn.csv")
    n1, n2 = len(pb), len(rn)
    p1, p2 = pb.mean(), rn.mean()

    print("=== 1) IC 95% para a proporção ===")
    for nome, p, n in [("PB", p1, n1), ("RN", p2, n2)]:
        _, se, li, ls = ic_proporcao(p, n)
        print(f"{nome}: n={n}  p={p:.4f}  EP={se:.4f}  IC95% = [{li:.4f} ; {ls:.4f}]")

    print("\n=== 2) Teste de duas proporções ===")
    print("H0: p_PB = p_RN   |   H1: p_PB != p_RN   |   alpha = 5%")
    p_comb = (p1 * n1 + p2 * n2) / (n1 + n2)
    se0 = np.sqrt(p_comb * (1 - p_comb) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se0
    z_crit = stats.norm.ppf(1 - ALPHA / 2)
    p_valor = 2 * (1 - stats.norm.cdf(abs(z)))
    print(f"p combinada = {p_comb:.4f}  EP = {se0:.4f}")
    print(f"z = {z:.4f}   z crítico = ±{z_crit:.2f}   p-valor = {p_valor:.4f}")

    se_dif = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p1 - p2
    print(f"IC95% da diferença = [{d - z_crit * se_dif:.4f} ; {d + z_crit * se_dif:.4f}]")

    if abs(z) > z_crit:
        print("Conclusão: rejeita H0 -> há diferença significativa.")
    else:
        print("Conclusão: não rejeita H0 -> sem evidência de diferença significativa.")

    print("\n=== Conferência: teste t de Welch nos percentuais municipais ===")
    t = stats.ttest_ind(pb, rn, equal_var=False)
    print(f"t = {t.statistic:.4f}  p-valor = {t.pvalue:.4f}")


if __name__ == "__main__":
    main()
