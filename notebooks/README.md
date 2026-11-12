# IC e teste de proporções: PB x RN

Análise estatística comparando a proporção de pessoas com rendimento nominal mensal per capita de até 1/2 salário mínimo na **Paraíba** e no **Rio Grande do Norte**, a partir de amostras de 30 municípios de cada estado.

## Dados

- `data/amostra_pb.csv` e `data/amostra_rn.csv`
- Colunas: `municipio`, `ideb`, `percentual_baixa_renda`, `salario_medio`, `estado`
- Estimativa da proporção: média de `percentual_baixa_renda` (convertida para proporção). Cada município tem peso igual.

## Como rodar

```bash
pip install pandas numpy scipy
python analise.py
```

## Resultados

### IC de 95% para a proporção

| Estado | n | p̂ | Erro-padrão | IC 95% |
|---|---|---|---|---|
| PB | 30 | 0,5004 | 0,0913 | [0,3215 ; 0,6794] |
| RN | 30 | 0,4911 | 0,0913 | [0,3122 ; 0,6700] |

### Teste de hipóteses (duas proporções)

- **H0:** p_PB = p_RN
- **H1:** p_PB ≠ p_RN (bilateral)
- **Nível de significância:** α = 5% (z crítico = ±1,96)
- **Estatística:** z = 0,072; p-valor = 0,942
- **Conclusão:** não rejeita H0. Não há evidência de diferença significativa entre as proporções. O IC 95% da diferença, [−24,4 ; +26,2] p.p., contém o zero.

## Observações

- Com n = 30, o erro-padrão da proporção é grande, por isso os ICs são largos.
- Os municípios têm peso igual, independentemente da população.
- Conferência com teste t de Welch sobre os percentuais municipais: p = 0,38 (mesma conclusão).
- A coluna `ideb` tem dois valores faltantes (Baía da Traição/PB e Pedra Grande/RN); não afetam esta análise.
