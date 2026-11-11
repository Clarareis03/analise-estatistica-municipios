
from pathlib import Path
import pandas as pd

from src.data_processing.carregar_dados import carregar_dados


# =====================================================
# CONFIGURAÇÕES
# =====================================================

PASTA_PROJETO = Path(__file__).resolve().parents[2]

PASTA_TABELAS = PASTA_PROJETO / "outputs" / "tabelas"

VARIAVEIS = {
    "ideb": "IDEB",
    "percentual_baixa_renda": "Rendimento de até meio salário mínimo (%)",
    "salario_medio": "Salário médio mensal"
}


# =====================================================
# CÁLCULO DAS ESTATÍSTICAS DESCRITIVAS
# =====================================================

def calcular_estatisticas(dados, estado, tipo):

    """
    Calcula as estatísticas descritivas das variáveis.

    ddof = 0: variância populacional.
    ddof = 1: variância amostral.
    """

    # Define o divisor conforme o conjunto analisado
    ddof = 0 if tipo == "População" else 1

    resultados = []

    for coluna, nome_variavel in VARIAVEIS.items():

        # Remove somente os valores ausentes da variável analisada
        valores = dados[coluna].dropna()

        # Identifica a moda observada
        frequencias = valores.value_counts()

        if frequencias.empty or frequencias.max() == 1:
            moda = "Sem moda"
        else:
            modas = frequencias[frequencias == frequencias.max()].index
            moda = ", ".join(str(valor) for valor in modas.tolist())

        # Calcula as estatísticas
        resultado = {
            "estado": estado,
            "conjunto": tipo,
            "variavel": nome_variavel,
            "n": len(valores),
            "media": valores.mean(),
            "mediana": valores.median(),
            "moda": moda,
            "variancia": valores.var(ddof=ddof),
            "desvio_padrao": valores.std(ddof=ddof),
            "minimo": valores.min(),
            "q1": valores.quantile(0.25),
            "q3": valores.quantile(0.75),
            "maximo": valores.max()
        }

        resultados.append(resultado)

    return pd.DataFrame(resultados)


# =====================================================
# EXIBIÇÃO DOS RESULTADOS
# =====================================================

def exibir_estatisticas(resultados):

    for (estado, conjunto), grupo in resultados.groupby(
        ["estado", "conjunto"],
        sort=False
    ):

        print("\n" + "=" * 70)
        print(f"ESTATÍSTICAS DESCRITIVAS - {estado} - {conjunto}")
        print("=" * 70)

        print(
            grupo.drop(
                columns=["estado", "conjunto"]
            ).round(3).to_string(index=False)
        )


# =====================================================
# SALVAMENTO DOS RESULTADOS
# =====================================================

def salvar_estatisticas(resultados):

    PASTA_TABELAS.mkdir(
        parents=True,
        exist_ok=True
    )

    caminho = PASTA_TABELAS / "estatisticas_descritivas.csv"

    resultados.to_csv(
        caminho,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nResultados salvos em: {caminho}")


# =====================================================
# EXECUÇÃO DA ANÁLISE
# =====================================================

def executar_analise():

    # Carrega as populações completas
    dados_pb, dados_rn = carregar_dados()

    # Carrega as amostras já selecionadas
    amostra_pb = pd.read_csv(
        PASTA_PROJETO / "data" / "processed" / "amostra_pb.csv"
    )

    amostra_rn = pd.read_csv(
        PASTA_PROJETO / "data" / "processed" / "amostra_rn.csv"
    )

    # Calcula as estatísticas dos quatro conjuntos
    resultados = pd.concat([

        calcular_estatisticas(
            dados_pb, "PB", "População"
        ),

        calcular_estatisticas(
            dados_rn, "RN", "População"
        ),

        calcular_estatisticas(
            amostra_pb, "PB", "Amostra"
        ),

        calcular_estatisticas(
            amostra_rn, "RN", "Amostra"
        )

    ], ignore_index=True)

    # Exibe e salva os resultados
    exibir_estatisticas(resultados)

    salvar_estatisticas(resultados)

    return resultados


# =====================================================
# EXECUÇÃO DO MÓDULO
# =====================================================

if __name__ == "__main__":

    resultados = executar_analise()

    print("\nANÁLISE DESCRITIVA CONCLUÍDA COM SUCESSO!")
