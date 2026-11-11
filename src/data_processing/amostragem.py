
from pathlib import Path
import pandas as pd

from src.data_processing.carregar_dados import carregar_dados


# =====================================================
# CONFIGURAÇÕES
# =====================================================

PASTA_PROJETO = Path(__file__).resolve().parents[2]

PASTA_PROCESSADOS = PASTA_PROJETO / "data" / "processed"

TAMANHO_AMOSTRA = 30

SEMENTE_ALEATORIA = 42


# =====================================================
# FUNÇÃO DE AMOSTRAGEM ALEATÓRIA SIMPLES
# =====================================================

def selecionar_amostra(dados, estado):

    """
    Seleciona uma amostra aleatória simples de municípios.

    Parâmetros:
        dados: DataFrame com a população do estado.
        estado: sigla do estado.

    Retorna:
        DataFrame contendo os municípios selecionados.
    """

    if len(dados) < TAMANHO_AMOSTRA:
        raise ValueError(
            f"A população de {estado} possui menos "
            f"de {TAMANHO_AMOSTRA} municípios."
        )

    # Seleção aleatória simples sem reposição
    amostra = dados.sample(
        n=TAMANHO_AMOSTRA,
        replace=False,
        random_state=SEMENTE_ALEATORIA
    )

    # Reinicia os índices
    amostra = amostra.reset_index(drop=True)

    return amostra


# =====================================================
# FUNÇÃO PARA SALVAR AS AMOSTRAS
# =====================================================

def salvar_amostra(amostra, estado):

    """
    Salva a amostra selecionada em formato CSV.
    """

    PASTA_PROCESSADOS.mkdir(
        parents=True,
        exist_ok=True
    )

    nome_arquivo = f"amostra_{estado.lower()}.csv"

    caminho = PASTA_PROCESSADOS / nome_arquivo

    amostra.to_csv(
        caminho,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nAmostra salva em: {caminho}")


# =====================================================
# FUNÇÃO PARA EXIBIR OS RESULTADOS
# =====================================================

def exibir_amostra(amostra, estado):

    print("\n" + "=" * 55)
    print(f"AMOSTRAGEM ALEATÓRIA SIMPLES - {estado}")
    print("=" * 55)

    print(f"Quantidade selecionada: {len(amostra)}")

    print("\nMunicípios selecionados:")

    print(
        amostra[
            ["municipio", "ideb", "percentual_baixa_renda",
             "salario_medio"]
        ].to_string(index=False)
    )

    print("\nValores ausentes na amostra:")

    print(amostra.isnull().sum())


# =====================================================
# FUNÇÃO PRINCIPAL DA AMOSTRAGEM
# =====================================================

def executar_amostragem():

    # Carrega as populações completas
    dados_pb, dados_rn = carregar_dados()

    # Seleciona 30 municípios de cada estado
    amostra_pb = selecionar_amostra(dados_pb, "PB")

    amostra_rn = selecionar_amostra(dados_rn, "RN")

    # Exibe os resultados
    exibir_amostra(amostra_pb, "PARAÍBA")

    exibir_amostra(amostra_rn, "RIO GRANDE DO NORTE")

    # Salva os arquivos CSV
    salvar_amostra(amostra_pb, "PB")

    salvar_amostra(amostra_rn, "RN")

    return amostra_pb, amostra_rn


# =====================================================
# EXECUÇÃO DO MÓDULO
# =====================================================

if __name__ == "__main__":

    amostra_pb, amostra_rn = executar_amostragem()

    print("\n" + "=" * 55)
    print("AMOSTRAGEM CONCLUÍDA COM SUCESSO!")
    print("=" * 55)
