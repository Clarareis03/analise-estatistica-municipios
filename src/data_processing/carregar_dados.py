
from pathlib import Path
import pandas as pd


# CONFIGURAÇÃO DOS CAMINHOS

# Localiza automaticamente a pasta principal do projeto
PASTA_PROJETO = Path(__file__).resolve().parents[2]

PASTA_DADOS = PASTA_PROJETO / "data" / "raw"

ARQUIVO_PB = PASTA_DADOS / "dados_paraiba.xlsx"
ARQUIVO_RN = PASTA_DADOS / "dados_rn.xlsx"


# PADRONIZAÇÃO DAS COLUNAS

COLUNAS_PADRONIZADAS = [
    "municipio",
    "ideb",
    "percentual_baixa_renda",
    "salario_medio"
]

# FUNÇÃO PARA CARREGAR E PREPARAR OS DADOS

def carregar_planilha(caminho, estado):

    """
    Carrega uma planilha Excel e prepara seus dados.

    Parâmetros:
        caminho: caminho do arquivo Excel.
        estado: sigla do estado (PB ou RN).

    Retorna:
        DataFrame com os dados padronizados.
    """

    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}"
        )

    # Leitura da planilha
    dados = pd.read_excel(caminho)

    # Verificação da estrutura original
    if len(dados.columns) != 4:
        raise ValueError(
            f"A planilha {estado} deveria possuir 4 colunas."
        )

    # Padronização dos nomes das colunas
    dados.columns = COLUNAS_PADRONIZADAS

    # Remoção de espaços extras nos nomes dos municípios
    dados["municipio"] = (
        dados["municipio"].astype("string").str.strip()
    )

    # Conversão das variáveis quantitativas
    colunas_numericas = [
        "ideb",
        "percentual_baixa_renda",
        "salario_medio"
    ]

    for coluna in colunas_numericas:
        dados[coluna] = pd.to_numeric(
            dados[coluna],
            errors="coerce"
        )

    # Identificação do estado
    dados["estado"] = estado

    return dados


# FUNÇÃO PARA CARREGAR OS DOIS ESTADOS

def carregar_dados():

    """
    Carrega as planilhas da Paraíba e do Rio Grande do Norte.

    Retorna:
        dados_pb: DataFrame da Paraíba.
        dados_rn: DataFrame do Rio Grande do Norte.
    """

    dados_pb = carregar_planilha(ARQUIVO_PB, "PB")

    dados_rn = carregar_planilha(ARQUIVO_RN, "RN")

    return dados_pb, dados_rn



# FUNÇÃO PARA VERIFICAR A QUALIDADE DOS DADOS

def verificar_dados(dados, estado):

    print(f"\n{'=' * 50}")
    print(f"VERIFICAÇÃO DOS DADOS - {estado}")
    print("=" * 50)

    # Quantidade de municípios
    print(f"Total de municípios: {len(dados)}")

    # Estrutura do DataFrame
    print("\nInformações das variáveis:")
    dados.info()

    # Verificação de valores ausentes
    print("\nValores ausentes:")
    print(dados.isnull().sum())

    # Verificação de municípios duplicados
    duplicados = dados["municipio"].duplicated().sum()

    print(f"\nMunicípios duplicados: {duplicados}")

    # Estatísticas iniciais
    print("\nResumo estatístico:")
    print(dados.describe().round(2))

# IDENTIFICAÇÃO DE MUNICÍPIOS SEM IDEB

def verificar_ideb_ausente(dados, estado):

    municipios_sem_ideb = dados[
        dados["ideb"].isnull()
    ]

    print(f"\nMunicípios sem informação de IDEB - {estado}")

    if municipios_sem_ideb.empty:
        print("Nenhum município encontrado.")
    else:
        print(
            municipios_sem_ideb["municipio"].to_string(
                index=False
            )
        )

        print(
            f"\nTotal: {len(municipios_sem_ideb)} municípios"
        )



# EXECUÇÃO PARA TESTE DO MÓDULO

if __name__ == "__main__":

    dados_pb, dados_rn = carregar_dados()

    verificar_dados(dados_pb, "PARAÍBA")

    verificar_dados(dados_rn, "RIO GRANDE DO NORTE")

    verificar_ideb_ausente(dados_pb, "PB")

    verificar_ideb_ausente(dados_rn, "RN")
