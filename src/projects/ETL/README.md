# Projeto ETL: Sistema de Transações de Veículos

**Disciplina:** Ciência de Dados
**Instituição:** Centro Universitário de Brasília (CEUB) - 2025
**Professor:** Dan Lopes
**Autor:** Jess Forster

---

## 1. Visão Geral do Projeto

Este projeto implementa um pipeline completo de Extração, Transformação e Carga (ETL) em Python utilizando a biblioteca Pandas. O objetivo principal é consolidar dados brutos provenientes de três fontes distintas (transações, clientes e veículos) em um único conjunto de dados limpo, consistente e pronto para análises exploratórias ou modelagem preditiva.

A arquitetura do projeto segue o modelo Medallion (Bronze/Silver/Gold):
- **Bronze (Raw):** Dados brutos extraídos diretamente dos arquivos CSV originais.
- **Silver (Processed):** Dados limpos, padronizados e com tipagem correta, mantidos em tabelas individuais.
- **Gold (Output):** Tabela integrada (denormalizada) unindo as três fontes, enriquecida e validada para uso analítico.

## 2. Fontes de Dados

Os dados originais apresentam diversos desafios comuns no mundo real, como inconsistência semântica, valores nulos, formatos de data variados e duplicatas.

| Arquivo | Descrição | Registros | Colunas |
|---------|-----------|-----------|---------|
| `transacoes.csv` | Registros de operações financeiras (compra/venda) | 571 | 11 |
| `clientes.csv` | Cadastro de clientes Pessoa Física e Jurídica | 536 | 14 |
| `veiculos.csv` | Cadastro de veículos no estoque | 536 | 15 |

## 3. Pipeline de Transformação (ETL)

O script principal `src/etl_pipeline.py` executa as seguintes etapas:

### 3.1 Limpeza de Transações
- Remoção de 20 registros duplicados.
- Padronização do campo `status_transacao` (ex: "concluída", "CONCLUIDA" -> "Concluída").
- Padronização do campo `tipo_operacao` ("venda", "vend", "VENDA" -> "Venda").
- Limpeza do campo `forma_pagamento` e `desconto` (remoção de "R$" e conversão para numérico).
- Tratamento avançado de datas no campo `data_transacao`, suportando múltiplos formatos e timestamps.
- Tratamento de valores transacionais iguais a zero (convertidos para nulo).

### 3.2 Limpeza de Clientes
- Remoção de 15 registros duplicados.
- Padronização de gêneros ("m", "Masculino" -> "M").
- Limpeza e formatação de telefones (aplicação de máscara padrão `(XX) XXXXX-XXXX`).
- Padronização de e-mails para minúsculas e tratamento de "nao informado".
- Conversão da `renda_mensal_faturamento` para numérico flutuante.
- Padronização de nomes e cidades (Title Case) e estados (Upper Case sem pontos).

### 3.3 Limpeza de Veículos
- Remoção de 15 registros duplicados.
- Padronização da `categoria` ("suv", "sv", "S.U.V" -> "SUV").
- Padronização do `status_estoque` ("Vendio", "VENDIDO" -> "Vendido").
- Limpeza de quilometragem (remoção de valores negativos ou absurdos > 500.000).
- Padronização de placas (Upper Case sem espaços).
- Identificação de anomalias financeiras (preço de custo maior que preço de venda).

### 3.4 Integração e Carga
As três tabelas limpas são unidas utilizando operações de `LEFT JOIN` a partir da tabela de transações, utilizando `id_cliente` e `id_veiculo` como chaves. O resultado final gera o arquivo `dados_integrados_gold.csv` com 550 registros e 27 colunas selecionadas.

## 4. Validação de Qualidade

O pipeline inclui uma etapa automatizada de testes de qualidade:
- Verificação de unicidade das chaves primárias.
- Contagem de valores nulos em campos críticos (como `valor_transacao`).
- Verificação de integridade referencial (identificação de transações com clientes ou veículos órfãos).
- Detecção de valores negativos inconsistentes.
- Detecção de margens de lucro negativas.

## 5. Análise Exploratória (EDA)

O script `src/eda_visualizations.py` gera um conjunto de gráficos analisando a base Gold. As visualizações estão salvas no diretório `output/`:

1. **01_distribuicao_status_tipo.png:** Distribuição das transações por status e tipo de operação.
2. **02_volume_temporal.png:** Evolução do volume financeiro e quantidade de transações ao longo dos meses.
3. **03_pagamento_filial.png:** Preferências de forma de pagamento e volume por filial.
4. **04_analise_veiculos.png:** Top marcas, distribuição por categoria, status do estoque e preços por categoria.
5. **05_analise_clientes.png:** Proporção PF/PJ, distribuição de gênero, top estados e histograma de renda.
6. **06_metricas_integradas.png:** Valor médio transacionado por forma de pagamento e por marca do veículo.

## 6. Estrutura do Repositório

```text
ProjetoETL/
├── data/
│   ├── raw/                  # Arquivos CSV originais brutos
│   └── processed/            # Arquivos CSV limpos (Camada Silver)
├── docs/
│   └── README.md             # Esta documentação
├── src/
│   ├── etl_pipeline.py       # Script principal de ETL
│   └── eda_visualizations.py # Script de análise exploratória
└── output/
    ├── dados_integrados_gold.csv # Tabela final integrada
    └── *.png                 # Gráficos gerados pela EDA
```

## 7. Como Executar

Para rodar o pipeline completo:
```bash
python3 src/etl_pipeline.py
```

Para gerar as visualizações:
```bash
python3 src/eda_visualizations.py
```
