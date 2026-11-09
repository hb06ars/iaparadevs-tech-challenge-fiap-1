# FIAP Tech Challenge — Fase 1
## Inteligência Artificial aplicada ao apoio ao diagnóstico médico

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerization-blue.svg)](https://www.docker.com/)

## 1. Sobre o projeto

Este projeto foi desenvolvido como parte do **Tech Challenge — Fase 1**, da FIAP, com o objetivo de aplicar técnicas de Inteligência Artificial e Machine Learning para auxiliar no processo de diagnóstico médico.

A primeira etapa da implementação utiliza Machine Learning supervisionado para classificar tumores de mama como benignos ou malignos, com base em características extraídas de exames.

A solução contempla análise exploratória dos dados, pré-processamento, treinamento e comparação de algoritmos, validação cruzada, avaliação final, interpretabilidade e persistência do modelo treinado.

O projeto foi estruturado com orientação a objetos, separação de responsabilidades e possibilidade de execução local ou por Docker.

Como atividade complementar da fase, também foi desenvolvida uma etapa experimental de visão computacional para classificação de radiografias de tórax com uma Rede Neural Convolucional (CNN). Essa etapa está documentada em notebook separado e não faz parte da aplicação local/Docker descrita neste README.

## 2. Objetivos

- Explorar e compreender os dados utilizados no treinamento.
- Identificar padrões e correlações entre as características.
- Preparar os dados para treinamento dos algoritmos.
- Implementar e comparar diferentes modelos de classificação.
- Avaliar o desempenho utilizando métricas apropriadas.
- Aplicar técnicas de interpretabilidade para compreender as decisões dos modelos.
- Persistir o modelo treinado para utilização posterior.
- Disponibilizar uma estrutura reproduzível de execução.

## 3. Dataset

### Breast Cancer Wisconsin Diagnostic

Foi utilizado o dataset Breast Cancer Wisconsin Diagnostic, disponibilizado no Kaggle.

**Fonte:** https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

Características do dataset:

| Informação | Quantidade |
|---|---:|
| Registros | 569 |
| Características numéricas | 30 |
| Variáveis originais | 32 |
| Diagnósticos benignos | 357 |
| Diagnósticos malignos | 212 |

A variável `diagnosis` representa a classificação:

- `B`: Benigno.
- `M`: Maligno.

A coluna `id` é utilizada para identificação dos registros e não participa do treinamento.

### Dataset de imagens de pneumonia

Foi desenvolvida uma etapa complementar de investigação utilizando imagens de radiografias torácicas para classificação binária entre `NORMAL` e `PNEUMONIA`.

Fonte do dataset:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

O conjunto utilizado no experimento possui 5.216 imagens de treinamento, 16 de validação e 624 de teste. Foi treinada uma CNN com TensorFlow/Keras; o conjunto de teste apresentou acurácia de 90,54%, recall de 95,64%, precisão de 89,88% e F1-score de 92,67% para a classe `PNEUMONIA`.

**Artefatos:** [notebook da CNN](notebooks/pneumonia_cnn.ipynb) e [relatório técnico dos resultados](docs/relatorio_tecnico_cnn_pneumonia.md).

O notebook foi executado no Google Colab e espera encontrar o dataset em `Meu Drive/datasets/chest_xray/`, com as pastas `train`, `val` e `test`, cada uma contendo subpastas `NORMAL` e `PNEUMONIA`. Os resultados da execução são gravados no Google Drive em `pneumonia_cnn_resultados/`; eles não são incluídos no repositório. Para executar, abra o notebook no Colab, disponibilize o dataset nesse caminho e execute as células em ordem.

Esta é uma avaliação acadêmica e experimental: o conjunto de validação é muito pequeno, os resultados não demonstram generalização para outras populações e o modelo não foi integrado ao fluxo da aplicação nem validado para uso clínico. Não deve ser utilizado para diagnóstico ou decisão médica.

O notebook depende das APIs do Google Colab e do TensorFlow/Keras; essa etapa não é executada pelo `src.main`, pelo Docker nem pelas dependências atuais de `requirements.txt`.

**Status:** etapa experimental de visão computacional concluída e documentada.

## 4. Tecnologias utilizadas

| Tecnologia | Aplicação |
|---|---|
| Python 3.12 | Linguagem principal |
| Pandas | Manipulação dos dados |
| NumPy | Operações numéricas |
| Scikit-learn | Modelos de Machine Learning |
| TensorFlow / Keras | Rede Neural Convolucional experimental para imagens |
| Matplotlib | Visualização dos resultados |
| Seaborn | Visualização estatística |
| SHAP | Interpretabilidade dos modelos |
| Joblib | Persistência dos modelos |
| Docker | Containerização da aplicação |
| PyCharm | Ambiente de desenvolvimento |

## 5. Estrutura do projeto

```text
iaparadevs-tech-challenge-fiap-1/
│
├── data/
│   ├── breast_cancer/
│   │   └── breast_cancer.csv
│   └── pneumonia/
│
├── docs/
│   ├── imagens/
│   │   ├── curvas_treinamento_cnn.png
│   │   └── matriz_confusao_cnn.png
│   ├── relatorio_tecnico_cnn_pneumonia.md
│   └── relatorio_tecnico.md
│
├── models/
│
├── notebooks/
│   └── pneumonia_cnn.ipynb
│
├── results/
│   ├── graphics/
│   └── metrics/
│
├── src/
│   ├── exploration/
│   ├── evaluation/
│   ├── models/
│   ├── persistence/
│   ├── pipelines/
│   ├── prediction/
│   ├── preprocessing/
│   ├── web/
│   │   └── index.html
│   ├── api.py
│   └── main.py
│
├── tests/
│
├── .dockerignore
├── Dockerfile
├── requirements.txt
├── requirements-api.txt
└── README.md
```

## 6. Análise exploratória dos dados

A análise exploratória contempla:

- Verificação de valores ausentes.
- Identificação de registros duplicados.
- Verificação de identificadores duplicados.
- Análise dos tipos de dados.
- Estatísticas descritivas.
- Distribuição dos diagnósticos.
- Distribuição das características.
- Matriz de correlação.

Os gráficos são armazenados em `results/graphics/`.

## 7. Pré-processamento

O pré-processamento contempla:

- Separação das variáveis independentes (features) e da variável dependente (target).
- Conversão dos diagnósticos para valores numéricos:
  - Benigno: `0`.
  - Maligno: `1`.
- Remoção das colunas de identificação e diagnóstico do conjunto de features.
- Divisão estratificada dos dados.

### Divisão dos dados

| Conjunto | Registros | Proporção |
|---|---:|---:|
| Treinamento | 398 | 70% |
| Validação | 85 | 15% |
| Teste | 86 | 15% |
| Total | 569 | 100% |

A separação utiliza estratificação para preservar aproximadamente a distribuição das classes nos três conjuntos.

Também são realizadas verificações para garantir que não exista sobreposição de registros entre os conjuntos.

## 8. Modelos implementados

Foram implementados três algoritmos de classificação:

### Regressão Logística

Modelo utilizado como referência para classificação binária.

O pré-processamento inclui padronização das características por meio de `StandardScaler`.

### Random Forest

Algoritmo baseado em um conjunto de árvores de decisão, utilizando votação para determinar a classificação final.

Também é utilizado na análise de importância das características e na interpretação com SHAP.

### Support Vector Machine (SVM)

Algoritmo que busca encontrar uma fronteira de separação entre as classes.

Foi implementado utilizando kernel RBF e padronização das características.

## 9. Avaliação dos modelos

Foram utilizadas as seguintes métricas:

- **Accuracy:** proporção de classificações corretas.
- **Precision:** proporção de previsões positivas que realmente pertencem à classe positiva.
- **Recall:** proporção de registros positivos identificados corretamente.
- **F1-score:** média harmônica entre Precision e Recall.

### Resultados na validação

| Modelo | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Regressão Logística | 96,47% | 100% | 90,63% | 95,08% |
| Random Forest | 96,47% | 100% | 90,63% | 95,08% |
| SVM | 95,29% | 100% | 87,50% | 93,33% |

### Validação cruzada

Foi utilizada validação cruzada estratificada com cinco divisões (Stratified K-Fold).

Resultados médios:

| Modelo | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Regressão Logística | 98,24% | 100% | 95,26% | 97,57% |
| Random Forest | 94,72% | 94,17% | 91,82% | 92,82% |
| SVM | 97,23% | 97,87% | 94,55% | 96,15% |

A Regressão Logística apresentou o melhor resultado médio nas quatro métricas avaliadas.

### Avaliação final

Após a comparação dos modelos e a validação cruzada, a Regressão Logística foi selecionada para o treinamento final.

O modelo foi treinado utilizando os conjuntos de treinamento e validação, totalizando 483 registros. O conjunto de teste permaneceu separado, com 86 registros.

| Métrica | Resultado |
|---|---:|
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1-score | 100% |

Matriz de confusão:

| | Previsto benigno | Previsto maligno |
|---|---:|---:|
| Real benigno | 54 | 0 |
| Real maligno | 0 | 32 |

**Observação:** os resultados representam o desempenho observado neste conjunto de teste específico. O tamanho da amostra é limitado e os resultados não devem ser interpretados como garantia de desempenho em populações externas ou como validação para utilização clínica.

## 10. Interpretabilidade

Foram utilizadas duas abordagens complementares.

### Feature Importance

A importância das características foi extraída do modelo Random Forest.

As cinco características de maior importância foram:

1. `perimeter_worst`
2. `area_worst`
3. `concave points_mean`
4. `concave points_worst`
5. `radius_worst`

### SHAP

A biblioteca SHAP foi utilizada para analisar a contribuição das características nas previsões do modelo Random Forest.

Foram gerados gráficos:

- Beeswarm: apresenta a distribuição das contribuições das características.
- Importance: apresenta a importância global das características segundo os valores SHAP.

Essas técnicas auxiliam na compreensão do comportamento dos modelos, mas não demonstram relações causais nem substituem a avaliação de especialistas.

## 11. Persistência e utilização do modelo

O modelo final é persistido utilizando Joblib.

Arquivo gerado:

```text
models/breast_cancer_logistic_regression.joblib
```

Também foi implementada a classe `BreastCancerPredictor`, responsável por:

- Carregar o modelo persistido.
- Validar os dados de entrada.
- Verificar características ausentes ou adicionais.
- Realizar previsões.
- Retornar a classificação e as probabilidades estimadas.

As classificações retornadas são:

- `Benigno`
- `Maligno`

Essa funcionalidade demonstra a possibilidade de reutilização do modelo sem a necessidade de executar novamente todo o treinamento.

## 12. Como executar o projeto localmente

### Pré-requisitos

- Python 3.12.
- Git.
- Dataset Breast Cancer Wisconsin Diagnostic.

### Clonar o repositório

```bash
git clone https://github.com/hb06ars/iaparadevs-tech-challenge-fiap-1.git
```

Acesse a pasta:

```bash
cd iaparadevs-tech-challenge-fiap-1
```

### Criar ambiente virtual

Windows PowerShell:

```powershell
python -m venv .venv
```

Ativar:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Instalar dependências

```powershell
python -m pip install --upgrade pip
```

```powershell
pip install -r requirements.txt
```

### Disponibilizar o dataset

Coloque o arquivo `breast_cancer.csv` no seguinte caminho:

```text
data/breast_cancer/breast_cancer.csv
```

### Executar

Na raiz do projeto:

```powershell
python -m src.main
```

### Iniciar a aplicação web de demonstração

> **Observação:** antes de iniciar a aplicação web, execute `python -m src.main` na raiz do projeto. Essa etapa treina o modelo e gera `models/breast_cancer_logistic_regression.joblib`, que é carregado pela API para fazer as previsões. Se a API for iniciada sem esse arquivo, as análises não poderão ser realizadas.

Depois que o pipeline concluir, instale as dependências da API:

```powershell
pip install -r requirements-api.txt
```

Inicie o servidor local; o navegador será aberto automaticamente quando a aplicação estiver pronta:

```powershell
python -m src.web_runner
```

A aplicação também pode ser iniciada sem abertura automática do navegador com `python -m uvicorn src.api:app --reload`. Acesse `http://127.0.0.1:8000` para enviar um arquivo `.csv`. A API disponibiliza ainda a documentação interativa em `http://127.0.0.1:8000/docs` e um CSV de exemplo na própria página. O arquivo deve usar vírgulas para separar colunas e pontos para separar casas decimais. Ele deve ter uma linha de cabeçalho com as 30 características numéricas do modelo; as colunas `id` e `diagnosis` podem ser incluídas e serão ignoradas.

Esta interface é uma demonstração acadêmica: as classificações e probabilidades são experimentais, não constituem diagnóstico e não devem ser usadas para decisões clínicas.

## 13. Como executar utilizando Docker

### Pré-requisitos

- Docker Desktop instalado e em execução.

### Construir a imagem

Na raiz do projeto:

```powershell
docker build -t fiap-breast-cancer .
```

### Executar o container

No PowerShell:

```powershell
docker run --rm `
  -v "${PWD}/data/breast_cancer:/app/data/breast_cancer" `
  -v "${PWD}/results:/app/results" `
  -v "${PWD}/models:/app/models" `
  fiap-breast-cancer
```

O dataset é disponibilizado ao container por meio de volume, assim como os diretórios de resultados e modelos.

Dessa maneira, os arquivos gerados permanecem disponíveis na máquina local após a execução.

Após essa execução concluir e salvar o modelo em `models/`, inicie a interface web com:

```powershell
docker run --rm -p 8000:8000 `
  -v "${PWD}/models:/app/models" `
  -v "${PWD}/data/breast_cancer:/app/data/breast_cancer" `
  --entrypoint python `
  fiap-breast-cancer -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Acesse `http://127.0.0.1:8000`. A montagem dos volumes disponibiliza o modelo treinado e o dataset usado para gerar a planilha de exemplo.

## 14. Testes

Foram desenvolvidos testes para verificar o carregamento e a utilização do modelo persistido, assim como o funcionamento da classe responsável pelas previsões.

Antes de executar os testes, execute o pipeline principal para gerar o modelo:

```powershell
python -m src.main
```

Executar os testes a partir da raiz:

```powershell
python -m tests.test_breast_cancer_predictor
```

```powershell
python -m tests.test_model_loading
```

## 15. Resultados gerados

Os resultados são armazenados nos seguintes diretórios:

**Gráficos — `results/graphics/`**

- Distribuição dos diagnósticos.
- Distribuição das características.
- Matriz de correlação.
- Matrizes de confusão.
- Feature Importance.
- SHAP Beeswarm.
- SHAP Importance.

**Métricas — `results/metrics/`**

- Comparação dos modelos.
- Resultados da validação cruzada.
- Avaliação final do conjunto de teste.

## 16. Considerações finais

O desenvolvimento permitiu aplicar diferentes técnicas de Machine Learning em um problema de classificação relacionado à área da saúde.

A comparação entre algoritmos, a validação cruzada e a análise de interpretabilidade contribuíram para compreender melhor o desempenho e o comportamento dos modelos.

A solução foi estruturada para permitir execução local e em container Docker, além de possibilitar a persistência e reutilização do modelo treinado.

É importante destacar que a aplicação possui caráter acadêmico e experimental. Os resultados não representam diagnóstico médico e não devem ser utilizados isoladamente para decisões clínicas. A decisão diagnóstica deve permanecer sob responsabilidade de profissionais de saúde qualificados.

---

**Projeto acadêmico desenvolvido para o Tech Challenge — Fase 1, FIAP.**


## Criadores:
- José Henrique Siqueira Brandão (hb06ars)