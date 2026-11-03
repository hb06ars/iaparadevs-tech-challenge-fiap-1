# 1. Diagnóstico de câncer de mama utilizando Machine Learning

## 1.1 Objetivo

O objetivo desta etapa é desenvolver modelos de classificação capazes de auxiliar na identificação de tumores benignos e malignos a partir de características extraídas de exames de câncer de mama.

A solução utiliza algoritmos de Machine Learning supervisionado para identificar padrões nos dados e realizar previsões sobre o diagnóstico.

O sistema tem finalidade de apoio à decisão, não substituindo a avaliação de profissionais da saúde.

## 1.2 Dataset utilizado

Foi utilizado o dataset Breast Cancer Wisconsin Diagnostic, disponibilizado publicamente no Kaggle.

Fonte: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

A base possui 569 registros e 32 colunas originais, incluindo:

- Identificador do registro (`id`);
- Diagnóstico (`diagnosis`);
- 30 características numéricas relacionadas às células tumorais.

A variável de diagnóstico possui duas classes:

- B (Benigno): 357 registros;
- M (Maligno): 212 registros.

Para a modelagem, a variável de diagnóstico foi convertida para representação numérica:

- Benigno: 0;
- Maligno: 1.

A coluna `id` foi removida, pois representa apenas um identificador e não possui valor preditivo relevante para o problema.

## 1.3 Exploração e pré-processamento dos dados

Inicialmente, foi realizada uma análise exploratória para compreender a estrutura e a qualidade dos dados.

Foram verificadas:

- Dimensões da base;
- Tipos das variáveis;
- Valores ausentes;
- Registros duplicados;
- Identificadores duplicados;
- Estatísticas descritivas;
- Distribuição dos diagnósticos;
- Correlação entre as características.

Não foram identificados valores ausentes ou registros duplicados na base utilizada.

Também foram geradas visualizações para facilitar a interpretação das distribuições e das correlações entre as características.

Após a exploração, os dados foram separados em variáveis independentes (features) e variável dependente (target).

## 1.4 Divisão dos conjuntos de dados

Para avaliar os modelos de maneira adequada, a base foi dividida em três conjuntos:

| Conjunto | Registros | Proporção |
|---|---:|---:|
| Treinamento | 398 | 70% |
| Validação | 85 | 15% |
| Teste | 86 | 15% |
| **Total** | **569** | **100%** |

Foi utilizada a função `train_test_split`, com `random_state=42` e estratificação pela variável de diagnóstico (`stratify`).

A estratificação busca preservar a proporção entre as classes benignas e malignas nos diferentes conjuntos.

O conjunto de treinamento foi utilizado para ajustar os modelos. O conjunto de validação foi utilizado para comparar as alternativas. Após a seleção da Regressão Logística, os conjuntos de treinamento e validação foram reunidos para o treinamento final.

O conjunto de teste permaneceu isolado até a avaliação final.

## 1.5 Modelos utilizados

Foram implementados três algoritmos de classificação:

### Regressão Logística

Algoritmo de classificação que estima a probabilidade de uma observação pertencer a determinada classe.

Foi utilizado o `StandardScaler` para padronizar as características numéricas, seguido do classificador `LogisticRegression`.

### Random Forest

Algoritmo baseado em um conjunto de árvores de decisão. Cada árvore contribui para a classificação final, permitindo capturar relações mais complexas entre as características.

Foi configurado com 200 árvores (`n_estimators=200`).

### Support Vector Machine (SVM)

Algoritmo que busca encontrar uma fronteira de separação entre as classes.

Foi utilizado o kernel RBF, que permite representar fronteiras de decisão não lineares. Também foi aplicado o `StandardScaler` antes da classificação.

## 1.6 Pipeline de pré-processamento

Nos modelos de Regressão Logística e SVM, a padronização foi implementada dentro de um `Pipeline` do Scikit-learn.

Dessa maneira, durante o treinamento, o `StandardScaler` aprende os parâmetros de normalização exclusivamente com os dados utilizados no ajuste do modelo.

Na validação cruzada, esse procedimento é repetido em cada divisão, reduzindo o risco de vazamento de dados (*data leakage*).

## 1.7 Validação cruzada

Foi utilizada validação cruzada estratificada com cinco divisões (`StratifiedKFold`, com cinco folds).

Essa técnica permite avaliar o desempenho dos modelos em diferentes partições dos dados de treinamento, reduzindo a dependência de uma única divisão.

Os resultados médios foram:

| Modelo | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Regressão Logística | 98,24% | 100,00% | 95,26% | 97,57% |
| SVM | 97,23% | 97,87% | 94,55% | 96,15% |
| Random Forest | 94,72% | 94,17% | 91,82% | 92,82% |

A Regressão Logística apresentou os melhores resultados médios nas métricas avaliadas, além de menor variação no F1-score entre as divisões.

Por esse motivo, foi selecionada para a avaliação final.

## 1.8 Avaliação final no conjunto de teste

Após a seleção do modelo, a Regressão Logística foi treinada novamente utilizando os 483 registros dos conjuntos de treinamento e validação.

Em seguida, foram realizadas previsões sobre os 86 registros reservados para teste.

Os resultados obtidos foram:

| Métrica | Resultado |
|---|---:|
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1-score | 100% |

A matriz de confusão apresentou os seguintes resultados:

| Diagnóstico real | Previsto benigno | Previsto maligno |
|---|---:|---:|
| Benigno | 54 | 0 |
| Maligno | 0 | 32 |

Portanto, todos os 86 registros do conjunto de teste foram classificados corretamente.

## 1.9 Interpretabilidade do modelo

Para compreender quais características tiveram maior influência nas previsões, foram utilizadas técnicas de interpretabilidade.

A importância das características foi inicialmente analisada por meio do modelo Random Forest.

Entre as características com maior importância, destacaram-se:

- `perimeter_worst`;
- `area_worst`;
- `concave points_mean`;
- `concave points_worst`;
- `radius_worst`.

Também foi utilizada a técnica SHAP (*SHapley Additive exPlanations*) para investigar a contribuição das características nas previsões do modelo.

Essas análises permitem compreender melhor os padrões utilizados pelos algoritmos, embora importância estatística não represente causalidade nem equivalha diretamente à relevância clínica.

## 1.10 Discussão crítica e limitações

Os resultados demonstram que os modelos conseguiram identificar padrões relevantes no dataset utilizado. A Regressão Logística apresentou o melhor desempenho na validação cruzada e obteve 100% de acerto no conjunto de teste.

Entretanto, esse resultado não deve ser interpretado como garantia de desempenho equivalente em pacientes reais.

O conjunto de teste possui 86 registros, sendo uma amostra limitada para sustentar conclusões sobre a generalização clínica do modelo.

Além disso, o desempenho pode variar quando o sistema é aplicado a dados provenientes de outras instituições, equipamentos, populações ou protocolos de exames.

Para uma possível aplicação prática, seria necessário realizar validações externas, avaliações clínicas, monitoramento contínuo e análise de riscos.

O modelo deve ser considerado uma ferramenta de apoio à decisão. A interpretação dos exames e a decisão diagnóstica final devem permanecer sob responsabilidade dos profissionais de saúde.

## 1.11 Conclusão da etapa

A etapa de classificação com dados estruturados foi concluída com a implementação e comparação de três algoritmos de Machine Learning.

A Regressão Logística apresentou o melhor desempenho na validação cruzada e alcançou resultados perfeitos no conjunto de teste utilizado.

Também foram realizadas análises de importância das características e interpretabilidade com SHAP.

Os resultados representam uma base promissora para o desenvolvimento de uma solução de apoio ao diagnóstico, respeitando as limitações de generalização e a necessidade de validação clínica.