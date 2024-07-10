# Classificação de classes de armas no jogo Elden Ring utilizando redes neurais convolucionais

## Sobre o Trabalho

Este trabalho foi desenvolvido para a disciplina de Tópicos em Inteligência Computacional e envolve a classificação de diversas classes de armas do jogo de ação e RPG Elden Ring. Para tal, utiliza de Redes Neurais Convolucionais (CNN), e o modelo é capaz de identificar e classificar imagens de armas com base em suas características visuais.

O artigo contendo a explicação completa e detalhada, incluindo resultados, está disponível no documento `Artigo.pdf` neste mesmo repositório.

## Arquitetura do Modelo

O modelo utiliza a arquitetura VGG16, adaptada para a tarefa específica de classificação das armas encontradas no jogo. O código para o treinamento e validação do modelo pode ser encontrado nos arquivos `treino.py` e `modelo.py`, respectivamente.

### Base de Dados

As imagens utilizadas para treinar o modelo foram coletadas da [wiki do Elden Ring](https://eldenring.wiki.fextralife.com/Elden+Ring+Wiki), que possui várias imagens separadas por classes de armas.

### Características do Modelo

- **Data Augmentation**: Técnicas como rotação, redimensionamento, e inversão são aplicadas para aumentar a diversidade do conjunto de dados originalmente limitado.
- **Fine-tuning**: Ajustes finos no modelo pré-treinado (VGG16) para adaptar às necessidades específicas da classificação de imagens de armas.

## Tecnologia Utilizada

- Python 3.8
- Biblioteca Keras com TensorFlow
- Matplotlib e Seaborn para visualização de dados
- scikit-learn

## Como Executar

### Pré-requisitos

- Python 3.8 ou superior
- Biblioteca TensorFlow
- Biblioteca Pandas
- Biblioteca Seaborn
- Biblioteca scikit-learn

### Instalação das Bibliotecas Necessárias
```bash
pip install tensorflow pandas seaborn scikit-learn
```

### Clonar o Repositório

```bash
git clone https://github.com/vitorhugosdc/topicos-inteligencia-computacional
cd topicos-inteligencia-computacional
```

## Preparar e Executar o Treinamento do Modelo

```bash
python treino.py
```

## Preparar e Executar o Modelo de Testes

```bash
python modelo.py
```

## Resultados

O modelo alcançou uma acurácia global de 85,18%, com a capacidade de distinguir entre diferentes classes de armas com alta precisão. Abaixo estão detalhes dos resultados obtidos, incluindo a matriz de confusão e métricas de performance como recall e precisão para cada classe de arma testada.

## Recall e Precision

| Classe            | Recall | Precision |
|-------------------|--------|-----------|
| **Axe**           | 100%   | 100%      |
| **Curved Sword**  | 75%    | 66.7%     |
| **Dagger**        | 75%    | 75%       |
| **Hammer**        | 100%   | 100%      |
| **Katana**        | 66.7%  | 100%      |
| **Spear**         | 100%   | 100%      |
| **Straight Sword**| 100%   | 83%       |

### Matriz de confusão

<img src="https://github.com/vitorhugosdc/assets/blob/main/raw/topicos-inteligencia-computacional/matriz-de-confusao.png" width="600">
