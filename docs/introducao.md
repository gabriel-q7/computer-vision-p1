# Introdução

Este relatório reúne as quatro atividades práticas de Visão Computacional e Deep Learning desenvolvidas no repositório `computer-vision-p1`. Cada atividade foi implementada como um notebook executável no Google Colab e explora uma família diferente de técnicas.

| Atividade | Tema | Técnica principal | Dataset |
|---:|----------------|----------------------|----------------|
| 1 | Classificação de severidade de acidentes | Vision Transformer (do zero e pré-treinado) e ResNet-18 | Car Accidents and Deformation Dataset (CADD) |
| 2 | Reconhecimento semântico em publicidade | CLIP (ViT-B/32) em regime *zero-shot* | ADS-16 |
| 3 | Classificador com CNN pré-treinada | ResNet-50 com extração de características | Images Dataset (7 classes) |
| 4 | Estudos de caso | ResNet-18 com transfer learning e GAN condicional; análise de transfer learning em tráfego urbano | COVID-19 Radiography Database |

## Ambiente de execução

Todos os notebooks foram executados no Google Colab com GPU NVIDIA Tesla T4, PyTorch 2.11 e torchvision 0.26 (CUDA 12.8). Os datasets são públicos e foram obtidos pela API oficial do Kaggle. Cada notebook fixa a semente `42` para Python, NumPy e PyTorch.

## Organização do relatório

Cada capítulo segue a mesma estrutura: objetivo, dados, método, resultados e discussão crítica, incluindo as limitações. Os números citados são os registrados nas saídas dos notebooks versionados no repositório, e as figuras foram extraídas dessas mesmas saídas.

Ao longo do relatório, adota-se uma postura crítica em relação aos próprios resultados: métricas muito altas ou muito baixas são investigadas, e problemas de protocolo (por exemplo, seleção de modelo no mesmo conjunto usado para reportar o desempenho) são apontados explicitamente.
