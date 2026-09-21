# computer-vision-p1

Repositório organizado em quatro atividades de Visão Computacional. Atualmente, apenas a Atividade 1 está implementada.

## Estrutura

| Atividade | Tema | Status | Diretório |
| --- | --- | --- | --- |
| 1 | Projeto Livre: Visão Computacional com Transformers | Implementada | `atividade-1-projeto-livre/` |
| 2 | Reconhecimento Semântico em Publicidade Visual com CLIP | A fazer | `atividade-2-clip-publicidade/` |
| 3 | Classificador com CNN Pré-treinada | A fazer | `atividade-3-cnn-pre-treinada/` |
| 4 | Estudos de Caso | A fazer | `atividade-4-estudos-de-caso/` |

## Atividade 1 — Projeto Livre: Visão Computacional com Transformers

Implementação de um pipeline de classificação de severidade de acidentes usando um Vision Transformer customizado em PyTorch.

Notebook principal: `atividade-1-projeto-livre/vit_accident_severity_colab.ipynb`.

Uso esperado no Google Colab:

1. Baixe e extraia o dataset Kaggle `marslanarshad/car-accidents-and-deformation-datasetannotated`.
2. Garanta a estrutura `dataset/images/{train,val}` e `dataset/labels/{train,val}`.
3. No notebook, ajuste `DATASET_ROOT` para o diretório extraído.
4. Execute as células em ordem usando um runtime com GPU.

O notebook cria labels de classificação por imagem usando `max(class_id)` sobre as anotações YOLO e refaz o split por grupo de imagem para reduzir o vazamento associado a nomes de augmentations.

## Próximas atividades

As pastas das Atividades 2, 3 e 4 foram criadas como espaço reservado para suas futuras implementações.
