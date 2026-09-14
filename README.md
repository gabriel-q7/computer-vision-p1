# computer-vision-p1

Primeira versão funcional de um pipeline de classificação de severidade de acidentes com Vision Transformer customizado em PyTorch.

Notebook principal:

- `vit_accident_severity_colab.ipynb`

Uso esperado no Google Colab:

1. Baixe e extraia o dataset Kaggle `marslanarshad/car-accidents-and-deformation-datasetannotated`.
2. Garanta a estrutura `dataset/images/{train,val}` e `dataset/labels/{train,val}`.
3. No notebook, ajuste `DATASET_ROOT` para o diretório extraído.
4. Execute as células em ordem com runtime GPU.

O notebook cria labels de classificação por imagem usando `max(class_id)` sobre as annotations YOLO e refaz o split em nível de grupo de imagem para reduzir leakage conhecido por nomes de augmentations.
