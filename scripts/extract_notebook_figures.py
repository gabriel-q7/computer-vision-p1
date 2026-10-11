#!/usr/bin/env python3
"""Extrai as figuras usadas no relatório a partir das saídas salvas nos notebooks.

Cada entrada de FIGURES aponta para (notebook, índice da célula, n-ésima imagem PNG da célula)
e para o arquivo de destino. Se um notebook for reexecutado e a ordem das células mudar,
basta ajustar os índices abaixo. Usa apenas a biblioteca padrão.

Uso (a partir da raiz do repositório):
    python3 scripts/extract_notebook_figures.py
"""

import base64
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

NB1 = 'atividade-1-projeto-livre/vit_accident_severity_colab.ipynb'
NB2 = 'atividade-2-clip-publicidade/clip_semantic_advertising_colab.ipynb'
NB3 = 'atividade-3-cnn-pre-treinada/cnn_feature_extraction_colab.ipynb'
NB4 = 'atividade-4-estudos-de-caso/covid_cgan_case_study_colab.ipynb'

# (notebook, célula, n-ésima imagem da célula, destino relativo à raiz)
FIGURES = [
    (NB1, 29, 0, 'atividade-1-projeto-livre/figures/vit-scratch-loss.png'),
    (NB1, 35, 1, 'atividade-1-projeto-livre/figures/vit-scratch-confusao-normalizada.png'),
    (NB1, 41, 0, 'atividade-1-projeto-livre/figures/vit-scratch-attention-rollout.png'),

    (NB2, 23, 0, 'atividade-2-clip-publicidade/figures/threshold-conceitos.png'),
    (NB2, 26, 0, 'atividade-2-clip-publicidade/figures/ranking-conceitos.png'),
    (NB2, 32, 2, 'atividade-2-clip-publicidade/figures/busca-a-car.png'),
    (NB2, 32, 4, 'atividade-2-clip-publicidade/figures/busca-luxury-car-urban.png'),
    (NB2, 32, 6, 'atividade-2-clip-publicidade/figures/busca-premium-lifestyle.png'),

    (NB3, 7, 0, 'atividade-3-cnn-pre-treinada/figures/distribuicao-classes.png'),
    (NB3, 19, 0, 'atividade-3-cnn-pre-treinada/figures/curvas-treinamento.png'),
    (NB3, 22, 0, 'atividade-3-cnn-pre-treinada/figures/matriz-confusao.png'),

    (NB4, 17, 0, 'atividade-4-estudos-de-caso/figures/distribuicao-original-vs-amostra.png'),
    (NB4, 35, 4, 'atividade-4-estudos-de-caso/figures/cgan-amostras-epoca-400.png'),
    (NB4, 35, 5, 'atividade-4-estudos-de-caso/figures/cgan-losses.png'),
    (NB4, 38, 0, 'atividade-4-estudos-de-caso/figures/sinteticas-vs-reais.png'),
    (NB4, 48, 0, 'atividade-4-estudos-de-caso/figures/confusao-a-vs-b.png'),
    (NB4, 48, 1, 'atividade-4-estudos-de-caso/figures/historico-a-vs-b.png'),
]


def png_outputs(cell: dict) -> list:
    return [output['data']['image/png'] for output in cell.get('outputs', [])
            if 'image/png' in output.get('data', {})]


def main() -> int:
    notebooks = {}
    failures = 0
    for notebook, cell_index, image_index, target in FIGURES:
        if notebook not in notebooks:
            notebooks[notebook] = json.loads((REPO_ROOT / notebook).read_text(encoding='utf-8'))
        cells = notebooks[notebook]['cells']
        images = png_outputs(cells[cell_index]) if cell_index < len(cells) else []
        if image_index >= len(images):
            print(f'[ERROR] {notebook}: célula {cell_index} não tem a imagem #{image_index}', file=sys.stderr)
            failures += 1
            continue
        target_path = REPO_ROOT / target
        target_path.parent.mkdir(parents=True, exist_ok=True)
        data = images[image_index]
        target_path.write_bytes(base64.b64decode(''.join(data) if isinstance(data, list) else data))
        print(f'[INFO] {target}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
