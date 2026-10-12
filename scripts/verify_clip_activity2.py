#!/usr/bin/env python3
"""Reexecuta o pipeline CLIP da Atividade 2 para verificar afirmações que só aparecem em figuras.

Calcula, de forma determinística:
- a composição do corpus ADS-16 (pastas de anúncios × imagens pessoais POS/NEG) e as duplicatas (MD5);
- a distribuição das similaridades imagem-conceito e a tabela de limiares;
- o top-5 das oito consultas, a faixa de escores, as imagens recorrentes ("hubs") e
  quantas posições dos top-5 são imagens pessoais.

Requer torch, Pillow, numpy e o pacote `clip` da OpenAI (pip install git+https://github.com/openai/CLIP.git).
Uso: python3 scripts/verify_clip_activity2.py <diretório do ADS-16 extraído>
"""

import collections
import hashlib
import sys
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.jfif', '.tif', '.tiff'}
TEMPLATE = 'an advertisement featuring {}'
CONCEPTS = ['a car', 'a person', 'a woman', 'a man', 'food product', 'a drink', 'a smartphone', 'a computer',
            'a television', 'a piece of furniture', 'a building', 'outdoor scenery', 'nature', 'a road', 'a vehicle',
            'text and logo', 'a fashion product', 'a household product', 'a sports product', 'a luxury product']
QUERIES = ['a product advertisement', 'a person', 'a car', 'a person using a smartphone',
           'a luxury car in an urban environment', 'a cold drink being advertised', 'a premium lifestyle',
           'an advertisement designed to attract attention']


def kind(path: Path) -> str:
    name = path.parent.name
    if name.isdigit():
        return 'anúncio'
    return 'pessoal-' + name.rsplit('-', 1)[-1] if name.startswith('U') else 'outro'


def encode_text(model, texts):
    with torch.no_grad():
        features = model.encode_text(clip.tokenize(texts))
    return features / features.norm(dim=-1, keepdim=True)


def main() -> None:
    root = Path(sys.argv[1])
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    md5 = [hashlib.md5(p.read_bytes()).hexdigest() for p in paths]
    print(f'Imagens: {len(paths)} | pastas: {len({p.parent.name for p in paths})} | conteúdos distintos: {len(set(md5))}')
    print('Composição:', dict(collections.Counter(kind(p) for p in paths)))

    model, preprocess = clip.load('ViT-B/32', device='cpu')
    model.eval()
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(paths), 64):
            batch = torch.stack([preprocess(Image.open(p).convert('RGB')) for p in paths[start:start + 64]])
            features = model.encode_image(batch)
            embeddings.append(features / features.norm(dim=-1, keepdim=True))
    images = torch.cat(embeddings)

    similarity = (images @ encode_text(model, [TEMPLATE.format(c) for c in CONCEPTS]).T).numpy().astype(np.float64)
    p1, p99 = np.percentile(similarity, [1, 99])
    print(f'Similaridades ({similarity.size} valores): mín {similarity.min():.4f} | p1 {p1:.4f} | '
          f'p99 {p99:.4f} | máx {similarity.max():.4f}')
    for threshold in (0.20, 0.25, 0.28, 0.30, 0.35):
        detected = (similarity >= threshold).sum(axis=1)
        print(f'Limiar {threshold:.2f}: média {detected.mean():.2f} conceitos/imagem | '
              f'sem conceito {100 * (detected == 0).mean():.1f}%')

    recurrence, scores, personal = collections.Counter(), [], 0
    for query in QUERIES:
        query_scores = (images @ encode_text(model, [TEMPLATE.format(query)]).T).squeeze(1).numpy()
        top = np.argsort(query_scores)[::-1][:5]
        print(f'\n{query}')
        for rank, index in enumerate(top, start=1):
            print(f'  #{rank} {query_scores[index]:.3f} {kind(paths[index]):14s} '
                  f'{paths[index].parent.name}/{paths[index].name} md5={md5[index][:8]}')
        scores += [query_scores[i] for i in top]
        personal += sum(kind(paths[i]) != 'anúncio' for i in top)
        recurrence.update({md5[i] for i in top})

    print(f'\nEscores dos top-5: mín {min(scores):.3f} | máx {max(scores):.3f}')
    print(f'Posições dos top-5 ocupadas por imagens pessoais: {personal} de {5 * len(QUERIES)}')
    names = {m: f'{p.parent.name}/{p.name}' for p, m in zip(paths, md5)}
    print('Imagens mais recorrentes (consultas em que aparecem):',
          [(names[m], n) for m, n in recurrence.most_common(5)])


if __name__ == '__main__':
    main()
