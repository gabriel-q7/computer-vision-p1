#!/usr/bin/env python3
"""Confere os números do relatório contra as saídas salvas nos notebooks.

Para cada afirmação numérica dos capítulos (`atividade-*/RELATORIO.md`), o script:

1. extrai os valores brutos das saídas registradas nos notebooks (a fonte de verdade);
2. recalcula as grandezas derivadas (somas, proporções, diferenças, intervalos de Wilson,
   métricas reconstruídas a partir das matrizes de confusão);
3. verifica se o valor formatado em pt-BR aparece no texto do capítulo correspondente.

Também faz checagens de consistência interna (por exemplo, a acurácia impressa pelo notebook
deve coincidir com a recalculada a partir da matriz de confusão).

Usa apenas a biblioteca padrão. Uso (a partir da raiz do repositório):
    python3 scripts/verify_report_numbers.py
"""

import json
import math
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

NB1 = 'atividade-1-projeto-livre/vit_accident_severity_colab.ipynb'
NB2 = 'atividade-2-clip-publicidade/clip_semantic_advertising_colab.ipynb'
NB3 = 'atividade-3-cnn-pre-treinada/cnn_feature_extraction_colab.ipynb'
NB4 = 'atividade-4-estudos-de-caso/covid_cgan_case_study_colab.ipynb'

REP1 = 'atividade-1-projeto-livre/RELATORIO.md'
REP2 = 'atividade-2-clip-publicidade/RELATORIO.md'
REP3 = 'atividade-3-cnn-pre-treinada/RELATORIO.md'
REP4 = 'atividade-4-estudos-de-caso/RELATORIO.md'

FAILURES = []
PASSES = 0


# ------------------------------------------------------------------ utilidades

def cell_texts(notebook: str) -> list:
    """Texto das saídas de cada célula (stream, text/plain e text/markdown)."""
    nb = json.loads((REPO_ROOT / notebook).read_text(encoding='utf-8'))
    texts = []
    for cell in nb['cells']:
        parts = []
        for output in cell.get('outputs', []):
            data = output.get('data', {})
            parts.append(''.join(output.get('text', '')))
            parts.append(''.join(data.get('text/plain', '')))
            parts.append(''.join(data.get('text/markdown', '')))
        texts.append(re.sub(r'\x1b\[[0-9;]*m', '', '\n'.join(parts)))
    return texts


def find(pattern: str, text: str, cast=float, group: int = 1):
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f'padrão não encontrado nas saídas: {pattern!r}')
    return cast(match.group(group))


def br(value: float, decimals: int) -> str:
    """Número no formato pt-BR (vírgula decimal, ponto de milhar)."""
    text = f'{value:,.{decimals}f}'
    return text.replace(',', '_').replace('.', ',').replace('_', '.')


def report(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding='utf-8')


def expect_in(label: str, fragment: str, path: str) -> None:
    """O fragmento (já calculado e formatado) precisa aparecer no capítulo."""
    global PASSES
    # O negrito do Markdown não altera o valor; é ignorado nos dois lados da comparação.
    if fragment.replace('**', '') in report(path).replace('**', ''):
        PASSES += 1
    else:
        FAILURES.append(f'{path}: {label} — esperado no texto: {fragment!r}')


def expect_close(label: str, computed: float, printed: float, tol: float) -> None:
    """Consistência interna entre valor recalculado e valor impresso pelo notebook."""
    global PASSES
    if abs(computed - printed) <= tol:
        PASSES += 1
    else:
        FAILURES.append(f'{label}: recalculado {computed:.6f} ≠ impresso {printed:.6f}')


def wilson(successes: int, n: int, z: float = 1.96):
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return center - half, center + half


def metrics_from_confusion(cm: list) -> dict:
    """Acurácia, precisão/recall/F1 por classe e macro F1 a partir da matriz de confusão."""
    n = len(cm)
    total = sum(map(sum, cm))
    out = {'accuracy': sum(cm[i][i] for i in range(n)) / total, 'precision': [], 'recall': [], 'f1': []}
    for k in range(n):
        tp = cm[k][k]
        predicted = sum(cm[i][k] for i in range(n))
        actual = sum(cm[k])
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out['precision'].append(precision)
        out['recall'].append(recall)
        out['f1'].append(f1)
    out['macro_f1'] = sum(out['f1']) / n
    return out


def epoch_log(text: str, total_epochs: int) -> list:
    """Linhas 'Epoch XX/N | train_loss=... train_acc=... | val_loss=... val_acc=... val_macro_f1=...'."""
    pattern = (rf'Epoch (\d+)/{total_epochs} \| train_loss=([\d.]+) train_acc=([\d.]+) \| '
               r'val_loss=([\d.]+) val_acc=([\d.]+) val_macro_f1=([\d.]+)')
    return [dict(zip(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_f1'],
                     [int(m[0])] + [float(x) for x in m[1:]]))
            for m in re.findall(pattern, text)]


def best_epoch(rows: list, key: str) -> dict:
    """Primeira época com o maior valor (o notebook só troca o checkpoint com melhora estrita)."""
    best = rows[0]
    for row in rows[1:]:
        if row[key] > best[key]:
            best = row
    return best


# ------------------------------------------------------------------ Atividade 1

def check_activity_1() -> None:
    texts = cell_texts(NB1)
    full = '\n'.join(texts)

    n_valid = find(r'Imagens válidas com label: (\d+)', full, int)
    expect_in('imagens válidas', f'{br(n_valid, 0)} imagens válidas', REP1)
    expect_in('grupos no train e val originais',
              f"**{find(r'Grupos presentes no train e val originais: (\d+)', full, int)} grupos**", REP1)
    expect_in('total de grupos', f"({br(find(r'normalizar raízes de augmentation: (\d+)', full, int), 0)} grupos)", REP1)

    counts = [int(x) for x in re.findall(r'Classe \d \(.+?\): (\d+)', full)[:10]]
    train_counts, val_counts = counts[:5], counts[5:]
    names = ['Sem Acidente', 'Acidente Leve', 'Acidente Moderado', 'Acidente Severo', 'Perda Total']
    for name, tr, va in zip(names, train_counts, val_counts):
        expect_in(f'distribuição {name}', f'| {name} | {tr} | {va} |', REP1)
    expect_in('total treino/validação', f'| **Total** | **{sum(train_counts)}** | **{sum(val_counts)}** |', REP1)
    expect_close('treino + validação = imagens válidas', sum(train_counts) + sum(val_counts), n_valid, 0)

    weights = [sum(train_counts) / (5 * c) for c in train_counts]
    expect_in('pesos de classe (mín./máx.)',
              f'de {br(min(weights), 2)} para *Perda Total* a {br(max(weights), 2)} para *Sem Acidente*', REP1)

    logs = {total: epoch_log(full, total) for total in (5, 20, 10)}
    models = [('ResNet-18 (ImageNet)', 5, 'baseline', r'do baseline'),
              ('ViT do zero', 20, 'scratch', None),
              ('ViT-B/16 (ImageNet-1k)', 10, 'pretrained', r'do ViT Pré-Treinado')]
    # O resumo final imprime acurácia e macro F1 com precisão completa (evita erro de arredondamento duplo)
    exact = {'baseline': (r'Accuracy do Baseline \(ResNet18\): ([\d.]+)', r'Macro F1 do Baseline \(ResNet18\): ([\d.]+)'),
             'scratch': (r'Accuracy do ViT \(Scratch Otimizado\): ([\d.]+)', r'Macro F1 do ViT \(Scratch\): ([\d.]+)'),
             'pretrained': (r'Accuracy do ViT Pré-Treinado \(ViT-B/16\): ([\d.]+)', r'Macro F1 do ViT Pré-Treinado \(ViT-B/16\): ([\d.]+)')}
    summary = {}
    for label, total, key, suffix in models:
        best = best_epoch(logs[total], 'val_f1')
        if suffix:
            acc = find(rf'Accuracy de validação {suffix}: ([\d.]+)', full)
            prec = find(rf'Macro precision {suffix}: ([\d.]+)', full)
            rec = find(rf'Macro recall {suffix}: ([\d.]+)', full)
            f1 = find(rf'Macro F1 {suffix}: ([\d.]+)', full)
        else:
            acc = find(r'^Accuracy: ([\d.]+)', full)
            prec = find(r'^Macro precision: ([\d.]+)', full)
            rec = find(r'^Macro recall: ([\d.]+)', full)
            f1 = find(r'^Macro F1: ([\d.]+)', full)
        acc, f1 = find(exact[key][0], full), find(exact[key][1], full)
        expect_close(f'{label}: acurácia final = acurácia da melhor época', acc, best['val_acc'], 1e-4)
        expect_close(f'{label}: macro F1 final = macro F1 da melhor época', f1, best['val_f1'], 1e-4)
        values = [br(acc, 3), br(prec, 3), br(rec, 3), br(f1, 3)]
        if key == 'pretrained':
            row = f"| **{label}** | {best['epoch']} | " + ' | '.join(f'**{v}**' for v in values) + ' |'
        else:
            row = f"| {label} | {best['epoch']} | " + ' | '.join(values) + ' |'
        expect_in(f'linha de resultados {label}', row, REP1)
        summary[key] = {'acc': acc, 'f1': f1, 'best': best, 'log': logs[total]}

    majority = max(val_counts) / sum(val_counts)
    expect_in('baseline da classe majoritária', f'**{br(majority, 3)}** de acurácia', REP1)
    expect_in('ViT do zero x majoritária',
              f"({br(summary['scratch']['acc'], 3)} contra {br(majority, 3)})", REP1)
    max_train_scratch = max(r['train_acc'] for r in summary['scratch']['log'])
    expect_in('acurácia máxima de treino do ViT do zero', f'chegou a apenas {br(max_train_scratch, 2)}', REP1)

    best_pre = summary['pretrained']['best']
    gap = 100 * (best_pre['train_acc'] - summary['pretrained']['acc'])
    expect_in('treino x validação do ViT-B/16 na época do checkpoint',
              f"{br(best_pre['train_acc'], 3)} de acurácia no treino contra {br(summary['pretrained']['acc'], 3)} na validação", REP1)
    expect_in('lacuna treino-validação do ViT-B/16', f'lacuna de ~{br(gap, 0)} p.p.', REP1)

    pre_cell = next(t for t in texts if 'Macro F1 do ViT Pré-Treinado' in t)
    f1_rows = [row for row in re.findall(r'^\d\s+(.+?)\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+\d+$', pre_cell, flags=re.MULTILINE)
               if row[0].strip() != 'Macro']
    short = {'Sem Acidente': 'Sem Acidente', 'Acidente Leve': 'Leve', 'Acidente Moderado': 'Moderado',
             'Acidente Severo': 'Severo', 'Perda Total': 'Perda Total'}
    fragment = '; '.join(f'*{short[name.strip()]}* {br(float(v), 3)}' for name, v in f1_rows)
    expect_in('F1 por classe do ViT-B/16', fragment, REP1)


# ------------------------------------------------------------------ Atividade 2

def check_activity_2() -> None:
    full = '\n'.join(cell_texts(NB2))
    n_images = find(r'Total de imagens encontradas: (\d+)', full, int)
    expect_in('total de imagens', f'**{br(n_images, 0)} imagens**', REP2)
    expect_in('total de pastas', f"**{find(r'Total de categorias distintas: (\d+)', full, int)} pastas**", REP2)

    rows = re.findall(r'^\d\s+(0\.\d+)\s+([\d.]+)\s+\d+\s+\d+\s+([\d.]+)%', full, flags=re.MULTILINE)
    for threshold, mean, empty in rows:
        if float(threshold) > 0.35:
            continue
        expect_in(f'linha do limiar {threshold}',
                  f'| {br(float(threshold), 2)} | {br(float(mean), 2)} | {br(float(empty), 1)}%', REP2)

    ranking = re.findall(r'^(\d+)\s+(a fashion product|a person|a household product|a woman|a luxury product|a building)'
                         r'\s+(0\.\d+)\s+(\d+)', full, flags=re.MULTILINE)
    for position, concept, similarity, count in ranking:
        frequency = 100 * int(count) / n_images
        expect_in(f'ranking {concept}',
                  f'| {position} | {concept} | {br(float(similarity), 4)} | {count} | {br(frequency, 2)}% |', REP2)
    top = max(ranking, key=lambda r: int(r[3]))
    expect_in('conceito mais frequente', f'aparece em apenas {top[3]} de {br(n_images, 0)} imagens', REP2)


# ------------------------------------------------------------------ Atividade 3

def check_activity_3() -> None:
    full = '\n'.join(cell_texts(NB3))
    n_files = find(r'Arquivos válidos: (\d+)', full, int)
    exact = find(r'Removidos por duplicata exata \(MD5\): (\d+)', full, int)
    near = find(r'Removidos por quase duplicata \(dHash ≤ \d+\): (\d+)', full, int)
    kept = find(r'Imagens distintas mantidas: (\d+)', full, int)
    expect_close('arquivos − exatas − quase = mantidas', n_files - exact - near, kept, 0)
    expect_in('removidas por MD5', f'| −{br(exact, 0)} |', REP3)
    expect_in('removidas por quase duplicata', f'| −{near} |', REP3)
    expect_in('imagens distintas', f'| **{br(kept, 0)}** |', REP3)

    split = re.findall(r'^(bike|cars|cats|dogs|flowers|horses|human)\s+(\d+)\s+(\d+)\s+(\d+)$', full, flags=re.MULTILINE)
    for name, total, train, val in split:
        expect_close(f'{name}: treino + validação = total', int(train) + int(val), int(total), 0)
        expect_in(f'linha da divisão {name}', f'| {name} | {total} | {train} | {val} |', REP3)
    n_train = sum(int(r[2]) for r in split)
    n_val = sum(int(r[3]) for r in split)
    expect_close('soma das classes = imagens distintas', n_train + n_val, kept, 0)
    expect_in('total da divisão', f'| **{br(kept, 0)}** | **{br(n_train, 0)}** | **{n_val}** |', REP3)
    expect_in('menor distância dHash entre conjuntos',
              f"a {find(r'menor distância dHash: (\d+) bits', full, int)} bits de distância", REP3)

    epochs = [dict(zip(['epoch', 'tl', 'ta', 'vl', 'va'], [int(m[0])] + [float(x) for x in m[1:]]))
              for m in re.findall(r'Epoch (\d+)/10 \| train loss: ([\d.]+) \| train acc: ([\d.]+) \| '
                                  r'val loss: ([\d.]+) \| val acc: ([\d.]+)', full)]
    best = best_epoch(epochs, 'va')
    expect_in('melhor época', f"O melhor checkpoint (época {best['epoch']})", REP3)
    for row in epochs:
        if row['epoch'] in (1, 5, 7, 10):
            va = f"**{br(row['va'], 4)}**" if row['epoch'] == best['epoch'] else br(row['va'], 4)
            expect_in(f"linha da época {row['epoch']}",
                      f"| {row['epoch']} | {br(row['tl'], 4)} | {br(row['ta'], 4)} | {br(row['vl'], 4)} | {va} |", REP3)

    errors = find(r'Classificações incorretas: (\d+) de (\d+)', full, int)
    total_val = find(r'Classificações incorretas: \d+ de (\d+)', full, int)
    correct = total_val - errors
    accuracy = correct / total_val
    expect_close('acurácia impressa = acertos / total', accuracy, find(r'Accuracy global: ([\d.]+)', full), 1e-4)
    expect_close('melhor acurácia de validação = acurácia final', best['va'], accuracy, 1e-4)
    low, high = wilson(correct, total_val)
    expect_in('acertos', f'**{correct} das {total_val} imagens**', REP3)
    expect_in('acurácia', f'**acurácia de {br(accuracy, 4)}**', REP3)
    expect_in('IC de Wilson', f'**[{br(low, 3)}; {br(high, 4)}]**', REP3)

    # Matriz de confusão reconstruída: o único erro é cats → dogs (recall de cats < 1 e precisão de dogs < 1)
    per_class = dict(re.findall(r'^\d\s+(\w+)\s+([\d.]+)\s+[\d.]+$', full, flags=re.MULTILINE))
    supports = {name: int(val) for name, _, _, val in split}
    classes = sorted(supports)
    cm = [[supports[c] if c == r else 0 for c in classes] for r in classes]
    cats, dogs = classes.index('cats'), classes.index('dogs')
    cm[cats][cats] -= 1
    cm[cats][dogs] += 1
    for name in classes:
        expect_close(f'recall reconstruído de {name}', cm[classes.index(name)][classes.index(name)] / supports[name],
                     float(per_class[name]), 1e-3)
    reconstructed = metrics_from_confusion(cm)
    expect_in('menor F1 por classe', f'F1 de pelo menos {br(min(reconstructed["f1"]), 3)}', REP3)

    first = epochs[0]
    expect_in('acurácia de validação na 1ª época', f"já é de {br(first['va'], 3)} na primeira época", REP3)
    last = epochs[-1]
    expect_close('lacuna treino-validação < 0,2 p.p. (última época)', min(100 * abs(last['ta'] - last['va']), 0.2), 100 * abs(last['ta'] - last['va']), 0.05)


# ------------------------------------------------------------------ Atividade 4

def check_activity_4() -> None:
    texts = cell_texts(NB4)
    full = '\n'.join(texts)

    counts = {c: find(rf'^{c}\s+(\d+)$', full, int) for c in ('COVID', 'Normal', 'Viral Pneumonia')}
    total = sum(counts.values())
    expect_close('total impresso = soma das classes', find(r'Total de imagens nas três classes: (\d+)', full, int), total, 0)
    expect_in('total de imagens', f'havia {br(total, 0)} imagens PNG', REP4)
    expect_in('contagens por classe', f"{br(counts['Normal'], 0)} *Normal*, {br(counts['Viral Pneumonia'], 0)} "
                                      f"*Viral Pneumonia* e {br(counts['COVID'], 0)} *COVID*", REP4)
    expect_in('duplicatas removidas', f"{find(r'duplicata exata na mesma classe: (\d+)', full, int)} duplicatas exatas", REP4)
    pct = [round(100 * counts[c] / total) for c in ('Normal', 'Viral Pneumonia', 'COVID')]
    expect_in('distribuição original (%)', f'distribuição original ({pct[0]}/{pct[1]}/{pct[2]})', REP4)

    sources = dict((name, int(covid)) for name, covid in
                   re.findall(r'^(bimcv\.cipf\.es|eurorad\.org|github/\S+|sirm\.org)\s+(\d+)\s+0\s+0$', full, flags=re.MULTILINE))
    others = sum(v for k, v in sources.items() if k != 'bimcv.cipf.es')
    expect_close('fontes de COVID somam o total de COVID', sum(sources.values()), counts['COVID'], 0)
    expect_in('fonte bimcv', f"| `bimcv.cipf.es` | {br(sources['bimcv.cipf.es'], 0)} | 0 | 0 |", REP4)
    expect_in('outras fontes', f'| Outras (GitHub, Eurorad, SIRM) | {br(others, 0)} | 0 | 0 |', REP4)

    split_cell = next(t for t in texts if 'Treino de ajuste' in t)
    def split_counts(label_idx):
        rows = re.findall(r'^(Normal|Viral Pneumonia|COVID)\s+([\d.]+)\s+[\d.]+\s+([\d.]+)\s+[\d.]+', split_cell, flags=re.MULTILINE)
        return rows
    rows = split_counts(0)
    first_block, second_block = rows[:3], rows[3:6]
    val = [int(float(r[2])) for r in first_block]
    fit = [int(float(r[1])) for r in second_block]
    internal = [int(float(r[2])) for r in second_block]
    expect_in('linha treino de ajuste', f'| Treino de ajuste | {fit[0]} | {fit[1]} | {fit[2]} | {sum(fit)} |', REP4)
    expect_in('linha validação interna', f'| Validação interna | {internal[0]} | {internal[1]} | {internal[2]} | {sum(internal)} |', REP4)
    expect_in('linha validação real', f'| Validação real | {val[0]} | {val[1]} | {val[2]} | {sum(val)} |', REP4)
    expect_close('divisão soma 1.200', sum(fit) + sum(internal) + sum(val), 1200, 0)
    expect_in('parâmetros treináveis', f"({br(find(r'parâmetros treináveis: ([\d,]+)', full, lambda s: int(s.replace(',', ''))), 0)} parâmetros)", REP4)
    expect_close('parâmetros da camada final = 512·3 + 3', 512 * 3 + 3,
                 find(r'parâmetros treináveis: ([\d,]+)', full, lambda s: int(s.replace(',', ''))), 0)

    iters = find(r'Iterações por época: (\d+) \| total previsto: (\d+)', full, int)
    total_iters = find(r'total previsto: (\d+)', full, int)
    expect_close('iterações da cGAN = iterações/época × épocas', iters * 400, total_iters, 0)
    expect_in('iterações e tempo da cGAN',
              f"400 épocas ({br(total_iters, 0)} iterações, {find(r'treino_cgan: ([\d.]+) min', full, lambda s: round(float(s)))} min na T4)", REP4)

    covid_fit, n_synth = fit[2], find(r'^(\d+) imagens sintéticas de COVID-19 salvas', full, int)
    covid_b = covid_fit + n_synth
    share_a = 100 * covid_fit / sum(fit)
    share_b = 100 * covid_b / (sum(fit) + n_synth)
    expect_in('COVID no treino B', f'de {covid_fit} para {covid_b} exemplos ({br(share_b, 1)}% do treino)', REP4)
    expect_in('peso de COVID A → B', f'de {round(share_a)}% para {round(share_b)}% do treino', REP4)

    diag = {k: float(v) for k, v in re.findall(r'^(\w+)\s+([\d.]+)$', next(t for t in texts if 'razao_diversidade' in t), flags=re.MULTILINE)}
    expect_close('razão de diversidade', diag['diversidade_sinteticas'] / diag['diversidade_reais_covid'], diag['razao_diversidade'], 1e-3)
    expect_close('razão de memorização', diag['nn_sintetica_para_real_mediana'] / diag['nn_real_para_real_mediana'], diag['razao_memorizacao'], 1e-3)
    expect_in('linha de diversidade', f"| {br(diag['diversidade_sinteticas'], 2)} | {br(diag['diversidade_reais_covid'], 2)} | {br(diag['razao_diversidade'], 2)} |", REP4)
    expect_in('linha de memorização', f"| {br(diag['nn_sintetica_para_real_mediana'], 2)} | {br(diag['nn_real_para_real_mediana'], 2)} | {br(diag['razao_memorizacao'], 2)} |", REP4)
    expect_in('linha de desvio de intensidade', f"| {br(diag['desvio_intensidade_sintetica'], 3)} | {br(diag['desvio_intensidade_real_covid'], 3)} |", REP4)
    expect_in('redução de diversidade', f"cerca de {round(100 * (1 - diag['razao_diversidade']))}% menor", REP4)

    # Matrizes de confusão reconstruídas a partir das saídas (recall por classe × suporte, precisão de VP e COVID = 1)
    results = {}
    for name in ('A', 'B'):
        cell = next(t for t in texts if f'Experimento {name} — validação real' in t)
        report_rows = dict((cls, (float(p), float(r))) for cls, p, r in
                           re.findall(r'^\s*(Normal|Viral Pneumonia|COVID)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+\s+\d+$', cell, flags=re.MULTILINE))
        tp = find(r'TP=(\d+) FN=\d+ FP=\d+', cell, int)
        fn = find(r'TP=\d+ FN=(\d+)', cell, int)
        fp = find(r'FP=(\d+)', cell, int)
        vp_ok = round(report_rows['Viral Pneumonia'][1] * val[1])
        cm = [[val[0], 0, 0], [val[1] - vp_ok, vp_ok, 0], [fn, 0, tp]]
        expect_close(f'{name}: TP + FN = casos de COVID', tp + fn, val[2], 0)
        expect_close(f'{name}: sem falsos positivos de COVID', fp, 0, 0)
        m = metrics_from_confusion(cm)
        for k, cls in enumerate(['Normal', 'Viral Pneumonia', 'COVID']):
            expect_close(f'{name}: precisão reconstruída de {cls}', m['precision'][k], report_rows[cls][0], 1e-3)
            expect_close(f'{name}: recall reconstruído de {cls}', m['recall'][k], report_rows[cls][1], 1e-3)
        expect_close(f'{name}: acurácia reconstruída', m['accuracy'], find(r'Acurácia global : ([\d.]+)', cell), 1e-3)
        expect_close(f'{name}: macro F1 reconstruída', m['macro_f1'], find(r'Macro F1\s+: ([\d.]+)', cell), 1e-3)
        results[name] = {'m': m, 'tp': tp, 'fn': fn, 'fp': fp, 'cm': cm,
                         'epoch': find(rf'\[{name}\] melhor época: (\d+)', full, int)}

    a, b = results['A'], results['B']
    n_pos = val[2]
    def row(label, va, vb, diff):
        return f'| {label} | {va} | {vb} | {diff} |'
    expect_in('linha acurácia', row('Acurácia global', br(a['m']['accuracy'], 3), br(b['m']['accuracy'], 3),
                                    f"+{br(100 * (b['m']['accuracy'] - a['m']['accuracy']), 1)} p.p."), REP4)
    expect_in('linha macro F1', row('Macro F1', br(a['m']['macro_f1'], 3), br(b['m']['macro_f1'], 3),
                                    f"+{br(100 * (b['m']['macro_f1'] - a['m']['macro_f1']), 1)} p.p."), REP4)
    rec_a, rec_b = a['m']['recall'][2], b['m']['recall'][2]
    expect_in('linha recall COVID',
              f"| **Recall COVID-19** | **{br(rec_a, 3)}** ({a['tp']}/{n_pos}) | **{br(rec_b, 3)}** ({b['tp']}/{n_pos}) | "
              f"**+{br(100 * (rec_b - rec_a), 1)} p.p.** (+{br(100 * (rec_b - rec_a) / rec_a, 0)}% relativo) |", REP4)
    expect_in('linha precisão COVID', row('Precisão COVID-19', br(a['m']['precision'][2], 3), br(b['m']['precision'][2], 3),
                                          f"{br(100 * (b['m']['precision'][2] - a['m']['precision'][2]), 1)} p.p."), REP4)
    expect_in('linha F1 COVID', row('F1-score COVID-19', br(a['m']['f1'][2], 3), br(b['m']['f1'][2], 3),
                                    f"+{br(100 * (b['m']['f1'][2] - a['m']['f1'][2]), 1)} p.p."), REP4)
    expect_in('linha falsos negativos', row('Falsos negativos COVID', a['fn'], b['fn'], f"−{a['fn'] - b['fn']}"), REP4)
    expect_in('linha falsos positivos', row('Falsos positivos COVID', a['fp'], b['fp'], b['fp'] - a['fp']), REP4)
    expect_in('linha checkpoint', row('Checkpoint (época)', a['epoch'], b['epoch'], '—'), REP4)
    expect_in('diferença absoluta de recall', f"**{br(100 * (rec_b - rec_a), 1)} pontos percentuais**", REP4)
    expect_in('casos adicionais', f"**{['zero', 'um', 'dois', 'três', 'quatro', 'cinco'][b['tp'] - a['tp']]} casos adicionais**", REP4)
    expect_in('valor de um caso', f'cada caso vale {br(100 / n_pos, 1)} p.p.', REP4)

    for name, res in results.items():
        low, high = wilson(res['tp'], n_pos)
        printed = re.search(rf'Recall COVID {name}: [\d.]+\s+IC95% Wilson \[([\d.]+), ([\d.]+)\]', full)
        expect_close(f'{name}: Wilson inferior', low, float(printed.group(1)), 1e-3)
        expect_close(f'{name}: Wilson superior', high, float(printed.group(2)), 1e-3)
        expect_in(f'{name}: IC de Wilson', f'{name} em [{br(low, 3)}; {br(high, 3)}]', REP4)
    boot = re.search(r'Diferença de recall \(B − A\), IC95% bootstrap: \[([+-][\d.]+), ([+-][\d.]+)\] p\.p\.', full)
    expect_in('IC bootstrap do recall', f"**[{br(float(boot.group(1)), 1).replace('-', '−') if float(boot.group(1)) < 0 else '+' + br(float(boot.group(1)), 1)}; +{br(float(boot.group(2)), 1)}] p.p.**", REP4)
    boot_f1 = re.search(r'Diferença de macro F1 \(B − A\), IC95% bootstrap: \[([+-][\d.]+), ([+-][\d.]+)\]', full)
    expect_in('IC bootstrap da macro F1', f"[+{br(float(boot_f1.group(1)), 3)}; +{br(float(boot_f1.group(2)), 3)}]", REP4)
    expect_in('fração do bootstrap', f"em {round(100 * find(r'recall de B > A: ([\d.]+)', full))}% das reamostragens", REP4)

    expect_in('FN do texto da discussão', f"com {b['fn']} em vez de {a['fn']} falsos negativos", REP4)
    vp_a, vp_b = a['m']['recall'][1], b['m']['recall'][1]
    expect_in('recall de Viral Pneumonia', f'(de {br(vp_a, 3)} para {br(vp_b, 3)})', REP4)
    covid_as_normal = b['cm'][2][0] / n_pos
    expect_in('COVID classificado como Normal em B', f'**{round(100 * covid_as_normal)}% dos casos de COVID', REP4)
    expect_close('nenhum COVID previsto como Viral Pneumonia', a['cm'][2][1] + b['cm'][2][1], 0, 0)

    def history(name):
        cell = next(t for t in texts if f'[{name}] parâmetros treináveis' in t)
        return {int(e): (float(f1), float(r)) for e, f1, r in
                re.findall(r'época (\d+) \| .*? macroF1\(int\) ([\d.]+) \| recall COVID\(int\) ([\d.]+)', cell)}
    hist_a, hist_b = history('A')[a['epoch']], history('B')[b['epoch']]
    expect_in('validação interna (recall e macro F1)',
              f'(recall de {br(hist_a[1], 2)} para {br(hist_b[1], 2)} e macro F1 de {br(hist_a[0], 2)} para {br(hist_b[0], 2)})', REP4)

    expect_in('frames por câmera', f'~{round(800 / 12)} *frames* por câmera', REP4)
    expect_in('lacuna do cenário', f'lacuna de {93 - 61} p.p.', REP4)


# ------------------------------------------------------------------ execução

def main() -> int:
    for check in (check_activity_1, check_activity_2, check_activity_3, check_activity_4):
        try:
            check()
        except Exception as error:  # padrão ausente nas saídas também é uma falha de verificação
            FAILURES.append(f'{check.__name__}: {type(error).__name__}: {error}')
    print(f'Verificações aprovadas: {PASSES}')
    print(f'Falhas: {len(FAILURES)}')
    for failure in FAILURES:
        print(f'  - {failure}')
    return 1 if FAILURES else 0


if __name__ == '__main__':
    sys.exit(main())
