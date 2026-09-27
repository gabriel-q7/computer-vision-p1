# computer-vision-p1

Repositório organizado em quatro atividades de Visão Computacional.

## Estrutura

| Atividade | Tema | Status | Diretório |
| --- | --- | --- | --- |
| 1 | Projeto Livre: Visão Computacional com Transformers | Implementada | `atividade-1-projeto-livre/` |
| 2 | Reconhecimento Semântico em Publicidade Visual com CLIP | Implementada | `atividade-2-clip-publicidade/` |
| 3 | Classificador com CNN Pré-treinada | A fazer | `atividade-3-cnn-pre-treinada/` |
| 4 | Estudos de Caso | A fazer | `atividade-4-estudos-de-caso/` |

## Atividade 1 — Projeto Livre: Visão Computacional com Transformers

Implementação de um pipeline de classificação de severidade de acidentes usando um Vision Transformer customizado em PyTorch, com suporte a fine-tuning de ViT pré-treinado e visualização avançada de atenção via Attention Rollout.

Notebook principal: `atividade-1-projeto-livre/vit_accident_severity_colab.ipynb`.

## Atividade 2 — Reconhecimento Semântico em Publicidade Visual com CLIP

Implementação de inferência zero-shot com CLIP (`ViT-B/32`) no dataset ADS-16 para ranking semântico de 20 conceitos e busca semântica por linguagem natural (8 consultas variando entre genéricas, concretas, específicas e abstratas).

Notebook principal: `atividade-2-clip-publicidade/clip_semantic_advertising_colab.ipynb`.

Uso esperado no Google Colab:

1. Carregue o dataset ADS-16 para `/content/ads16` (ou utilize o helper de extração no notebook).
2. Execute as células em ordem utilizando um runtime com GPU T4.
3. O notebook extrai embeddings visuais e textuais, calcula a similaridade de cosseno, analisa thresholds heurísticos e executa busca semântica com visualizações e análises qualitativas completas.

## Próximas atividades

As pastas das Atividades 3 e 4 foram criadas como espaço reservado para suas futuras implementações.
