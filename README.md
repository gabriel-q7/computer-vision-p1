# computer-vision-p1

Repositório organizado em quatro atividades de Visão Computacional.

## Estrutura

| Atividade | Tema | Status | Diretório |
| --- | --- | --- | --- |
| 1 | Projeto Livre: Visão Computacional com Transformers | Implementada | `atividade-1-projeto-livre/` |
| 2 | Reconhecimento Semântico em Publicidade Visual com CLIP | Implementada | `atividade-2-clip-publicidade/` |
| 3 | Classificador com CNN Pré-treinada | Implementada | `atividade-3-cnn-pre-treinada/` |
| 4 | Estudos de Caso | Implementada | `atividade-4-estudos-de-caso/` |

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

## Atividade 4 — Estudos de Caso em Deep Learning

**4.1:** detecção de COVID-19 no [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database). Amostra reprodutível de 1.200 imagens (840 Normal / 240 Viral Pneumonia / 120 COVID), divisão estratificada e agrupada por hash perceptual, ResNet-18 pré-treinada (backbone congelado) e uma GAN condicional (64×64) treinada apenas com o treino real. Compara o Experimento A (somente reais) com o Experimento B (+ sintéticas de COVID-19), tendo o recall de COVID-19 como métrica primária, com bootstrap pareado para medir a incerteza.

**4.2:** análise escrita de transfer learning para análise de tráfego urbano (ResNet-50, 12 câmeras).

Notebook principal: `atividade-4-estudos-de-caso/covid_cgan_case_study_colab.ipynb`.

Uso esperado no Google Colab:

1. Selecione o runtime com GPU T4 e execute as células em ordem. O dataset público é baixado automaticamente via CLI `kaggle` (com `kagglehub` como alternativa), sem exigir credenciais; também é possível enviar o `.zip` manualmente para `/content/`.
2. Opcional: se o Kaggle exigir autenticação, configure os Secrets `KAGGLE_USERNAME`/`KAGGLE_KEY` (ou `KAGGLE_API_TOKEN`) no painel 🔑 do Colab.
3. A quantidade de imagens sintéticas é controlada por `N_SYNTHETIC`. Os resultados (JSON/CSV, checkpoints, figuras e imagens geradas) são salvos em `/content/results_atividade4`.
