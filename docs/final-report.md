---
title: "Deep Learning and Computer Vision"
subtitle: "Relatório das Atividades 1 a 4"
author: "Gabriel Aguiar Aquino"
date: "12 de outubro de 2026"
lang: pt-BR
geometry: margin=2.5cm
toc: true
numbersections: true
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{H}
---

\newpage


<!-- Source: docs/repositorio.md -->

# Repositório do código {.unnumbered}

Todo o código deste trabalho está disponível publicamente no GitHub:

**<https://github.com/gabriel-q7/computer-vision-p1>**

| Diretório | Conteúdo |
|------------------------------------|----------------------------------------------|
| `atividade-1-projeto-livre/` | Notebook da Atividade 1 (ViT para severidade de acidentes) e capítulo do relatório |
| `atividade-2-clip-publicidade/` | Notebook da Atividade 2 (CLIP *zero-shot* em publicidade) e capítulo do relatório |
| `atividade-3-cnn-pre-treinada/` | Notebook da Atividade 3 (ResNet-50 com extração de características) e capítulo do relatório |
| `atividade-4-estudos-de-caso/` | Notebook da Atividade 4 (COVID-19 com cGAN e análise de tráfego urbano) e capítulo do relatório |
| `scripts/` | Geração deste relatório, extração das figuras e verificação automática dos números |
| `docs/` | Seções comuns do relatório e o PDF gerado |

Os notebooks são executáveis no Google Colab e estão versionados com as saídas das execuções que deram origem aos resultados apresentados aqui. O relatório pode ser regenerado com `./scripts/generate-report.sh`, que confere cada número citado contra essas saídas antes de gerar o PDF.


\newpage


<!-- Source: docs/ai-use-disclosure.md -->

# Declaração de uso de inteligência artificial {.unnumbered}

Este trabalho foi desenvolvido com o apoio de duas ferramentas de inteligência artificial generativa:

- **ChatGPT** (OpenAI), usado no **planejamento** das atividades;
- **Claude Code** (Anthropic), usado como assistente de programação e de redação.

**Como as ferramentas foram utilizadas**

- **Planejamento (ChatGPT):** apoio na interpretação dos enunciados e na organização das etapas de cada atividade.
- **Implementação (Claude Code):** geração e revisão do código dos notebooks (carregamento de dados, treinamento, avaliação e visualização), a partir dos enunciados das atividades.
- **Revisão metodológica (Claude Code):** identificação de erros e problemas metodológicos, como vazamento de dados, critérios de seleção de checkpoint e agrupamento de imagens quase idênticas.
- **Relatório (Claude Code):** organização e redação deste relatório a partir das saídas efetivamente registradas nos notebooks, além do script que consolida o relatório em PDF.

**Responsabilidades do autor**

- Execução dos notebooks no Google Colab e conferência das saídas.
- Revisão crítica do planejamento, do código, das análises e do texto, além das decisões finais sobre o que entra no trabalho.
- Responsabilidade integral pelo conteúdo entregue.

**Integridade dos resultados**

Todos os números, tabelas e figuras de resultados apresentados neste relatório foram extraídos das saídas das execuções dos notebooks; nenhum valor experimental foi produzido ou estimado pelas ferramentas de IA. Quando uma execução apresentou problema metodológico, o problema é relatado em vez de omitido.


\newpage


<!-- Source: docs/introducao.md -->

# Introdução

Este relatório reúne as quatro atividades práticas de Visão Computacional e Deep Learning desenvolvidas no repositório [`computer-vision-p1`](https://github.com/gabriel-q7/computer-vision-p1). Cada atividade foi implementada como um notebook executável no Google Colab e explora uma família diferente de técnicas.

| Atividade | Tema | Técnica principal | Dataset |
|-----------:|------------------|----------------------|------------------|
| 1 | Classificação de severidade de acidentes | Vision Transformer (do zero e pré-treinado) e ResNet-18 | Car Accidents and Deformation Dataset (CADD) |
| 2 | Reconhecimento semântico em publicidade | CLIP (ViT-B/32) em regime *zero-shot* | ADS-16 |
| 3 | Classificador com CNN pré-treinada | ResNet-50 com extração de características | Images Dataset (7 classes) |
| 4 | Estudos de caso | ResNet-18 com transfer learning e GAN condicional; análise de transfer learning em tráfego urbano | COVID-19 Radiography Database |

## Ambiente de execução

Todos os notebooks foram executados no Google Colab com GPU NVIDIA Tesla T4. Os notebooks das Atividades 1 e 4 registram as versões usadas: PyTorch 2.11 e torchvision 0.26 (CUDA 12.8); os das Atividades 2 e 3 não imprimem as versões. Os datasets são públicos e foram obtidos pela API oficial do Kaggle. Cada notebook fixa a semente `42` para Python, NumPy e PyTorch.

## Organização do relatório

Os capítulos das Atividades 1 a 3 seguem a mesma estrutura: objetivo, dados, método, resultados, discussão crítica e limitações. O capítulo da Atividade 4 se divide em duas partes: o estudo experimental de COVID-19 (4.1), que parte do diagnóstico do cenário proposto e termina com plano de melhorias e limitações, e a análise escrita do caso de tráfego urbano (4.2), sem experimentos. Os números citados são os registrados nas saídas dos notebooks versionados no repositório, e as figuras foram extraídas dessas mesmas saídas.

Ao longo do relatório, adota-se uma postura crítica em relação aos próprios resultados: métricas muito altas ou muito baixas são investigadas, e problemas de protocolo (por exemplo, seleção de modelo no mesmo conjunto usado para reportar o desempenho) são apontados explicitamente.


\newpage


<!-- Source: atividade-1-projeto-livre/RELATORIO.md -->

# Atividade 1 — Vision Transformer para Severidade de Acidentes

Notebook: `atividade-1-projeto-livre/vit_accident_severity_colab.ipynb`.

## Objetivo

Construir um pipeline de classificação de imagens em cinco níveis de severidade de acidentes de trânsito e comparar três modelos:

1. **ResNet-18** pré-treinada no ImageNet (baseline convolucional, *fine-tuning* completo);
2. **Vision Transformer (ViT) implementado e treinado do zero**;
3. **ViT-B/16 pré-treinado** no ImageNet-1k (*fine-tuning* completo).

Também se investiga a interpretabilidade do ViT por meio de **Attention Rollout** (Abnar & Zuidema, 2020), comparando os mapas de atenção com as *bounding boxes* anotadas.

## Dados

O **Car Accidents and Deformation Dataset (CADD)** traz anotações no formato YOLO (detecção de objetos) em cinco classes: *Sem Acidente*, *Acidente Leve*, *Acidente Moderado*, *Acidente Severo* e *Perda Total*.

**Rótulo por imagem.** Como a tarefa é de classificação, cada imagem recebe a classe **mais severa** entre os objetos anotados (`label = max(class_ids)`). Essa é uma simplificação: uma imagem com um carro intacto e outro com perda total é rotulada como *Perda Total*.

**Inspeção.** Foram encontradas 1.221 imagens válidas com rótulo; duas imagens da validação original foram descartadas por não terem arquivo de rótulo.

**Divisão com controle de vazamento.** O dataset contém cópias aumentadas das mesmas fotos (sufixos como `_flip`, `_flip30`, `_flip70`). Na divisão original do dataset, **70 grupos** de imagens apareciam simultaneamente no treino e na validação, o que caracteriza vazamento. O notebook remove esses sufixos para obter um identificador de grupo (1.016 grupos) e refaz a divisão 80/20 **por grupo**, garantindo que nenhuma foto e suas variações apareçam nos dois conjuntos.

| Classe | Treino | Validação |
|---|---:|---:|
| Sem Acidente | 48 | 17 |
| Acidente Leve | 245 | 59 |
| Acidente Moderado | 253 | 78 |
| Acidente Severo | 163 | 40 |
| Perda Total | 271 | 47 |
| **Total** | **980** | **241** |

A classe *Sem Acidente* é a menos representada. Para compensar, a função de perda usa **pesos de classe** inversamente proporcionais à frequência (de 0,72 para *Perda Total* a 4,08 para *Sem Acidente*) e **label smoothing** de 0,1.

## Método

Entrada de 224×224 com normalização do ImageNet. No treino, aplicou-se *data augmentation* com `RandomResizedCrop` (escala 0,7–1,0), espelhamento horizontal, rotação de ±15° e `ColorJitter`. Todos os modelos usam AdamW e mixed precision na GPU.

| Modelo | Inicialização | Épocas | LR | Agenda de LR |
|----------------------------|--------------|--------:|------:|----------------------|
| ResNet-18 | ImageNet | 5 | 3e-4 | Cosseno |
| ViT do zero (6 blocos, 6 cabeças, dim. 384, patch 16) | Aleatória | 20 | 3e-4 | Warmup linear (3 ép.) + cosseno |
| ViT-B/16 | ImageNet-1k | 10 | 1e-4 | Warmup linear (2 ép.) + cosseno |

O ViT do zero foi implementado explicitamente: *patch embedding* por convolução com *stride* igual ao *patch*, token `[CLS]`, *embeddings* posicionais aprendidos e blocos *pre-norm* com atenção multi-cabeça calculada como $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$. Em todos os modelos, o melhor estado foi escolhido pela **macro F1 de validação**.

## Resultados

| Modelo | Melhor época | Acurácia | Macro precisão | Macro recall | Macro F1 |
|--------------------------|------:|------:|------:|------:|------:|
| ResNet-18 (ImageNet) | 3 | 0,527 | 0,539 | 0,550 | 0,518 |
| ViT do zero | 19 | 0,253 | 0,316 | 0,290 | 0,253 |
| **ViT-B/16 (ImageNet-1k)** | 8 | **0,568** | **0,578** | **0,606** | **0,576** |

Como referências mínimas: um classificador que sempre prevê a classe mais frequente da validação (*Acidente Moderado*, 78 de 241) obtém **0,324** de acurácia, e o acaso com cinco classes fica em torno de 0,20.

**F1 por classe do ViT-B/16 pré-treinado:** *Sem Acidente* 0,571; *Leve* 0,617; *Moderado* 0,496; *Severo* 0,438; *Perda Total* 0,756.

![Loss de treino e de validação do ViT treinado do zero.](atividade-1-projeto-livre/figures/vit-scratch-loss.png){width=70%}

![Matriz de confusão normalizada por linha do ViT treinado do zero (validação).](atividade-1-projeto-livre/figures/vit-scratch-confusao-normalizada.png){width=62%}

## Interpretabilidade: Attention Rollout

O Attention Rollout propaga as matrizes de atenção (média das cabeças, com a identidade somada para representar as conexões residuais) por todas as camadas e mede quanto cada *patch* contribui para o token `[CLS]`. A figura compara a atenção da última camada, o rollout e as *bounding boxes* reais.

![Attention Rollout do ViT treinado do zero em uma imagem de validação (rótulo real *Perda Total*, previsto *Acidente Severo*).](atividade-1-projeto-livre/figures/vit-scratch-attention-rollout.png)

**Leitura crítica.** No exemplo acima, o rollout concentra a atenção na faixa de **céu e árvores** no canto superior direito, e não nos veículos delimitados pelas caixas. Isso é coerente com o baixo desempenho do ViT do zero: o modelo não aprendeu a focar nos danos e parece usar pistas de fundo. A legenda "Foco Semântico nos Carros" usada no notebook **não** é sustentada por esse exemplo. O rollout foi calculado apenas para o ViT do zero; o ViT-B/16 do torchvision não expõe os pesos de atenção, de modo que a análise não foi repetida para o melhor modelo.

## Discussão

- **ViT do zero não aprende com ~1.000 imagens.** O ViT treinado do zero ficou **abaixo da baseline da classe majoritária** (0,253 contra 0,324). A acurácia de treino chegou a apenas 0,40 em 20 épocas, o que indica **subajuste**, e não sobreajuste. Transformers não têm os vieses indutivos das convoluções (localidade e equivariância à translação) e precisam de muito mais dados ou de pré-treinamento. A matriz de confusão mostra previsões espalhadas, com forte tendência a *Sem Acidente*, provavelmente puxada pelo maior peso dessa classe na perda.
- **Pré-treinamento é o fator decisivo.** O ViT-B/16 pré-treinado teve o melhor resultado (macro F1 0,576), seguido da ResNet-18 (0,518). Com pré-treinamento, o Transformer supera a CNN mesmo com poucos dados.
- **Sobreajuste nos modelos pré-treinados.** Na época do checkpoint (8), o ViT-B/16 tinha 0,865 de acurácia no treino contra 0,568 na validação. A lacuna de ~30 p.p. sugere que regularização adicional, congelamento parcial ou mais dados ajudariam.
- **Classes intermediárias são as mais difíceis.** *Moderado* e *Severo* têm os menores F1, o que é esperado em uma escala ordinal com fronteiras subjetivas entre níveis vizinhos.

## Limitações

1. **Seleção otimista do checkpoint.** O melhor estado de cada modelo foi escolhido pela macro F1 na **mesma validação** usada para reportar os resultados. Os números acima tendem, portanto, a ser otimistas; o correto seria separar uma validação interna para seleção ou usar um conjunto de teste independente.
2. **Uma única semente e uma única divisão.** Com 241 imagens de validação (17 de *Sem Acidente*), as métricas têm variância alta; diferenças de poucos pontos entre modelos não são conclusivas.
3. **Rótulo por agregação.** A regra "classe mais severa" ignora a presença de múltiplos veículos e transforma um problema de detecção em classificação.
4. **Agrupamento heurístico.** O controle de vazamento depende de sufixos de nome de arquivo; cópias aumentadas com outros nomes não seriam detectadas.
5. **Reprodutibilidade parcial.** O notebook usa `cudnn.benchmark = True` e `cudnn.deterministic = False`, de modo que reexecuções na GPU podem variar ligeiramente.
6. **Interpretabilidade limitada.** Apenas três exemplos de rollout, todos do modelo mais fraco.


\newpage


<!-- Source: atividade-2-clip-publicidade/RELATORIO.md -->

# Atividade 2 — Reconhecimento Semântico em Publicidade Visual com CLIP

Notebook: `atividade-2-clip-publicidade/clip_semantic_advertising_colab.ipynb`.

## Objetivo

Usar o **CLIP** (*Contrastive Language-Image Pre-training*, OpenAI) em regime estritamente **zero-shot** (sem treinamento, sem *fine-tuning* e sem atualização de pesos) para:

1. medir a presença de **20 conceitos semânticos** nas imagens do corpus e ordená-los por frequência;
2. implementar uma **busca semântica por texto** (*zero-shot retrieval*) e analisar oito consultas que variam de genéricas a abstratas.

## Dados

O **ADS-16 Computational Advertising Dataset** foi baixado do Kaggle (`groffo/ads16-dataset`). A varredura recursiva encontrou **2.697 imagens** distribuídas em **260 pastas**. Segundo o artigo original (Roffo e Vinciarelli, 2016), o ADS-16 reúne **300 anúncios reais** em 20 categorias de produto e, separadamente, imagens pessoais enviadas pelos 120 participantes do estudo, anotadas como positivas ou negativas. A contagem por tipo de pasta confirma essa composição:

| Tipo de pasta | Pastas | Imagens |
|------------------------------------|------:|------:|
| Anúncios (pastas `1` a `20`) | 20 | 301 |
| Imagens pessoais positivas (`U*-IM-POS`) | 120 | 1.219 |
| Imagens pessoais negativas (`U*-IM-NEG`) | 120 | 1.177 |
| **Total** | **260** | **2.697** |

Portanto, **apenas 11% do corpus são anúncios**; o restante são fotos pessoais dos participantes. O notebook trata todas as 2.697 imagens como anúncios e usa o nome da pasta apenas para exibição. Há ainda 47 arquivos repetidos: o corpus tem 2.650 conteúdos distintos (MD5).

> **Verificação independente.** A composição do corpus, as duplicatas, os percentis de similaridade e a recorrência de imagens nos top-5 não aparecem nas saídas do notebook. Eles foram calculados reexecutando o pipeline localmente (`scripts/verify_clip_activity2.py`, CLIP `ViT-B/32` em CPU). A reexecução reproduziu os mesmos top-5 das oito consultas, a mesma faixa de escores e a tabela de limiares do notebook (pequenas diferenças na 2ª casa decimal decorrem de a execução original usar precisão reduzida na GPU).

## Método

| Etapa | Implementação |
|------------|------------------------------------------|
| Modelo | CLIP `ViT-B/32`, `model.eval()` e `torch.no_grad()` em toda a inferência |
| Embeddings de imagem | Encoder visual do CLIP, 512 dimensões, normalização $L_2$ |
| Conceitos | 20 conceitos (ex.: *a car*, *a person*, *text and logo*, *a luxury product*) |
| Prompt | Template contextual `"an advertisement featuring {}"` |
| Similaridade | $S = X_{img} X_{txt}^\top$ (cosseno, pois os vetores têm norma 1), matriz 2.697 × 20 |
| Presença de conceito | Similaridade $\geq$ limiar operacional |
| Busca | Texto → embedding normalizado → produto escalar com todas as imagens → top-5 |

A similaridade de cosseno do CLIP **não é uma probabilidade**: ela mede proximidade angular no espaço multimodal, e qualquer limiar de "presença" é uma regra heurística.

## Resultados

### Escolha do limiar

| Limiar | Conceitos por imagem (média) | Imagens sem nenhum conceito |
|---:|---:|---:|
| 0,20 | 7,89 | 12,3% |
| 0,25 | 0,44 | 84,2% |
| **0,28** | **0,04** | **98,1%** |
| 0,30 | 0,01 | 99,6% |
| 0,35 | 0,00 | 100,0% |

![Média de conceitos detectados por imagem em função do limiar.](atividade-2-clip-publicidade/figures/threshold-conceitos.png){width=75%}

O notebook fixou o limiar em **0,28**. As similaridades imagem-conceito ocupam uma faixa estreita: 98% dos 53.940 valores (2.697 × 20) ficam entre 0,134 e 0,259 (percentis 1 e 99), e o máximo é 0,338. Por isso, entre 0,20 e 0,25 a média despenca de quase 8 para menos de 0,5 conceito por imagem.

### Ranking de conceitos (limiar 0,28)

| Posição | Conceito | Similaridade média | Imagens acima do limiar | Frequência |
|----:|--------------------|----------:|----------:|--------:|
| 1 | a fashion product | 0,1996 | 18 | 0,67% |
| 2 | a person | 0,2076 | 9 | 0,33% |
| 3 | a household product | 0,1986 | 9 | 0,33% |
| 4 | a woman | 0,1958 | 8 | 0,30% |
| 5 | a luxury product | 0,2035 | 7 | 0,26% |
| ... | ... | ... | ... | ... |
| 20 | a building | 0,1765 | 0 | 0,00% |

![Frequência dos 20 conceitos no corpus com limiar 0,28.](atividade-2-clip-publicidade/figures/ranking-conceitos.png){width=75%}

### Busca semântica

Foram executadas oito consultas: genéricas (*a product advertisement*), concretas (*a person*, *a car*), específicas (*a person using a smartphone*, *a luxury car in an urban environment*, *a cold drink being advertised*) e abstratas (*a premium lifestyle*, *an advertisement designed to attract attention*). Os escores do top-5 ficaram entre 0,264 e 0,338.

![Top-5 para a consulta concreta "a car".](atividade-2-clip-publicidade/figures/busca-a-car.png)

![Top-5 para a consulta específica "a luxury car in an urban environment".](atividade-2-clip-publicidade/figures/busca-luxury-car-urban.png)

![Top-5 para a consulta abstrata "a premium lifestyle".](atividade-2-clip-publicidade/figures/busca-premium-lifestyle.png)

**O que as figuras mostram** (inspeção visual das saídas):

- **Consultas concretas de objeto funcionam bem.** Em *a car*, as cinco imagens são anúncios da categoria automotiva (Fiat 500, BMW, acessórios Raven, peças Renault). Uma delas, porém, é um anúncio **apenas de texto** ("Cheapest Car Parts Online"), sem nenhum carro: o CLIP também "lê" o texto da imagem.
- **Consultas específicas funcionam parcialmente.** *A luxury car in an urban environment* recupera um Maserati (duas vezes), um BMW e o Fiat 500, mas o fundo urbano raramente está presente; a 5ª imagem é o interior de um carro em um túnel. Três dos cinco resultados (os dois Maserati e o túnel) são **fotos pessoais de participantes**, não anúncios.
- **Consultas abstratas são fracas.** *A premium lifestyle* retorna um anúncio do aplicativo de relacionamento Badoo, um da American Apparel, um anúncio de app de encontros, uma imagem de campanha da PETA (enviada por um participante, não um anúncio do corpus) e um anúncio do McDonald's. Apenas parte desses resultados pode ser associada a "estilo de vida premium".

## Discussão

- **O limiar 0,28 é restritivo demais para contar conceitos.** Com ele, 98,1% das imagens não têm nenhum conceito, e o conceito mais frequente aparece em apenas 18 de 2.697 imagens. O ranking passa a depender de poucas imagens e é instável: mudar a similaridade de poucas imagens altera as posições. Note que a ordem por **similaridade média** é diferente (*a person* teria a maior média, 0,2076). Alternativas mais robustas seriam: limiar por conceito (por exemplo, um percentil da distribuição de cada conceito), atribuição *top-k* por imagem ou comparação relativa entre conceitos (softmax sobre os 20 *prompts*).
- **Fenômeno de *hubness*.** Um pequeno conjunto de imagens domina os resultados de consultas sem relação entre si. O anúncio do **Fiat 500** aparece no top-5 de **6 das 8** consultas (inclusive *a person* e *a person using a smartphone*), o do **Badoo** em 5 das 8, e o da **Kaspersky** em 4. Curiosamente, os quatro anúncios mais recorrentes são todos o arquivo `11.png` de suas categorias (2, 20, 17 e 10), o que sugere um mesmo formato de peça. Esses "hubs" são anúncios prototípicos (produto, pessoa e texto em destaque) que ficam próximos de muitos textos no espaço do CLIP. Isso explica por que a busca parece razoável em consultas genéricas e falha em consultas que exigem detalhes.
- **Efeito do template de prompt.** A função de busca aplica o mesmo template dos conceitos às consultas. Assim, *a product advertisement* se torna "an advertisement featuring a product advertisement", e todas as consultas são puxadas na direção de "anúncio típico", o que reforça o *hubness*. Comparar a busca com e sem o template seria um experimento simples e informativo.
- **Leitura de texto (OCR implícito).** O CLIP associa palavras escritas na imagem ao conceito, como em "Car Parts", "Smartphone offers" e "Freshly Brewed Coffee" (este último aparece em *a cold drink being advertised*). Isso é útil para anúncios, mas também é uma vulnerabilidade conhecida a "ataques tipográficos".
- **Mistura de anúncios e fotos pessoais.** Como o corpus inclui as imagens pessoais dos participantes, 8 das 40 posições dos top-5 são fotos pessoais, e o ranking de conceitos descreve sobretudo essas fotos (89% do corpus), não a publicidade. Para responder à pergunta da atividade, a análise deveria se restringir às 301 imagens das pastas de anúncios.
- **Duplicatas no corpus.** Há 47 arquivos repetidos. As latas de refrigerante em *cold drink* são uma cópia exata (mesmo MD5) e ocupam duas vagas do top-5; o Maserati em *luxury car* aparece duas vezes como miniatura e original (conteúdos diferentes, mesma foto). Uma deduplicação por hash exato e perceptual antes da indexação resolveria.
- **Divergência com a tabela qualitativa do notebook.** A tabela de "comportamento observado" do notebook descreve resultados mais favoráveis do que os exibidos nas figuras (por exemplo, "precisão de 100%" para *a person*, cujo top-5 inclui o anúncio do Fiat 500 sem pessoa visível). Este relatório adota a descrição baseada nas figuras efetivamente geradas.

## Limitações

1. **Sem *ground truth*.** Não há anotação de presença de conceitos nem relevância por consulta; toda a avaliação é qualitativa. Uma rotulagem manual de uma amostra permitiria medir *precision@k*.
2. **Limiar heurístico global**, aplicado igualmente a conceitos com distribuições de similaridade diferentes.
3. **Vieses do pré-treinamento.** O CLIP herda associações culturais dos dados da web (o que é "luxo", "beleza", "atenção"), o que afeta diretamente as consultas abstratas.
4. **Embedding global.** Um único vetor por imagem não localiza objetos; composições como "pessoa *usando* smartphone" são resolvidas apenas aproximadamente.


\newpage


<!-- Source: atividade-3-cnn-pre-treinada/RELATORIO.md -->

# Atividade 3 — Classificador com CNN Pré-treinada

Notebook: `atividade-3-cnn-pre-treinada/cnn_feature_extraction_colab.ipynb`.

## Objetivo

Construir um baseline de **transfer learning por extração de características**: uma ResNet-50 pré-treinada no ImageNet é usada como extratora congelada, e apenas uma nova camada linear de classificação é treinada. O baseline não usa *data augmentation*, para servir de referência limpa às propostas de melhoria.

## Dados

**Images Dataset** (Kaggle, `pavansanagapati/images-dataset`): 3.606 arquivos de imagem (`.jpg`, `.png` e `.bmp`) em 7 classes, sem nenhum arquivo corrompido.

### Remoção de duplicatas

Uma primeira execução do notebook atingiu **100% de acurácia** na validação. Um resultado perfeito exige verificação, e a inspeção dos caminhos revelou duas estruturas paralelas, `data/<classe>` e `data/data/<classe>`, com as mesmas contagens por classe. A checagem por hash MD5 confirmou que o dataset vem **inteiramente duplicado**. Ao reproduzir fora do notebook a divisão por arquivo usada inicialmente (mesma semente), verificou-se que **568 das 722 imagens de validação (78,7%) tinham uma cópia exata no treino**. Aquela acurácia media memorização, não generalização.

O notebook foi corrigido para remover as repetições **antes** da divisão:

| Etapa | Imagens |
|---|---:|
| Arquivos válidos | 3.606 |
| Removidos por duplicata exata (mesmo MD5) | −1.842 |
| Removidos por quase duplicata (dHash 16×16, distância ≤ 10 bits) | −19 |
| Removidos por rótulo ambíguo (mesmo conteúdo em classes diferentes) | 0 |
| **Imagens distintas mantidas** | **1.745** |

As 19 quase duplicatas são a mesma foto em outra resolução ou compressão; os pares foram conferidos visualmente.

### Divisão treino/validação

A divisão das imagens distintas é estratificada (80/20, semente 42). Ao final, o notebook verifica que nenhum MD5 se repete entre treino e validação e que a imagem de validação mais parecida com alguma do treino ainda está a 12 bits de distância no hash perceptual, acima do limiar de 10.

| Classe | Total | Treino | Validação |
|---|---:|---:|---:|
| bike | 360 | 288 | 72 |
| cars | 420 | 336 | 84 |
| cats | 202 | 162 | 40 |
| dogs | 201 | 161 | 40 |
| flowers | 210 | 168 | 42 |
| horses | 178 | 142 | 36 |
| human | 174 | 139 | 35 |
| **Total** | **1.745** | **1.396** | **349** |

![Distribuição das imagens distintas por classe.](atividade-3-cnn-pre-treinada/figures/distribuicao-classes.png){width=75%}

## Método

| Item | Configuração |
|------------|------------------------------------------|
| Backbone | ResNet-50, pesos `ResNet50_Weights.DEFAULT` (ImageNet), **todas as camadas congeladas** |
| Cabeça | `Linear(2048 → 7)`: 14.343 parâmetros treináveis de 23,5 milhões |
| Pré-processamento | `Resize(224)`, `CenterCrop(224)`, normalização ImageNet; sem augmentation |
| Treinamento | Entropia cruzada, Adam (lr 1e-3) apenas na cabeça, lote 32, 10 épocas |
| Seleção | Melhor acurácia de validação |

## Resultados

| Época | Loss treino | Acurácia treino | Loss validação | Acurácia validação |
|---:|---:|---:|---:|---:|
| 1 | 0,7637 | 0,8639 | 0,2347 | 0,9943 |
| 5 | 0,0463 | 0,9964 | 0,0487 | 0,9943 |
| 7 | 0,0340 | 0,9964 | 0,0375 | **0,9971** |
| 10 | 0,0200 | 0,9986 | 0,0291 | 0,9971 |

![Curvas de loss e acurácia por época.](atividade-3-cnn-pre-treinada/figures/curvas-treinamento.png){width=90%}

O melhor checkpoint (época 7) classificou corretamente **348 das 349 imagens** de validação: **acurácia de 0,9971**, com intervalo de confiança de Wilson (95%) de **[0,984; 0,9995]**. Todas as classes têm F1 de pelo menos 0,987 (o menor é o de *cats*, com recall de 39/40). A macro F1 e as médias de precisão e recall arredondam para 1,00 com duas casas decimais.

![Matriz de confusão na validação.](atividade-3-cnn-pre-treinada/figures/matriz-confusao.png){width=60%}

O único erro é um **gato visto de costas**, sem o rosto visível, sobre um cobertor colorido, classificado como cachorro. É uma confusão plausível: sem a face, a pelagem e a postura são as principais pistas, e elas são compartilhadas pelas duas classes.

![Única classificação incorreta da validação.](atividade-3-cnn-pre-treinada/figures/erro-classificacao.png){width=90%}

## Discussão

- **A tarefa é fácil para features do ImageNet.** As sete classes (bicicletas, carros, gatos, cães, flores, cavalos e pessoas) correspondem a conceitos amplamente representados no pré-treinamento e são linearmente separáveis no espaço de 2.048 dimensões da ResNet-50. A acurácia de validação já é de 0,994 na primeira época.
- **A correção mudou pouco o número, mas mudou o que ele significa.** A acurácia caiu de 1,000 para 0,997. A diferença é pequena porque o problema é fácil, mas o novo valor mede generalização para imagens nunca vistas, enquanto o anterior era dominado por cópias do treino. A validação ficou menor (349 contra 722 imagens), e o intervalo de confiança reflete isso.
- **Não há sinal de sobreajuste.** A loss de validação acompanha a de treino, e a lacuna de acurácia é inferior a 0,2 p.p.
- **Lição metodológica.** Deduplicar antes de dividir deve ser rotina, sobretudo quando o resultado é "bom demais".

## Propostas de melhoria

O notebook discute técnicas para experimentos posteriores, mantendo o baseline intocado:

| Técnica | Benefício esperado | Risco |
|------------------|------------------|------------------|
| Augmentation geométrica (flip horizontal, rotações e crops pequenos) | Variedade de enquadramento; menos sobreajuste | Alterar a semântica de objetos orientados |
| Augmentation de cor (`ColorJitter` conservador) | Robustez a iluminação e câmera | Apagar pistas de cor relevantes |
| Variação de escala (`RandomResizedCrop` moderado) | Robustez a distância e enquadramento | Cortar o objeto |
| Normalização ImageNet | Compatibilidade com o pré-treinamento | Já utilizada; não é augmentation |

Com um baseline de 99,7%, o ganho dessas técnicas não é mensurável nesta validação: restaria no máximo um acerto a conquistar. Para avaliá-las, seria preciso um conjunto mais difícil, por exemplo com imagens de outra fonte, objetos parcialmente visíveis ou fundos complexos.

## Limitações

1. O checkpoint foi selecionado pela acurácia na mesma validação usada para reportar o resultado. O efeito aqui é pequeno, pois as épocas 7 a 10 variam em uma única imagem, mas o protocolo correto separa uma validação interna.
2. Uma única semente e nenhum conjunto de teste independente.
3. A validação vem da mesma fonte do treino; o desempenho em imagens de outras origens não foi medido.
4. A deduplicação por hash perceptual não detecta fotos diferentes da mesma cena ou do mesmo animal.


\newpage


<!-- Source: atividade-4-estudos-de-caso/RELATORIO.md -->

# Atividade 4 — Estudos de Caso em Deep Learning

Notebook: `atividade-4-estudos-de-caso/covid_cgan_case_study_colab.ipynb`.

## Atividade 4.1 — Detecção de COVID-19 em radiografias

### Cenário e objetivo

O enunciado descreve um classificador de radiografias de tórax com 1.200 imagens (840 *Normal*, 240 *Viral Pneumonia*, 120 *COVID-19*). Esse classificador usa uma ResNet-18 treinada do zero, SGD com taxa de aprendizado fixa de 0,01, 15 épocas, nenhum *data augmentation*, divisão 80/20 não estratificada e avaliação apenas por acurácia. Os resultados relatados no cenário foram 93% de acurácia no treino e 61% na validação, e a equipe clínica percebe que o modelo "ignora" os casos de COVID-19. Esses números pertencem ao enunciado e não foram reproduzidos aqui.

O objetivo é montar um experimento controlado com a mesma composição de classes e testar se imagens sintéticas de COVID-19, geradas por uma **GAN condicional (cGAN)**, melhoram a detecção da classe minoritária. A métrica primária é o **recall de COVID-19**, $TP/(TP+FN)$, sempre lida em conjunto com a precisão, o F1, a macro F1 e a matriz de confusão.

### Diagnóstico do cenário original

| # | Problema | Status | Consequência |
|--:|--------------------------|------------|------------------------------------|
| 1 | Desbalanceamento (70/20/10%) | Comprovado | Prever sempre *Normal* já rende 70% de acurácia; muitos falsos negativos de COVID |
| 2 | Divisão não estratificada | Comprovado | Validação com poucos casos de COVID; recall instável |
| 3 | ResNet-18 do zero com ~960 imagens | Comprovado | Memorização, coerente com a lacuna de 32 p.p. entre treino e validação |
| 4 | Sem *data augmentation* | Comprovado | Menor robustez a posicionamento e exposição |
| 5 | Taxa de aprendizado fixa, sem ajuste | Comprovado | Convergência subótima; a última época pode não ser a melhor |
| 6 | Apenas acurácia global | Comprovado | Esconde o recall quase nulo da classe de interesse |
| 7 | Sem conjunto de teste independente | Comprovado | Estimativas otimistas |
| 8 | Vazamento entre pacientes | Risco | Métricas infladas se o mesmo paciente estiver nos dois conjuntos |
| 9 | Atalhos visuais (fonte, equipamento, marcações) | Risco, com indício nos metadados | O modelo pode aprender a origem da imagem, e não a doença |
| 10 | Sem validação externa | Comprovado | Generalização desconhecida para outros hospitais e populações |

### Dados

O **COVID-19 Radiography Database** foi baixado pela API do Kaggle sem credenciais. Considerando apenas as três classes de interesse, havia 15.153 imagens PNG de 299×299 pixels: 10.192 *Normal*, 1.345 *Viral Pneumonia* e 3.616 *COVID*. Nenhuma estava corrompida, e 54 duplicatas exatas (MD5) foram removidas antes da seleção. Os metadados **não** trazem identificador de paciente, mas revelam a **fonte** de cada imagem:

| Fonte | COVID | Normal | Viral Pneumonia |
|--------------------------------|------:|------:|------:|
| `kaggle.com` | 0 | 10.192 | 1.345 |
| `bimcv.cipf.es` | 2.474 | 0 | 0 |
| Outras (GitHub, Eurorad, SIRM) | 1.142 | 0 | 0 |

**Todas** as imagens de COVID vêm de repositórios diferentes dos usados para *Normal* e *Viral Pneumonia*. Classe e origem estão, portanto, completamente confundidas, o que torna real o risco de o classificador aprender a "assinatura" da fonte (equipamento, pós-processamento, marcações) em vez de achados clínicos.

A amostra experimental (840/240/120) foi sorteada com semente fixa. A proporção 70/20/10 foi imposta pelo cenário e difere da distribuição original (67/9/24).

![Distribuição de classes: dataset original × amostra experimental.](atividade-4-estudos-de-caso/figures/distribuicao-original-vs-amostra.png){width=65%}

**Divisão com controle de vazamento.** Como não há ID de paciente, a divisão por paciente não foi possível. Como mitigação, imagens quase idênticas foram agrupadas por hash perceptual (dHash 16×16, distância de Hamming ≤ 10 bits) antes da divisão. Nenhum par atingiu o limiar (a menor distância entre conjuntos foi de 12 bits), de modo que cada imagem formou seu próprio grupo. Uma primeira versão com hash 8×8 encadeou centenas de radiografias distintas em um único grupo e produziu uma validação com uma única imagem. O erro foi detectado nas saídas e corrigido, e o notebook passou a interromper a execução se os grupos ou a divisão fugirem do esperado.

| Conjunto | Normal | Viral Pneumonia | COVID | Total | Uso |
|--------------|------:|------:|------:|------:|------------------------------|
| Treino de ajuste | 571 | 163 | 82 | 816 | Treinar o classificador e a cGAN |
| Validação interna | 101 | 29 | 14 | 144 | Selecionar o checkpoint |
| Validação real | 168 | 48 | 24 | 240 | Apenas a avaliação final de A e B |

### Método

**Classificador (idêntico nos Experimentos A e B).** ResNet-18 pré-treinada no ImageNet com o backbone congelado (extração de características); apenas a camada final `Linear(512 → 3)` é treinada (1.539 parâmetros). Entrada de 224×224, canal cinza replicado em 3 canais, normalização do ImageNet. No treino, transformação afim moderada (rotação de ±7°, translação de 4%, escala 0,95–1,05), **sem espelhamento horizontal**, porque inverter a lateralidade cria anatomia implausível. Perda de entropia cruzada sem pesos de classe, Adam (lr 1e-3, *weight decay* 1e-4) com decaimento cosseno, lote 32, 15 épocas. O checkpoint é escolhido pela **macro F1 na validação interna**, critério fixado antes do treinamento. A mesma semente garante inicialização idêntica nos dois experimentos, o que foi conferido pela soma dos pesos iniciais da camada final.

**GAN condicional.** Arquitetura no estilo DCGAN em 64×64, treinada **somente** com o treino de ajuste (a ausência de imagens de validação é verificada por *assert*), com amostragem balanceada por classe.

| Componente | Configuração |
|------------|--------------------------------------------|
| Gerador | $z \in \mathbb{R}^{100}$ + *embedding* da classe (50) → Linear → 512×4×4 → 4 ConvT (BN + ReLU) → 1×64×64, Tanh |
| Discriminador | Imagem + mapa da classe (2 canais) → 4 Conv com normalização espectral e LeakyReLU → Linear |
| Perda | BCE com logits (não saturante), *label smoothing* de 0,9 no rótulo real |
| Otimização | Adam, lr 2e-4, $\beta = (0{,}5;\ 0{,}999)$, lote 64, 400 épocas (4.800 iterações, 9 min na T4) |

Foram geradas **120 imagens sintéticas de COVID-19**. O Experimento B usa o mesmo treino real do A, acrescido dessas imagens, o que eleva a classe COVID de 82 para 202 exemplos (21,6% do treino).

### Imagens sintéticas

![Amostras da cGAN na época 400, com ruído fixo (linhas = classe condicionante).](atividade-4-estudos-de-caso/figures/cgan-amostras-epoca-400.png)

![Sintéticas de COVID-19 (acima) e reais de COVID-19 do treino reduzidas para 64×64 (abaixo).](atividade-4-estudos-de-caso/figures/sinteticas-vs-reais.png)

| Indicador (64×64) | Sintéticas | Reais (COVID) | Razão |
|------------------------------------|--------:|--------:|------:|
| Distância média entre pares (diversidade) | 12,78 | 16,38 | 0,78 |
| Distância mediana à imagem real mais próxima | 7,10 | 8,85 | 0,80 |
| Desvio-padrão da intensidade | 0,198 | 0,234 | — |
| Pares de sintéticas quase idênticos | 0 | — | — |

**Inspeção.** As sintéticas reproduzem a forma geral do tórax (campos pulmonares escuros, mediastino claro e bordas escuras), mas são borradas, têm textura granulada típica de convoluções transpostas e não mostram costelas nem clavículas nítidas. As sintéticas de COVID são, em geral, mais "lavadas" e difusas do que as das outras classes. Isso pode refletir as opacidades presentes nos exemplos reais, mas também pode ser apenas perda de contraste: a variância de intensidade é menor que a das reais. A diversidade é cerca de 22% menor que a das imagens reais (possível colapso parcial de modos), e não há sinal de cópia direta do treino. A partir da época ~250, a saída do discriminador para imagens reais e falsas se afasta progressivamente, indício de que ele começa a sobreajustar o pequeno conjunto real. **Nenhuma imagem sintética foi avaliada por radiologista, e aparência plausível não equivale a validade clínica.**

![Perdas da cGAN e saída média do discriminador por época.](atividade-4-estudos-de-caso/figures/cgan-losses.png){width=90%}

### Resultados: Experimento A × Experimento B

Ambos os modelos foram avaliados uma única vez na validação real (240 imagens reais, 24 de COVID-19).

| Métrica | Experimento A | Experimento B | Diferença |
|------------------------|-------------:|-------------:|---------------------:|
| Acurácia global | 0,863 | 0,883 | +2,1 p.p. |
| Macro F1 | 0,663 | 0,736 | +7,2 p.p. |
| **Recall COVID-19** | **0,125** (3/24) | **0,250** (6/24) | **+12,5 p.p.** (+100% relativo) |
| Precisão COVID-19 | 1,000 | 1,000 | 0,0 p.p. |
| F1-score COVID-19 | 0,222 | 0,400 | +17,8 p.p. |
| Falsos negativos COVID | 21 | 18 | −3 |
| Falsos positivos COVID | 0 | 0 | 0 |
| Checkpoint (época) | 10 | 13 | — |

A diferença absoluta de recall é de **12,5 pontos percentuais**. A diferença relativa (+100%) parece grande apenas porque a base é muito baixa: corresponde a **três casos adicionais** detectados.

![Matrizes de confusão na validação real.](atividade-4-estudos-de-caso/figures/confusao-a-vs-b.png)

![Evolução do treinamento: loss, macro F1 e recall de COVID na validação interna.](atividade-4-estudos-de-caso/figures/historico-a-vs-b.png)

**Incerteza.** Com 24 casos positivos, cada caso vale 4,2 p.p. de recall. Os intervalos de Wilson (95%) se sobrepõem amplamente: A em [0,043; 0,310] e B em [0,120; 0,449]. O *bootstrap* pareado (2.000 reamostragens) dá IC95% de **[+0,0; +28,6] p.p.** para a diferença de recall e de [+0,010; +0,144] para a diferença de macro F1. O recall de B supera o de A em 95% das reamostragens.

### Discussão

1. **A geração sintética aumentou o recall de COVID-19?** Nesta execução, sim: de 0,125 para 0,250 (+12,5 p.p.), com 18 em vez de 21 falsos negativos. Na validação interna, o efeito foi semelhante (recall de 0,43 para 0,64 e macro F1 de 0,79 para 0,87).
2. **Houve perda de precisão ou aumento de falsos positivos?** Não. Os dois modelos nunca previram COVID para imagens de outras classes (precisão de 1,0). O ganho não veio de "chutar COVID" com mais frequência.
3. **A acurácia global melhorou?** Ligeiramente (+2,1 p.p.). Também melhorou o recall de *Viral Pneumonia* (de 0,750 para 0,792).
4. **Há melhor equilíbrio entre classes?** Moderadamente: a macro F1 subiu 7,2 p.p. Ainda assim, **75% dos casos de COVID continuam classificados como *Normal***, e nenhum foi confundido com *Viral Pneumonia*. O desempenho na classe de interesse permanece clinicamente inaceitável.
5. **A qualidade das sintéticas explica os resultados?** Em parte. As sintéticas aumentam o peso da classe COVID na perda, de 10% para 22% do treino. Parte do ganho pode vir apenas desse **rebalanceamento**, que pesos de classe ou reamostragem também produziriam, e não de informação visual nova. Há ainda o risco de um atalho "imagem borrada ⇒ COVID", já que as sintéticas têm 64×64 pixels e são ampliadas para 224×224. Como a validação contém apenas imagens reais, esse atalho não explica diretamente os acertos, mas pode limitar o ganho.
6. **A evidência é robusta?** Não. O intervalo de confiança da diferença de recall encosta em zero, há uma única semente e uma única divisão, não há teste externo, e a confusão entre classe e fonte dos dados impede afirmar que o modelo reconhece a doença. O resultado é um **indício favorável e preliminar**, não uma conclusão.

### Plano integrado de melhorias

**Baixo custo (próximo experimento, sem novos dados)**

1. Repetir A e B com várias sementes e com validação cruzada estratificada, reportando média, desvio e intervalos de confiança.
2. Comparar B com alternativas baratas de balanceamento: *class weights* e `WeightedRandomSampler`. Se igualarem o ganho, a cGAN não se justifica.
3. Fazer análise de limiar e calibração (curvas PR, *temperature scaling*). Com precisão de 1,0, há margem para reduzir o limiar de COVID e trocar alguns falsos positivos por menos falsos negativos.
4. Fazer *fine-tuning* parcial (descongelar a `layer4`), já que features do ImageNet não foram aprendidas em radiografias.
5. Investigar atalhos: Grad-CAM, treino com pulmões segmentados (máscaras do próprio dataset) e um classificador que tente prever a **fonte** da imagem.
6. Melhorar a cGAN: maior resolução, DiffAugment ou ADA, varredura de `N_SYNTHETIC` usando apenas a validação interna, e um controle que degrade imagens reais para 64×64.

**Exigem novos dados ou validação clínica**

1. Divisão por paciente, com identificadores reais.
2. COVID e controles vindos **das mesmas instituições e equipamentos**.
3. Conjunto de teste independente e validação externa multicêntrica.
4. Revisão das previsões e das imagens sintéticas por radiologistas.
5. Documentação de limitações e riscos (*model card*).

> Nenhum modelo desta atividade está apto a apoiar diagnóstico médico.

### Limitações

1. Sem identificador de paciente, o agrupamento por hash perceptual reduz, mas não elimina, o vazamento entre pacientes.
2. A validação tem apenas 24 casos de COVID, e cada caso move o recall em 4,2 p.p.
3. Os resultados vêm de uma única semente e de uma única divisão.
4. Classe e fonte dos dados estão completamente confundidas (COVID vem de repositórios distintos de *Normal* e *Viral Pneumonia*).
5. A cGAN em 64×64 foi treinada com 82 radiografias reais de COVID, sem métrica de qualidade validada para imagens médicas e sem revisão clínica.
6. O baseline com transfer learning **não** reproduz a ResNet-18 treinada do zero do cenário; os números não são comparáveis aos 93%/61% do enunciado.
7. A amostra tem proporções artificiais (70/20/10), diferentes do dataset e de qualquer prevalência clínica.

## Atividade 4.2 — Transfer learning para análise de tráfego urbano

### Cenário

Uma ResNet-50 pré-treinada no ImageNet passa por *fine-tuning* com 800 *frames* rotulados de 12 câmeras urbanas, em três classes (tráfego livre, moderado e congestionado). O modelo atinge 78% de acurácia na validação, mas falha sistematicamente à noite, sob chuva e em ângulos de câmera não vistos no treino.

### Problemas e impacto operacional

| Problema | Impacto operacional |
|------------------------------|------------------------------|
| **Poucos dados e baixa diversidade**: ~67 *frames* por câmera, provavelmente concentrados em dias claros | O modelo falha justamente nas condições críticas para a gestão do tráfego (chuva, noite, incidentes) |
| **Dependência entre *frames***: *frames* consecutivos da mesma câmera são quase idênticos; uma divisão aleatória coloca o "mesmo instante" no treino e na validação | Os 78% são otimistas e medem memorização da cena |
| **Mudança de domínio**: cada câmera tem perspectiva, lente e fundo próprios; noite (faróis, ruído) e chuva (reflexos, gotas) mudam a distribuição dos pixels | Degradação silenciosa ao instalar câmeras novas ou quando o clima muda |
| **Sem avaliação por condição operacional** | As falhas foram descobertas em operação, não na validação |
| **Classes ambíguas**: "moderado" é uma faixa contínua e depende do ângulo e da distância | Ruído de rótulo e indicadores inconsistentes entre câmeras |
| **Apenas acurácia global** | Pode esconder recall baixo de "congestionado", a classe mais relevante |
| **Sem avaliação temporal ou externa** | Comportamento desconhecido em outras semanas, estações, obras ou câmeras |
| **Sem adaptação de domínio nem monitoramento** | A deriva (câmera reposicionada, lente suja) passa despercebida |

### Medidas corretivas

1. Separar treino e validação **por câmera**, com avaliação *leave-one-camera-out*, e usar **divisão temporal** para evitar *frames* vizinhos em conjuntos diferentes.
2. Coletar dados **reais** de noite, chuva e novos ângulos, com amostragem estratificada por câmera, período e clima, e aprendizado ativo para priorizar o que rotular.
3. Definir **critérios objetivos de rótulo** (ocupação, densidade por faixa, fila em uma região fixa), com exemplos de referência, dupla anotação e medida de concordância (kappa).
4. Reportar **precisão, recall e F1 por classe, macro F1 e matriz de confusão**, com destaque para o recall de "congestionado".
5. **Avaliar por fatia**: câmera, período do dia e condição climática, com um desempenho mínimo exigido em cada fatia.
6. Usar ***data augmentation*** (brilho, gamma, ruído, desfoque, gotas e reflexos simulados) e **adaptação de domínio** (ajuste de BatchNorm ao domínio-alvo, DANN, pseudo-rótulos) quando justificadas, sempre validadas em dados reais.
7. **Monitorar após a implantação**: deriva de dados e de confiança, auditorias periódicas com rótulos humanos, comparação com sensores independentes e retreinamento versionado.

### Por que 78% de acurácia não garante desempenho confiável

A acurácia é uma **média** sobre a distribuição da validação. Se a validação compartilha câmeras, ângulos, dias e condições com o treino, ela mede a capacidade de reconhecer cenas já vistas, e não de operar em condições novas. O número também não mostra **onde** estão os 22% de erro. As falhas sistemáticas observadas indicam que os erros se concentram justamente nas fatias mais críticas (noite, chuva, câmeras novas). Sob mudança de domínio, o desempenho em uma distribuição não é evidência sobre outra: em uma noite chuvosa numa câmera nova, a acurácia real pode ser muito inferior a 78%. Confiabilidade operacional exige avaliação por câmera, por condição e ao longo do tempo, além de monitoramento contínuo.


\newpage


<!-- Source: docs/referencias.md -->

# Referências {.unnumbered}

**Artigos**

- ABNAR, S.; ZUIDEMA, W. Quantifying Attention Flow in Transformers. *Proceedings of ACL*, 2020.
- CHOWDHURY, M. E. H. et al. Can AI help in screening Viral and COVID-19 pneumonia? *IEEE Access*, v. 8, p. 132665–132676, 2020.
- DOSOVITSKIY, A. et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR*, 2021.
- GOODFELLOW, I. et al. Generative Adversarial Nets. *NeurIPS*, 2014.
- HE, K.; ZHANG, X.; REN, S.; SUN, J. Deep Residual Learning for Image Recognition. *CVPR*, 2016.
- MIRZA, M.; OSINDERO, S. Conditional Generative Adversarial Nets. *arXiv:1411.1784*, 2014.
- MIYATO, T. et al. Spectral Normalization for Generative Adversarial Networks. *ICLR*, 2018.
- RADFORD, A.; METZ, L.; CHINTALA, S. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. *ICLR*, 2016.
- RADFORD, A. et al. Learning Transferable Visual Models From Natural Language Supervision. *ICML*, 2021.
- RAHMAN, T. et al. Exploring the effect of image enhancement techniques on COVID-19 detection using chest X-ray images. *Computers in Biology and Medicine*, v. 132, 104319, 2021.
- ROFFO, G.; VINCIARELLI, A. Personality in Computational Advertising: A Benchmark. *International Workshop on Emotions and Personality in Personalized Systems (EMPIRE 2016)*, ACM RecSys, Boston, 2016. CEUR-WS, v. 1680.
- ZECH, J. R. et al. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study. *PLOS Medicine*, v. 15, n. 11, e1002683, 2018.

**Datasets**

- Car Accidents and Deformation Dataset (Annotated). Kaggle: `marslanarshad/car-accidents-and-deformation-datasetannotated`.
- ADS-16 Computational Advertising Dataset. Kaggle: `groffo/ads16-dataset`.
- Images Dataset. Kaggle: `pavansanagapati/images-dataset`.
- COVID-19 Radiography Database. Kaggle: `tawsifurrahman/covid19-radiography-database`.

**Outras fontes**

- Meegle. Transfer Learning for Traffic Analysis. Disponível em: <https://www.meegle.com/en_us/topics/transfer-learning/transfer-learning-for-traffic-analysis>.
- Documentação do PyTorch e do torchvision: <https://pytorch.org/docs/>.


\newpage
