---
title: "Deep Learning and Computer Vision"
subtitle: "Relatório das Atividades 1 a 4"
author: "Gabriel Aguiar Aquino"
date: "11 de outubro de 2026"
lang: pt-BR
geometry: margin=2.5cm
toc: true
numbersections: true
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{H}
---

\newpage


<!-- Source: docs/ai-use-disclosure.md -->

# Declaração de uso de inteligência artificial {.unnumbered}

Este trabalho foi desenvolvido com o apoio de uma ferramenta de inteligência artificial generativa, o **Claude Code** (Anthropic), utilizado como assistente de programação e de redação.

**Como a ferramenta foi utilizada**

- Geração e revisão de código dos notebooks (carregamento de dados, treinamento, avaliação e visualização), a partir dos enunciados das atividades.
- Identificação de erros e problemas metodológicos, como vazamento de dados, critérios de seleção de checkpoint e agrupamento de imagens quase idênticas.
- Organização e redação deste relatório a partir das saídas efetivamente registradas nos notebooks, além do script que consolida o relatório em PDF.

**Responsabilidades do autor**

- Execução dos notebooks no Google Colab e conferência das saídas.
- Revisão crítica do código, das análises e do texto, além das decisões finais sobre o que entra no trabalho.
- Responsabilidade integral pelo conteúdo entregue.

**Integridade dos resultados**

Todos os números, tabelas e figuras de resultados apresentados neste relatório foram extraídos das saídas das execuções dos notebooks; nenhum valor experimental foi produzido ou estimado pela ferramenta de IA. Quando uma execução apresentou problema metodológico, o problema é relatado em vez de omitido.


\newpage


<!-- Source: docs/introducao.md -->

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
|------------------------------|-----------|------:|------:|----------------------|
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
- **Sobreajuste nos modelos pré-treinados.** O ViT-B/16 atingiu 0,894 de acurácia no treino contra 0,568 na validação. A lacuna de ~33 p.p. sugere que regularização adicional, congelamento parcial ou mais dados ajudariam.
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

1. medir a presença de **20 conceitos semânticos** em anúncios publicitários e ordená-los por frequência;
2. implementar uma **busca semântica por texto** (*zero-shot retrieval*) e analisar oito consultas que variam de genéricas a abstratas.

## Dados

O **ADS-16 Computational Advertising Dataset** foi baixado do Kaggle (`groffo/ads16-dataset`). A varredura recursiva encontrou **2.697 imagens** distribuídas em **260 pastas**. Há dois tipos de pasta: as numeradas de 1 a 20, com cerca de 15 anúncios cada, e pastas identificadas por código de usuário com sufixo `IM-POS`/`IM-NEG`, com 10 a 20 imagens. O notebook trata o nome da pasta como "categoria"; essa informação serve apenas para exibição, pois a tarefa não usa rótulos.

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

O notebook fixou o limiar em **0,28**. A curva mostra que as similaridades se concentram em uma faixa muito estreita (aproximadamente 0,18–0,34): entre 0,20 e 0,25 a média despenca de quase 8 para menos de 0,5 conceito por imagem.

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

- **Consultas concretas de objeto funcionam bem.** Em *a car*, as cinco imagens são anúncios automotivos (Fiat 500, BMW, acessórios Raven, peças Renault). Uma delas, porém, é um anúncio **apenas de texto** ("Cheapest Car Parts Online"), sem nenhum carro: o CLIP também "lê" o texto da imagem.
- **Consultas específicas funcionam parcialmente.** *A luxury car in an urban environment* recupera um Maserati e um BMW, mas o fundo urbano raramente está presente; a 5ª imagem é o interior de um carro em um túnel.
- **Consultas abstratas são fracas.** *A premium lifestyle* retorna um anúncio do aplicativo de relacionamento Badoo, um da American Apparel, uma campanha da PETA, um anúncio de app de encontros e um do McDonald's. Apenas parte desses resultados pode ser associada a "estilo de vida premium".

## Discussão

- **O limiar 0,28 é restritivo demais para contar conceitos.** Com ele, 98,1% das imagens não têm nenhum conceito, e o conceito mais frequente aparece em apenas 18 de 2.697 anúncios. O ranking passa a depender de poucas imagens e é instável: mudar a ordem de alguns poucos anúncios altera as posições. Note que a ordem por **similaridade média** é diferente (*a person* teria a maior média, 0,2076). Alternativas mais robustas seriam: limiar por conceito (por exemplo, um percentil da distribuição de cada conceito), atribuição *top-k* por imagem ou comparação relativa entre conceitos (softmax sobre os 20 *prompts*).
- **Fenômeno de *hubness*.** Um pequeno conjunto de imagens domina os resultados de consultas sem relação entre si. O anúncio do **Fiat 500** aparece no top-5 de **6 das 8** consultas (inclusive *a person* e *a person using a smartphone*), o do **Badoo** em 5 das 8, e o da **Kaspersky** em 4. Esses "hubs" são anúncios prototípicos (produto, pessoa e texto em destaque) que ficam próximos de muitos textos no espaço do CLIP. Isso explica por que a busca parece razoável em consultas genéricas e falha em consultas que exigem detalhes.
- **Efeito do template de prompt.** A função de busca aplica o mesmo template dos conceitos às consultas. Assim, *a product advertisement* se torna "an advertisement featuring a product advertisement", e todas as consultas são puxadas na direção de "anúncio típico", o que reforça o *hubness*. Comparar a busca com e sem o template seria um experimento simples e informativo.
- **Leitura de texto (OCR implícito).** O CLIP associa palavras escritas na imagem ao conceito, como em "Car Parts", "Smartphone offers" e "Freshly Brewed Coffee" (este último aparece em *a cold drink being advertised*). Isso é útil para anúncios, mas também é uma vulnerabilidade conhecida a "ataques tipográficos".
- **Duplicatas no corpus.** Imagens idênticas aparecem em posições consecutivas (o Maserati em *luxury car* e as latas de refrigerante em *cold drink*). Elas ocupam vagas do top-5 e inflam contagens; uma deduplicação por hash antes da indexação resolveria.
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

**Images Dataset** (Kaggle, `pavansanagapati/images-dataset`): 3.606 arquivos de imagem (`.jpg`, `.png`, `.bmp`) em 7 classes, sem nenhum arquivo corrompido.

| Classe | Total | Treino | Validação |
|---|---:|---:|---:|
| bike | 730 | 584 | 146 |
| cars | 840 | 672 | 168 |
| cats | 404 | 323 | 81 |
| dogs | 404 | 323 | 81 |
| flowers | 420 | 336 | 84 |
| horses | 404 | 323 | 81 |
| human | 404 | 323 | 81 |
| **Total** | **3.606** | **2.884** | **722** |

![Distribuição das imagens por classe.](atividade-3-cnn-pre-treinada/figures/distribuicao-classes.png){width=75%}

A divisão é estratificada (80/20, semente 42).

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
| 1 | 0,4452 | 0,9286 | 0,1321 | 0,9889 |
| 5 | 0,0224 | 0,9976 | 0,0274 | 0,9972 |
| 10 | 0,0099 | 0,9990 | 0,0134 | **1,0000** |

![Curvas de loss e acurácia por época.](atividade-3-cnn-pre-treinada/figures/curvas-treinamento.png){width=90%}

Na época 10, o modelo classificou corretamente **as 722 imagens de validação**: acurácia, precisão, recall e F1 iguais a 1,00 em todas as classes, sem nenhum erro a exibir.

![Matriz de confusão na validação.](atividade-3-cnn-pre-treinada/figures/matriz-confusao.png){width=60%}

## Análise crítica: vazamento por imagens duplicadas

Um resultado perfeito exige verificação. Os caminhos listados no notebook já indicavam duas estruturas paralelas (por exemplo, `data/horses/...` e `data/data/bike/...`). Para checar, o dataset foi baixado novamente e a divisão do notebook foi reproduzida com as mesmas sementes, calculando o hash MD5 de cada arquivo:

| Verificação | Resultado |
|----------------------------------------|---------------------:|
| Pastas `data/<classe>` e `data/data/<classe>` | mesmas contagens por classe (ex.: 420 + 420 em *cars*) |
| Conteúdos distintos (MD5) entre os 3.606 arquivos | **1.764** |
| Arquivos que têm ao menos uma cópia idêntica | **3.606 (100%)** |
| Imagens de validação com cópia exata no treino | **568 de 722 (78,7%)** |
| Imagens de validação sem cópia no treino (MD5 nem hash perceptual) | 154 |

Ou seja, o dataset vem **inteiramente duplicado**. Como a divisão foi feita por arquivo, quase 79% da validação consiste em imagens que o classificador viu no treino. A acurácia de 100% sobre 722 imagens **não é uma estimativa válida de generalização**.

Ainda assim, há uma evidência útil nas saídas: como o modelo não errou nenhuma imagem de validação, também acertou as **154 imagens sem cópia no treino**. Para 154 acertos em 154, o intervalo de confiança de Wilson (95%) para a acurácia é aproximadamente **[0,976; 1,000]**. A tarefa é de fato fácil para features do ImageNet: as sete classes (carros, bicicletas, cães, gatos, cavalos, flores e pessoas) correspondem a conceitos amplamente representados no pré-treinamento e tendem a ser linearmente separáveis no espaço de 2.048 dimensões da ResNet-50. As curvas, com acurácia de validação de 0,989 já na primeira época, são coerentes com essa leitura.

**Correção recomendada:** remover duplicatas por MD5 **antes** da divisão (ou dividir por grupo de conteúdo), reexecutar o experimento e reportar a acurácia sobre as ~1.764 imagens distintas.

## Propostas de melhoria

O notebook discute técnicas para experimentos posteriores, mantendo o baseline intocado:

| Técnica | Benefício esperado | Risco |
|------------------|------------------|------------------|
| Augmentation geométrica (flip horizontal, rotações e crops pequenos) | Variedade de enquadramento; menos sobreajuste | Alterar a semântica de objetos orientados |
| Augmentation de cor (`ColorJitter` conservador) | Robustez a iluminação e câmera | Apagar pistas de cor relevantes |
| Variação de escala (`RandomResizedCrop` moderado) | Robustez a distância e enquadramento | Cortar o objeto |
| Normalização ImageNet | Compatibilidade com o pré-treinamento | Já utilizada; não é augmentation |

Diante do vazamento identificado, a prioridade é outra: **deduplicar e reavaliar**. Com um baseline já próximo de 100%, ganhos de augmentation só serão mensuráveis com um conjunto de avaliação mais difícil, por exemplo com imagens de outra fonte.

## Limitações

1. Vazamento por duplicatas, descrito acima.
2. Seleção do checkpoint pela acurácia na mesma validação usada para reportar o resultado.
3. Apenas acurácia e matriz de confusão; com classes desbalanceadas (*cars* tem o dobro de *cats*), métricas por classe e macro F1 deveriam ser padrão (aqui todas valem 1,0).
4. Uma única semente e nenhum conjunto de teste independente.


\newpage


<!-- Source: atividade-4-estudos-de-caso/RELATORIO.md -->

# Atividade 4 — Estudos de Caso

(rascunho)


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
- ROFFO, G.; VINCIARELLI, A. Personality in Computational Advertising: A Benchmark. *International Workshop on Emotion and Sentiment in Social and Expressive Media (ESSEM)*, 2016.
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
