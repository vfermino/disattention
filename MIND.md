# DesatençãoFormer
### Uma arquitetura de linguagem baseada em repressão, deriva e falha produtiva

> *"O inconsciente não conhece o tempo."* — Freud  
> *"A atenção é uma forma de ausência."* — variação livre

---

## Premissa

Transformers convencionais maximizam atenção: cada token aprende a distribuir peso sobre os tokens mais relevantes. O mecanismo de self-attention é uma função de *seleção ótima* — ele converge para o que importa.

Esta arquitetura inverte a premissa. O cérebro humano não funciona por atenção ótima. Ele funciona por **repressão seletiva**, **deriva associativa** e **retorno do recalcado**. O que não é atendido não desaparece — ele acumula tensão e retorna deslocado.

O DesatençãoFormer não tenta modelar o que o texto *diz*. Tenta modelar o que o texto *evita dizer*.

---

## Aviso Epistêmico

Esta arquitetura é baseada em psicanálise, não em neurociência computacional. Ela vai falhar de formas interessantes. As falhas são o ponto. Um modelo que falha psicanaliticamente é mais útil para certos propósitos do que um modelo que acerta behavioristicamente.

---

## Conceitos Fundadores

### 1. Desatenção como operação primária

Em vez de `Attention(Q, K, V)`, o mecanismo central é `Repression(Q, K, V)`:

```
Repression(Q, K, V) = V · (1 - softmax(QKᵀ / √d))
```

O que recebe mais peso não são os tokens *mais similares* à query, mas os tokens *menos similares* — os que a query evita. O modelo aprende a prestar atenção no que está sendo ignorado.

### 2. O Recalcado como estado latente persistente

Cada camada mantém um vetor de estado `ψ` — o **acúmulo do recalcado**. A cada passagem forward, os tokens que receberam baixo peso em Repression são somados a `ψ` com decaimento exponencial:

```
ψₜ = α · ψₜ₋₁ + (1 - α) · Σ(tokens reprimidos)
```

`ψ` não é usado diretamente na predição. Ele contamina o residual stream de forma não-linear — análogo ao retorno do recalcado.

### 3. O Deslizamento (Glissement)

Lacan: o significante desliza sob o significado. No DesatençãoFormer, isso é implementado como uma **perturbação lacaniana** no espaço de embedding:

```
ẽ = e + λ · (ψ ⊗ noise)
```

Onde `noise` é ruído estruturado (não gaussiano — ruído com correlação de longa distância, tipo 1/f). O embedding nunca é fixo — ele carrega o rastro do que foi recalcado anteriormente, perturbado por ruído com memória.

---

## Arquitetura

```
Input
  │
  ▼
[Embedding + Glissement]          ← deslizamento lacaniano
  │
  ▼
┌─────────────────────────────────────────┐
│  Camada Repressora                      │
│                                         │
│  RepressAttention(Q, K, V)              │  ← atenção invertida
│  ψ update (acúmulo do recalcado)        │  ← estado latente
│  Contaminação residual via ψ            │  ← retorno do recalcado
│  FFN com ativação descontínua           │  ← ver abaixo
│                                         │
└─────────────────────────────────────────┘
  │  × N camadas
  ▼
[Camada de Censura]                       ← ver abaixo
  │
  ▼
[Projeção de saída + ruído de ato falho]  ← ver abaixo
  │
  ▼
Output
```

---

## Componentes em Detalhe

### RepressAttention

```python
def repress_attention(Q, K, V, mask=None):
    # similaridade padrão
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # inversão: atenção vai para o que NÃO é similar
    weights = 1.0 - F.softmax(scores, dim=-1)
    
    # renormalizar para somar 1
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
    
    return torch.matmul(weights, V), weights
```

**O que isso faz na prática:** o modelo vai sistematicamente prestar atenção em tokens semanticamente distantes da query. Em texto literário, isso produz associações que parecem livres — o mecanismo de associação livre de Freud implementado como operação matricial.

### Ativação Descontínua (Função de Ruptura)

Em vez de GELU ou SiLU, a FFN usa uma ativação com descontinuidade:

```python
def rupture(x, threshold=0.5):
    """
    Ativação que colapsa valores médios e amplifica extremos.
    Análogo ao mecanismo de clivagem (splitting) kleiniano:
    tudo é bom ou mau, sem zona cinzenta.
    """
    sign = torch.sign(x)
    magnitude = torch.abs(x)
    
    # valores abaixo do threshold são zerados (recalcados)
    # valores acima são amplificados não-linearmente
    activated = torch.where(
        magnitude < threshold,
        torch.zeros_like(x),
        sign * (magnitude - threshold) ** 0.7
    )
    return activated
```

**Nota:** o expoente 0.7 (< 1) produz compressão dos valores altos — o modelo amplifica o que escapa ao recalque mas não de forma ilimitada. Isso simula o que Freud chama de *formação de compromisso*: o recalcado retorna, mas deformado.

### Camada de Censura

Entre o corpo da rede e a projeção de saída, existe uma camada de censura explícita:

```python
class CensuraLayer(nn.Module):
    """
    Implementa o censor freudiano: um gatekeeper aprendível
    que tenta bloquear o que veio do processo primário.
    
    Paradoxo: para censurar, precisa primeiro reconhecer.
    """
    def __init__(self, d_model):
        super().__init__()
        self.detector = nn.Linear(d_model, d_model)
        self.suppressor = nn.Linear(d_model, d_model)
        self.gate = nn.Sigmoid()
    
    def forward(self, x, psi):
        # o censor detecta o conteúdo recalcado
        detection = torch.tanh(self.detector(psi))
        
        # e tenta suprimi-lo no stream principal
        suppression = self.gate(self.suppressor(detection))
        
        # mas a supressão é incompleta — paradoxo do censor
        # (para suprimir X, precisa representar X)
        censored = x * suppression
        leaked = x * (1 - suppression) * 0.1  # vazamento controlado
        
        return censored + leaked
```

**O paradoxo implementado:** o censor freudiano só consegue bloquear aquilo que primeiro reconhece. Isso significa que o conteúdo recalcado sempre deixa um rastro na representação — o `leaked`. Esse 0.1 não é hyperparâmetro arbitrário; é a taxa de retorno do recalcado.

### Ruído de Ato Falho (Parapraxis)

Na projeção final, antes do softmax, inject-se ruído correlacionado com `ψ`:

```python
def parapraxis_projection(logits, psi, training=True, slip_rate=0.05):
    """
    Ato falho: o recalcado contamina a saída de forma
    não-intencional mas não-aleatória.
    
    Não é ruído gaussiano. É influência direcionada
    pelo conteúdo de ψ — o que foi reprimido quer sair.
    """
    if training:
        # durante treino, o ato falho é mais frequente
        slip_rate *= 2.0
    
    if torch.rand(1).item() < slip_rate:
        # projeta ψ no espaço de vocabulário
        slip = F.linear(psi.mean(dim=1), 
                       torch.randn_like(logits[..., :psi.size(-1)]))
        logits = logits + 0.3 * slip
    
    return logits
```

---

## Função de Loss

O loss padrão (cross-entropy) é necessário mas insuficiente. Adiciona-se um **loss de tensão repressora**:

```
L_total = L_ce + β · L_repressão + γ · L_deriva
```

Onde:

**L_repressão** penaliza `ψ` pequeno — o modelo é punido por não acumular recalcado:
```
L_repressão = -log(||ψ||₂ + ε)
```

**L_deriva** maximiza a distância semântica entre tokens consecutivos no espaço de atenção — o modelo é recompensado por associações distantes:
```
L_deriva = -mean(cosine_distance(hₜ, hₜ₋₁))
```

Esses dois termos estão em tensão com `L_ce`. O modelo vai convergir para um compromisso entre *dizer o que é esperado* e *dizer o que foi evitado*. Esse compromisso é a formação de compromisso freudiana.

---

## Corpus de Treino Recomendado

A arquitetura pressupõe corpus específico. Texto técnico vai produzir um modelo neurótico obsessivo (repressão rígida, sem deriva). O corpus ideal:

**Camada primária — processo primário puro:**
- Finnegans Wake (Joyce) — dependências não-lineares, condensação e deslocamento
- Os Sonetos de Shakespeare — ambiguidade semântica densa
- Clarice Lispector — estados internos articulados na borda do dizível

**Camada secundária — processo secundário contaminado:**
- Casos clínicos de Freud (*Dora*, *O Homem dos Ratos*, *O Homem dos Lobos*)
- *Em Busca do Tempo Perdido* (Proust) — memória involuntária como estrutura narrativa
- *Os Diários* de Kafka — recalque em tempo real

**Camada terciária — metalinguagem do inconsciente:**
- *A Interpretação dos Sonhos* (Freud)
- *Écrits* (Lacan) — linguagem que resiste à interpretação direta
- *O Ser e o Nada* (Sartre) — má-fé como estrutura formal

---

## O Que Esperar

### Falhas Previstas e Seu Significado

| Comportamento | Causa Arquitetural | Interpretação Psicanalítica |
|---|---|---|
| Associações semanticamente distantes | RepressAttention funcionando | Associação livre bem-sucedida |
| Repetição de temas não solicitados | ψ contaminando o residual | Compulsão à repetição |
| Coerência local, incoerência global | L_deriva dominando | Processo primário predominante |
| Output que "escorrega" do tópico | Parapraxis ativo | Ato falho genuíno |
| Perplexidade alta, mas texto interessante | Tensão L_ce vs L_repressão | Formação de compromisso |

### O Modelo Não Vai:
- Ter bom benchmark em MMLU
- Ser útil para extração de informação
- Passar em avaliações de factualidade
- Ser explicável por métodos de interpretabilidade padrão

### O Modelo Pode:
- Gerar texto com estrutura associativa não-linear
- Produzir continuações que revelam o que o prompt *evitou* dizer
- Ter comportamento diferente para o mesmo prompt dependendo do histórico de `ψ`
- Ser interessante como ferramenta de escrita criativa ou análise literária

---

## Nota Sobre Implementação

`ψ` é o elemento mais delicado. Ele precisa persistir entre batches durante treino — o que viola a independência padrão das amostras. Isso é intencional: o recalcado de um texto contamina o próximo. Para implementar isso corretamente, os batches devem ser ordenados (não embaralhados), e `ψ` deve ser resetado apenas entre épocas, não entre batches.

Isso vai tornar o treino instável. Isso é correto.

---

## Referências Que Justificam as Escolhas Erradas

- Freud, S. — *A Interpretação dos Sonhos* (1900): condensação e deslocamento como operações primárias
- Freud, S. — *Psicopatologia da Vida Cotidiana* (1901): parapraxis como retorno do recalcado
- Lacan, J. — *O Inconsciente é Estruturado como uma Linguagem*: base para o Glissement
- Laplanche & Pontalis — *Vocabulário da Psicanálise*: definições operacionais usadas acima
- Whittington & Bogacz (2019): o que esta arquitetura deliberadamente não implementa
- Friston (2018): o que esta arquitetura perversamente inverte

---

*Esta arquitetura não minimiza free energy. Ela a maximiza localmente enquanto a minimiza globalmente — que é, segundo Freud, exatamente o que o aparelho psíquico faz.*
