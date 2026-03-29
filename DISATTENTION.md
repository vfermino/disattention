# DesatençãoFormer
**Especificação Técnica — PyTorch**  
Arquitetura de linguagem baseada em arquétipos junguianos como curvatura do espaço semântico.

---

## Dependências

```
torch>=2.2.0
numpy>=1.26.0
transformers>=4.40.0
sentence-transformers>=2.7.0
datasets>=2.19.0
tiktoken>=0.7.0
einops>=0.8.0
wandb>=0.17.0
```

---

## Visão Geral

```
Input IDs
    │
    ▼
TokenEmbedding + PositionalEmbedding
    │
    ▼
ArchetypalProjection        # curva o espaço de embedding via tensores C_k
    │
    ▼
[ DesatencaoBlock × N ]
│   ├── SymbolAttention     # self-attention com métrica arquetípica
│   ├── TensionFFN          # FFN com dois polos opostos
│   └── IndividuationNorm   # LayerNorm com decaimento por profundidade
    │
    ▼
TranscendentFunction        # síntese pré-output modulada por arquétipos
    │
    ▼
Linear(d_model, vocab_size)
    │
    ▼
Logits
```

Parâmetros default para RTX 4000 Ada (20 GB):

| Hyperparâmetro | Valor |
|---|---|
| `d_model` | 512 |
| `n_heads` | 8 |
| `n_layers` | 12 |
| `d_ff` | 2048 |
| `max_seq_len` | 1024 |
| `n_archetypes` | 16 |
| `vocab_size` | 50257 |
| Parâmetros totais | ~120M |

---

## 1. Tensores Arquetípicos

Os tensores arquetípicos são **pré-computados** antes do treino a partir do corpus mítico. Não têm gradiente. São a estrutura a priori que precede o aprendizado — o inconsciente coletivo que precede o indivíduo.

### Definição Formal

Dado corpus de passagens míticas `P_k = {p_1, ..., p_m}` para o arquétipo `k` e encoder base `φ: texto → ℝᵈ`:

```
E_k  = [φ(p_1), ..., φ(p_m)]  ∈ ℝᵐˣᵈ
Ē_k  = E_k - mean(E_k, dim=0)
C_k  = (Ē_kᵀ Ē_k) / (m - 1)    # covariância das manifestações
C_k  = C_k / trace(C_k)          # normalização por traço
```

`C_k` captura a *forma* do campo — a variância interna das manifestações do arquétipo — não o centróide. Dois arquétipos com centróides similares têm tensores distintos se sua variância interna for diferente.

### `build_archetypes.py`

```python
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# ARQUÉTIPOS JUNGUIANOS — conjunto completo
#
# Fontes primárias:
#   Jung, C.G. — Os Arquétipos e o Inconsciente Coletivo (OC vol. 9/1)
#   Jung, C.G. — Aion (OC vol. 9/2)
#   Jung, C.G. — Psicologia e Alquimia (OC vol. 12)
#   Jung, C.G. — Os Arquétipos do Inconsciente Coletivo (1936)
#   Von Franz, M-L. — O Feminino nos Contos de Fada
#   Hillman, J. — Re-Visioning Psychology
#
# Cada arquétipo contém passagens de textos míticos, literários e do
# próprio corpus joyceano que o instanciam. O tensor C_k é construído
# a partir da covariância dessas manifestações — captura a FORMA do
# campo arquetípico, não seu conteúdo médio.
#
# Polaridades: cada arquétipo tem polo positivo e negativo.
# As passagens devem cobrir AMBOS os polos — a tensão entre eles
# é o que torna o tensor útil.
# ─────────────────────────────────────────────────────────────────────────────

ARCHETYPES = {

    # ── 1. SELF ───────────────────────────────────────────────────────────────
    # O arquétipo da totalidade psíquica. Distinto do ego: precede e excede
    # o ego. Centro e circunferência simultaneamente. Símbolo: mandala,
    # círculo, quaternidade, pedra filosofal, Cristo como antropos.
    # Em Joyce: o ciclo do Finnegans Wake que começa onde termina.
    # Polaridade: integração ↔ inflação (ego confundido com Self = psicose).
    "self": [
        "The mandala is the self made visible — center and circumference at once",
        "That art thou: the drop knowing itself as ocean",
        "All rivers run into the sea; yet the sea is not full; unto the place from whence the rivers come thither they return",
        "The stone which the builders rejected has become the cornerstone",
        "What was before Abraham was — I am",
        "The end of all our exploring will be to arrive where we started and know the place for the first time",
        "riverrun, past Eve and Adam's, from swerve of shore to bend of bay, brings us by a commodius vicus of recirculation back to Howth Castle and Environs",
        "The Self is not only the centre but also the whole circumference which embraces both conscious and unconscious",
        "As above, so below; as within, so without — the whole in every part",
        "I am Alpha and Omega, the beginning and the end, the first and the last",
    ],

    # ── 2. SOMBRA ─────────────────────────────────────────────────────────────
    # O que o ego rejeita e projeta no outro. Não é o mal em si —
    # é o não-integrado. Pode ser positivo (potencial recusado) ou
    # negativo (impulso negado). A Sombra é sempre da mesma natureza do ego.
    # Em Joyce: HCE e o crime não-nomeado de Phoenix Park.
    # Símbolo: o duplo, o irmão inimigo, o perseguidor nos sonhos.
    # Polaridade: negação ↔ integração; projeção ↔ reconhecimento.
    "sombra": [
        "I am the spirit that always denies, and rightly so, for all that exists deserves to perish",
        "Better to reign in Hell than serve in Heaven",
        "Here Comes Everybody — the man of the crowd, criminal and saint indistinguishable",
        "Set dismembered Osiris and scattered the pieces across the length of Egypt",
        "What the hand dare seize the fire? Did he who made the Lamb make thee?",
        "The thing I feared has come upon me; what I dreaded has happened to me",
        "In the cave of the heart the shadow waits, wearing your own face",
        "He that is without sin among you, let him cast the first stone",
        "Mr Hyde emerged from Dr Jekyll and could not be put back",
        "The unconscious is not a den of iniquity; it is simply the unknown",
    ],

    # ── 3. ANIMA ──────────────────────────────────────────────────────────────
    # O feminino na psique masculina. Não é a mulher real — é a imagem
    # interna do feminino que projeta e distorce a percepção.
    # Quatro níveis: Eva (biológico) → Helena (estético) →
    # Maria (espiritual) → Sophia (sapiencial).
    # Em Joyce: Molly Bloom ("Yes"), ALP, a garota na praia em Portrait.
    # Polaridade: musa ↔ possessão; inspiração ↔ armadilha sentimental.
    "anima": [
        "Yes I said yes I will Yes — the affirmation beyond all argument",
        "Penelope weaving and unweaving, faithful in betrayal, betraying in fidelity",
        "Beatrice appeared to me clothed in the colour of living flame, ancient and ageless",
        "She is the Muse who speaks through the poet and then abandons him to silence",
        "Anna Livia Plurabelle — her waters are her words and her words are her waters",
        "A girl stood before him in midstream, alone and still, gazing out to sea",
        "The eternal feminine draws us onward and upward — das Ewig-Weibliche",
        "She who must be obeyed — the goddess who destroys those who cannot bear her presence",
        "I have been woman and man, young and old, plant and bird and silent fish",
        "The Anima is the archetype of life itself — irrational, capricious, possessive, delightful",
    ],

    # ── 4. ANIMUS ─────────────────────────────────────────────────────────────
    # O masculino na psique feminina. Distinto da Anima — opera de modo
    # diferente: onde a Anima é humor e irracionalidade, o Animus é
    # opinião e logos. Quatro níveis: poder físico → iniciativa →
    # logos (palavra) → significado espiritual.
    # Em Joyce: Stephen Dedalus como Animus de uma época.
    # Polaridade: clareza ↔ dogmatismo; logos ↔ sentença inapelável.
    "animus": [
        "Non serviam — I will not serve that in which I no longer believe",
        "The word is the deed; in the beginning was the logos and the logos was action",
        "He spoke with authority, and not as the scribes — the voice that does not negotiate",
        "I go to encounter for the millionth time the reality of experience and to forge in the smithy of my soul the uncreated conscience of my race",
        "The masculine spirit in woman appears as an opinion disconnected from relation",
        "Athena springs fully armed from the head of Zeus — thought that precedes feeling",
        "The animus says: it is so — and closes the question before it opens",
        "He who argues with a woman is arguing with her animus, not with her",
        "The sword is the animus — it cuts, divides, names, and kills what it names",
        "In the beginning was the Word — and the Word was already a judgment",
    ],

    # ── 5. GRANDE MÃE ────────────────────────────────────────────────────────
    # O arquétipo do feminino primordial, anterior à individuação.
    # Distinto da Anima: a Grande Mãe é coletiva e transpessoal.
    # Inclui os aspectos nutridores E os devoradores — ambos são a mesma figura.
    # Em Joyce: ALP como rio, como terra, como origem e destino.
    # Símbolo: terra, caverna, mar, lua, caldeirão, vaso, noite.
    # Polaridade: nutrição ↔ devoração; fertilidade ↔ morte.
    "grande_mae": [
        "Earth mother searching for her lost daughter through barren winter fields — Demeter's grief that made the world sterile",
        "Goddess gathering the scattered pieces of her beloved across Egypt to restore him to life — Isis reassembling Osiris",
        "Dark goddess dancing on corpses, skull necklace rattling, destroyer and creatrix simultaneously — Kali",
        "Mother holding her dead son, pietà of absolute sorrow and absolute grace inseparable",
        "She who kills her own children rather than yield them to another — Medea, love as annihilation",
        "the earth was without form and void, and darkness was upon the face of the deep, and the spirit moved upon the waters",
        "riverrun, past Eve and Adam's — ALP as the Liffey, the maternal river that is also the letter and the body",
        "The Great Mother is the earth that receives the dead and the womb that gives birth — the same darkness",
        "Hecate at the crossroads, three-faced, the moon in all its phases — maiden, mother, crone",
        "She swallowed her children whole so they could not surpass her — Kronos in feminine form",
    ],

    # ── 6. GRANDE PAI ────────────────────────────────────────────────────────
    # O arquétipo do masculino primordial e transpessoal. Ordem, lei,
    # espírito, logos coletivo. Distinto do Velho Sábio — o Grande Pai
    # é poder e estrutura; o Velho Sábio é conhecimento e iniciação.
    # Símbolo: céu, trovão, montanha, rei, tábua da lei, espada.
    # Polaridade: proteção/ordenação ↔ tirania/devoração do filho.
    "grande_pai": [
        "I am the Lord your God — the voice from the whirlwind that does not explain itself",
        "Zeus hurling the thunderbolt from Olympus — power that needs no justification",
        "The father who demands sacrifice: Abraham raising the knife over his son",
        "Kronos devouring his children to prevent being surpassed — the father who devours the future",
        "The law is the law — Creon who will not yield even when the law kills",
        "He descended to his fathers; in the city of David they buried him",
        "HCE — Here Comes Everybody — the fallen patriarch whose guilt structures the dream",
        "The king's two bodies: the mortal man and the immortal office that outlasts him",
        "Wotan wanders the world he can no longer govern, trading an eye for wisdom he cannot use",
        "The father's word is law — and the word always arrives too late or too early",
    ],

    # ── 7. HERÓI ─────────────────────────────────────────────────────────────
    # O ego em desenvolvimento confrontando o inconsciente.
    # Não é a virtude — é a transformação através do perigo.
    # A jornada do herói: separação → iniciação → retorno.
    # Em Joyce: Bloom como herói cotidiano (Ulisses sem Troia).
    # Símbolo: espada, escudo, dragão a ser morto, descida ao submundo.
    # Polaridade: coragem/transformação ↔ hubris/destruição.
    "heroi": [
        "No man am I called, and No man is what my mother and father call me — Odisseu na caverna do ciclope",
        "He who saw the deep, the country's foundation — Gilgamesh after Enkidu's death",
        "Stately, plump Buck Mulligan — but the true hero descends: Bloom walking through Dublin's underworld",
        "His soul swooned slowly as he heard the snow falling faintly through the universe — the moment of dissolution before return",
        "Here am I — the one who descends and does not return unchanged",
        "The hero's journey: road of trials, innermost cave, ordeal, reward, road back",
        "I have slain the Minotaur in the labyrinth and the thread leads me out — but I forgot to change the sail",
        "Into that darkness peering, long I stood there, wondering, fearing, doubting",
        "He emptied himself, taking the form of a servant — the hero who descends by choice",
        "The dragon hoards the gold: the hero's task is always to take what was locked away",
    ],

    # ── 8. VELHO SÁBIO ───────────────────────────────────────────────────────
    # O espírito como arquétipo — não o poder (Grande Pai) mas o conhecimento.
    # Aparece como guia, iniciador, portador de significado.
    # Distinto do Grande Pai: não comanda — orienta e pergunta.
    # Em Joyce: o Finnegan morto que tudo sabe; Tirésias em Eliot.
    # Símbolo: o ancião, o mago, o eremita, o adivinho cego.
    # Polaridade: sabedoria ↔ feiticeiro enganador (aspecto negativo).
    "velho_sabio": [
        "I am Tiresias, blind and knowing both sexes and all futures, and I have foresuffered all",
        "What are you asking, Oedipus? You do not want to know the answer",
        "Thus Finn MacCool lies sleeping under the hill, and will rise when Ireland needs him",
        "The Fisher King waits in the wasteland for the one who asks the question that heals",
        "In the beginning was the Word — and the Word was already a riddle requiring interpretation",
        "He who knows does not speak; he who speaks does not know — the sage's paradox",
        "Prospero: I have bedimmed the noontide sun and by my art provoked the winds — and then drowned the book",
        "The Merlin who knows Arthur's fate and cannot change it, can only accompany it",
        "Old age should burn and rave at close of day — but the sage knows when to stop burning",
        "Do not go where the path may lead; go instead where there is no path and leave a trail",
    ],

    # ── 9. PERSONA ───────────────────────────────────────────────────────────
    # A máscara social — a face que o ego apresenta ao mundo.
    # Não é falsa por natureza, mas torna-se patológica quando
    # o ego se identifica totalmente com ela.
    # Em Joyce: o Stephen que performa o artista; o Bloom que performa
    # o marido resignado.
    # Símbolo: a máscara, o uniforme, o título, o papel social.
    # Polaridade: adaptação social ↔ perda do Self; conformidade ↔ rigidez.
    "persona": [
        "All the world's a stage, and all the men and women merely players — each with his entrance and his exit",
        "He wore his smile like a mask — beneath it nothing, or everything, or the same smile again",
        "I am large, I contain multitudes — but which multitude is presenting itself now?",
        "The uniform makes the man: remove the rank and what remains is a man afraid",
        "Stephen Dedalus, teacher, artist, Irishman — each name a costume, none of them the wearer",
        "The persona is the individual's system of adaptation to the world — and his prison",
        "He played the part so long he forgot there was a player",
        "Smile and smile and be a villain — the persona as weapon",
        "Take off the mask: another mask beneath it. Take off that one too",
        "I have met them at close of day coming with vivid faces from counter or desk among grey eighteenth-century houses",
    ],

    # ── 10. PUER AETERNUS ────────────────────────────────────────────────────
    # O eterno jovem — arquétipo do filho divino, da possibilidade
    # não-realizada, da recusa em encarnar e envelhecer.
    # Recusa a limitação, vive em possibilidade pura.
    # Em Joyce: Stephen Dedalus explicitamente (Dédalo o artífice, Ícaro o filho).
    # Símbolo: Ícaro, Narciso, Peter Pan, o príncipe que não assume o trono.
    # Polaridade: criatividade/liberdade ↔ recusa de comprometimento/morte prematura.
    "puer_aeternus": [
        "I will not serve that in which I no longer believe — the Puer's manifesto",
        "Fabulous artificer — the hawklike man — the artist soaring above the world he refuses to inhabit",
        "He had daedalian wings and flew too near the sun — Icarus as the type of all pure aspiration",
        "I shall not grow old — Peter Pan's refusal which is also his death-in-life",
        "The boy who would not grow up lives in a country where no one ages and nothing has consequences",
        "Narcissus at the pool: in love with his own reflection, which is also his drowning",
        "Stephen looked at the tundish and thought: I am not what they made me. I will make myself",
        "The eternal youth has wings but no roots — he can fly anywhere and land nowhere",
        "Dionysus torn apart by the Titans and reassembled — the god who dies young always",
        "His gifts were extraordinary and his life was brief and that was the same thing",
    ],

    # ── 11. SENEX ────────────────────────────────────────────────────────────
    # O polo velho/saturno — oposto ao Puer. Onde o Puer é libertade
    # e recusa, o Senex é ordem, peso, lei cronológica, melancolia.
    # Não é o Velho Sábio — o Senex é rígido e saturnino, não sábio.
    # Em Joyce: o Dublin paralisado de Dubliners; o padre Simon Dedalus.
    # Símbolo: Saturno devorando o filho, o rei que não abdica, a lei que não se dobra.
    # Polaridade: estrutura/maturidade ↔ rigidez/paralisia/devoração do novo.
    "senex": [
        "The old men shall dream dreams — but first they must stop dreaming that they are still young",
        "Saturnus eating his children: the old order consuming what would succeed it",
        "He had lived too long in one city; the streets had grown into him like roots",
        "The law is the law and I am its servant — Creon, the Senex who cannot yield",
        "Dublin. I always knew I would leave and I never left — the Senex city",
        "The paralysis of the living dead: not Lazarus raised but Lazarus who declined the resurrection",
        "He had his convictions and he had his position and between them there was no space for the world to enter",
        "Old Goriot giving everything to daughters who ceased to see him — the Senex as tragic figure",
        "The weight of years is not wisdom; it is sometimes only weight",
        "He was not old; he had simply stopped moving — and stillness looked like age",
    ],

    # ── 12. TRICKSTER ────────────────────────────────────────────────────────
    # O trapaceiro — transgride fronteiras, viola tabus, provoca
    # transformação através do caos e do riso. Nem herói nem vilão.
    # Figura liminar: vive nas bordas dos sistemas.
    # Em Joyce: Buck Mulligan; o próprio Joyce como autor.
    # Símbolo: Hermes, Loki, Exu, Coyote, o Bufão do rei.
    # Polaridade: libertação/criatividade ↔ destruição/irresponsabilidade.
    "trickster": [
        "Buck Mulligan: the mocker, the usurper, the one who laughs at what the other holds sacred",
        "Loki who cut off Sif's hair for a joke — and then had to save the world from the consequences",
        "Hermes: messenger, thief, guide of souls, inventor of the lyre stolen from Apollo",
        "Coyote stole fire and got burned and stole it again — the trickster cannot stop",
        "The Fool in the Tarot: zero, the unnumbered card, the one who walks off cliffs",
        "Exu at the crossroads, where all paths meet and all bargains are made — and broken",
        "He who asks the wrong question at the right moment — and everything changes",
        "The jester speaks truth to the king because no one takes him seriously enough to punish him",
        "Mercurius: the coincidence of opposites, the god who is simultaneously divine and demonic",
        "The trickster breaks the rule not to destroy the system but to reveal that the system was never the point",
    ],

    # ── 13. KORE / DONZELA ───────────────────────────────────────────────────
    # O feminino jovem — não a mãe, não a sábia, mas a potencialidade
    # feminina não-realizada. A donzela que descende, que é raptada,
    # que atravessa a iniciação.
    # Distinto da Anima: Kore é transpessoal e coletiva.
    # Em Joyce: a garota-pássaro na praia em Portrait; Gerty MacDowell.
    # Símbolo: Perséfone, Cinderela, a torre, a flor, a virgem.
    # Polaridade: inocência/abertura ↔ raptura/descida/transformação forçada.
    "kore": [
        "Persephone gathering flowers in the meadow when the earth opened and swallowed her — the descent that was also a marriage",
        "A girl stood before him in midstream, gazing out to sea — the epiphany on the strand",
        "She was like a wild bird, fearless and still — the image of what art might become",
        "Gerty MacDowell on the rocks, arranging herself to be seen — the Kore performing Kore",
        "Sleeping Beauty: the hundred-year sleep between childhood and womanhood",
        "Cinderella among the ashes, waiting for the transformation she does not yet know she carries",
        "The maiden in the tower has hair long enough to climb — but she did not let it down for herself",
        "Proserpina in the underworld, eating the pomegranate seeds — each seed a month of winter",
        "She was fourteen and she stood at the window and everything was still possible",
        "The virgin goddess: not innocent but undefeated, not cold but unclaimed",
    ],

    # ── 14. CRIANÇA DIVINA ───────────────────────────────────────────────────
    # O arquétipo do nascimento, da potencialidade pura, do começo
    # que precede toda determinação. Distinto do Puer — a Criança Divina
    # é anterior à individuação, símbolo do Self emergente.
    # Em Joyce: o próprio livro como criança — Finnegans Wake como
    # nascimento que nunca cessa.
    # Símbolo: o menino Jesus, o bebê Horus, Krishna infante, o Rei Pescador jovem.
    # Polaridade: potência/renovação ↔ vulnerabilidade/abandono/infanticídio.
    "crianca_divina": [
        "And she brought forth her firstborn son and wrapped him in swaddling clothes and laid him in a manger",
        "The infant Horus hidden in the papyrus marsh while Set hunted him — the divine child in danger",
        "The child Krishna stealing butter, dancing on the serpent, holding the world in his mouth",
        "A child is born to us, a son is given — and the government shall be upon his shoulders",
        "Out of the mouth of babes and sucklings thou hast perfected praise",
        "The divine child is abandoned, exposed, threatened — and survives because the world cannot yet bear him",
        "Every child arrives as a stranger from somewhere the adults have forgotten",
        "The boy in the library reading: the world has not yet closed around him",
        "The child who asks why — before learning that why is not always a welcome question",
        "In my beginning is my end; in my end is my beginning — the child as the Self not yet divided",
    ],

    # ── 15. ESPÍRITO ─────────────────────────────────────────────────────────
    # O arquétipo do espírito como tal — não uma figura pessoal mas
    # uma presença que orienta, ilumina ou perturba.
    # Aparece em contos de fada como o ajudante sobrenatural, o animal
    # falante, o anão que sabe o nome verdadeiro.
    # Em Joyce: o fantasma do pai de Hamlet (citado em Ulysses); o
    # Espírito do Finno adormecido.
    # Símbolo: o vento, a chama, a voz no deserto, o daemon socrático.
    # Polaridade: orientação/iluminação ↔ possessão/fanatismo.
    "espirito": [
        "The wind bloweth where it listeth, and thou hearest the sound thereof, but canst not tell whence it cometh",
        "I am thy father's spirit, doomed for a certain term to walk the night — Hamlet's ghost",
        "The still small voice after the earthquake and the fire — not in the spectacle but in the silence",
        "He was filled with the spirit and went into the wilderness — the spirit that drives out, not in",
        "In the fairy tale the old man in the forest knows the path; he is the spirit appearing as helper",
        "Socrates heard his daimon: not a god, not a man, but the voice that said no when no should be said",
        "Pentecost: the spirit as fire distributed — not hierarchy but simultaneous ignition",
        "The muse descends and the poet writes what he did not intend to write",
        "Geist: the spirit of an age that no individual chose but all embody",
        "The spirit appears as what is needed: guide, trickster, gift, demand",
    ],

    # ── 16. QUATERNIDADE E FUNÇÕES PSICOLÓGICAS ──────────────────────────────
    # As quatro funções de Jung (pensamento, sentimento, sensação, intuição)
    # estruturam a psique como quaternidade — o número 4 como símbolo do Self
    # completo. A função inferior é sempre a mais inconsciente e a porta
    # para o inconsciente.
    # Em Joyce: Ulysses dividido em episódios correspondentes a órgãos e artes —
    # uma quaternidade expandida em estrutura.
    # Símbolo: a cruz, os quatro elementos, os quatro evangelistas, a roda.
    # Polaridade: completude ↔ unilateralidade; integração ↔ função inferior possessiva.
    "quaternidade": [
        "Four rivers went out from Eden — and the four is always the completion of the three",
        "Earth, water, fire, air — the four that compose all that decomposes",
        "The four evangelists: the man, the lion, the ox, the eagle — four faces of one message",
        "Thinking, feeling, sensation, intuition: the cross on which the psyche is crucified and integrated",
        "The quaternity is the minimum structure of wholeness — three is always incomplete",
        "Jung's four functions: the superior function we trust, the inferior function that betrays us",
        "The table has four legs — remove one and it is no longer a table but a philosophical problem",
        "North, south, east, west — the four directions that make a centre possible",
        "Ulysses: organ, art, colour, symbol — the book as mandala of the human",
        "The inferior function: what we cannot do well and therefore fear and project and eventually must face",
    ],
}


def build_archetype_tensors(
    archetypes: dict,
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    target_dim: int = 512,
    device: str = "cpu",
) -> dict:
    """
    Constrói tensores de curvatura C_k para cada arquétipo.
    Retorna dict {nome: tensor [target_dim, target_dim]}.
    Os tensores não têm gradiente.
    """
    encoder = SentenceTransformer(encoder_name, device=device)
    encoder_dim = encoder.get_sentence_embedding_dimension()

    tensors = {}
    for name, passages in archetypes.items():
        E = torch.tensor(
            encoder.encode(passages, normalize_embeddings=True),
            dtype=torch.float32,
            device=device,
        )  # [m, encoder_dim]

        E = E - E.mean(dim=0, keepdim=True)
        C_raw = (E.T @ E) / (len(passages) - 1)  # [encoder_dim, encoder_dim]

        # redimensionar para target_dim via projeção aleatória fixa
        # Johnson-Lindenstrauss: preserva distâncias com alta probabilidade
        if encoder_dim != target_dim:
            torch.manual_seed(42)
            proj = torch.randn(encoder_dim, target_dim, device=device)
            proj = proj / proj.norm(dim=0, keepdim=True)
            C_raw = proj.T @ C_raw @ proj  # [target_dim, target_dim]

        C = C_raw / (C_raw.trace().abs() + 1e-9)
        tensors[name] = C.detach()
        print(f"  [{name}] shape={C.shape}, trace={C.trace().item():.4f}")

    return tensors


if __name__ == "__main__":
    print("Construindo tensores arquetípicos...")
    print(f"Total de arquétipos: {len(ARCHETYPES)}")
    tensors = build_archetype_tensors(ARCHETYPES, target_dim=512)
    torch.save(tensors, "archetype_tensors.pt")
    print("Salvo: archetype_tensors.pt")
```

---

## 2. Arquitetura — `model.py`

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ArchetypalProjection(nn.Module):
    """
    Aplica curvatura arquetípica ao espaço de embedding.

    Para cada item do batch, computa pesos w_k = softmax(x_mean @ q_k)
    onde q_k são vetores de query aprendíveis por arquétipo.

    Métrica resultante:  M = I + Σ_k w_k * C_k
    Aplicada via:        x_curved = x @ M
    """

    def __init__(self, d_model: int, archetype_tensors: dict):
        super().__init__()
        self.d_model = d_model
        self.archetype_names = list(archetype_tensors.keys())
        n = len(self.archetype_names)

        for i, name in enumerate(self.archetype_names):
            self.register_buffer(f"C_{i}", archetype_tensors[name].float())

        self.queries = nn.Parameter(torch.randn(n, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        x: [B, S, d_model]
        retorna: x_curved [B, S, d_model], archetype_weights [B, n], M [B, d, d]
        """
        x_mean = x.mean(dim=1)
        w = F.softmax(x_mean @ self.queries.T / math.sqrt(self.d_model), dim=-1)

        I = torch.eye(self.d_model, device=x.device, dtype=x.dtype)
        M = I.unsqueeze(0).expand(x.size(0), -1, -1).clone()

        for i in range(len(self.archetype_names)):
            C  = getattr(self, f"C_{i}")
            wi = w[:, i].view(-1, 1, 1)
            M  = M + wi * C.unsqueeze(0)

        x_curved = torch.bmm(x, M)
        return x_curved, w, M


class SymbolAttention(nn.Module):
    """
    Multi-head self-attention com métrica arquetípica.
    Score(q, k) = (Q M Kᵀ) / sqrt(d_head)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        for w in [self.Wq, self.Wk, self.Wv, self.Wo]:
            nn.init.xavier_uniform_(w.weight)

    def forward(self, x, M, mask=None):
        B, S, D = x.shape
        Q  = self.Wq(x)
        K  = self.Wk(x)
        V  = self.Wv(x)
        QM = torch.bmm(Q, M)

        def split(t):
            return rearrange(t, 'b s (h d) -> b h s d', h=self.n_heads)

        QM, K, V = split(QM), split(K), split(V)
        scores = torch.matmul(QM, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores + mask
        attn = self.drop(F.softmax(scores, dim=-1))
        out  = rearrange(torch.matmul(attn, V), 'b h s d -> b s (h d)')
        return self.Wo(out)


class TensionFFN(nn.Module):
    """
    FFN com dois polos opostos que coexistem sem resolução.
    polo positivo: projeção sobre x
    polo negativo: projeção sobre -x
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.positive = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False), nn.GELU(), nn.Dropout(dropout)
        )
        self.negative = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False), nn.GELU(), nn.Dropout(dropout)
        )
        self.project = nn.Linear(d_ff * 2, d_model, bias=False)
        self.drop    = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.project.weight)

    def forward(self, x):
        pos = self.positive(x)
        neg = self.negative(-x)
        return self.drop(self.project(torch.cat([pos, neg], dim=-1)))

    def tension_loss(self, x):
        pos = self.positive(x)
        neg = self.negative(-x)
        diff = 1.0 - F.cosine_similarity(
            pos.reshape(-1, pos.size(-1)),
            neg.reshape(-1, neg.size(-1)),
        )
        return diff.mean()


class IndividuationNorm(nn.Module):
    """
    LayerNorm com taxa de individuação crescente por profundidade.
    output = (1 - r) * LayerNorm(x) + r * x
    """

    def __init__(self, d_model: int, layer_index: int, total_layers: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.r    = layer_index / max(total_layers - 1, 1)

    def forward(self, x):
        return (1.0 - self.r) * self.norm(x) + self.r * x


class DesatencaoBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, layer_index, total_layers, dropout=0.1):
        super().__init__()
        self.norm1 = IndividuationNorm(d_model, layer_index, total_layers)
        self.norm2 = IndividuationNorm(d_model, layer_index, total_layers)
        self.attn  = SymbolAttention(d_model, n_heads, dropout)
        self.ffn   = TensionFFN(d_model, d_ff, dropout)

    def forward(self, x, M, mask=None):
        x = x + self.attn(self.norm1(x), M, mask)
        x = x + self.ffn(self.norm2(x))
        return x


class TranscendentFunction(nn.Module):
    """
    Síntese dos opostos modulada pelo campo arquetípico ativo.
    gate_k = σ(W_gate_k · x) — um gate por arquétipo
    g      = Σ_k w_k * gate_k
    symbol = g * tanh(W_synth · x) + (1 - g) * x
    """

    def __init__(self, d_model: int, n_archetypes: int):
        super().__init__()
        self.synthesis = nn.Linear(d_model, d_model, bias=False)
        self.gates     = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_archetypes)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, w):
        s = torch.tanh(self.synthesis(x))
        g = torch.zeros_like(x)
        for i, gate_layer in enumerate(self.gates):
            wi = w[:, i].unsqueeze(-1).unsqueeze(-1)
            g  = g + wi * torch.sigmoid(gate_layer(x))
        return self.norm(g * s + (1.0 - g) * x)


class DesatencaoFormer(nn.Module):
    """
    Modelo completo.

    Args
    ----
    vocab_size         : tamanho do vocabulário
    d_model            : dimensão do modelo
    n_heads            : cabeças de atenção
    n_layers           : número de blocos
    d_ff               : dimensão interna da TensionFFN
    max_seq_len        : comprimento máximo de sequência
    archetype_tensors  : dict {nome: tensor [d_model, d_model]}
    dropout            : taxa de dropout
    """

    def __init__(
        self,
        vocab_size, d_model, n_heads, n_layers, d_ff,
        max_seq_len, archetype_tensors, dropout=0.1,
    ):
        super().__init__()
        self.d_model  = d_model
        self.n_layers = n_layers
        n_archetypes  = len(archetype_tensors)

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.drop_emb  = nn.Dropout(dropout)

        self.archetypal_proj = ArchetypalProjection(d_model, archetype_tensors)

        self.blocks = nn.ModuleList([
            DesatencaoBlock(d_model, n_heads, d_ff, i, n_layers, dropout)
            for i in range(n_layers)
        ])

        self.transcendent = TranscendentFunction(d_model, n_archetypes)
        self.lm_head      = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   std=0.02)

    def _causal_mask(self, S, device):
        mask = torch.full((S, S), float('-inf'), device=device)
        return torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids, return_internals=False):
        B, S   = input_ids.shape
        device = input_ids.device

        pos = torch.arange(S, device=device).unsqueeze(0)
        x   = self.drop_emb(self.token_emb(input_ids) + self.pos_emb(pos))

        x, w, M = self.archetypal_proj(x)
        mask     = self._causal_mask(S, device)

        x_prefnn = None
        for i, block in enumerate(self.blocks):
            if i == self.n_layers - 1:
                x_prefnn = block.norm2(x)
            x = block(x, M, mask)

        x      = self.transcendent(x, w)
        logits = self.lm_head(x)

        out = {"logits": logits, "archetype_weights": w}
        if return_internals:
            out["x_prefnn"] = x_prefnn
            out["M"]        = M
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

---

## 3. Loss — `loss.py`

```python
import torch
import torch.nn.functional as F


def desatencao_loss(
    logits, targets, model, x_prefnn, archetype_weights,
    alpha=0.10, beta=0.05, gamma=0.02,
):
    """
    L_total = L_ce
            + alpha * L_constelação
            - beta  * L_tensão          (maximizar distância dos polos)
            - gamma * L_individuação    (maximizar divergência das repr.)
    """
    B, S, V = logits.shape

    # L_ce
    l_ce = F.cross_entropy(
        logits.view(B * S, V), targets.view(B * S), ignore_index=-1
    )

    # L_constelação
    h       = F.normalize(x_prefnn, dim=-1)
    sim     = torch.bmm(h, h.transpose(1, 2))
    w_exp   = archetype_weights.unsqueeze(1).expand(B, S, -1)
    arch_sim = torch.bmm(w_exp, w_exp.transpose(1, 2))
    l_const = -(sim * arch_sim).mean()

    # L_tensão
    l_tension = torch.tensor(0.0, device=logits.device)
    for block in model.blocks:
        l_tension = l_tension + block.ffn.tension_loss(x_prefnn)
    l_tension = l_tension / len(model.blocks)

    # L_individuação
    h_mean    = x_prefnn.mean(dim=1, keepdim=True)
    l_individ = (x_prefnn - h_mean).norm(dim=-1).mean()

    l_total = l_ce + alpha * l_const - beta * l_tension - gamma * l_individ

    return {
        "loss":      l_total,
        "l_ce":      l_ce.detach(),
        "l_const":   l_const.detach(),
        "l_tension": l_tension.detach(),
        "l_individ": l_individ.detach(),
    }
```

---

## 4. Dataset — `dataset.py`

```python
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


class MythicDataset(Dataset):
    """
    Dataset com ordem mítica preservada.
    NÃO embaralhar — a contaminação arquetípica entre textos é intencional.
    """

    def __init__(self, file_paths: list, seq_len: int = 1024):
        self.seq_len = seq_len
        enc = tiktoken.get_encoding("gpt2")

        tokens = []
        for path in file_paths:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            tokens.extend(enc.encode(text))
            tokens.append(enc.eot_token)

        self.tokens   = torch.tensor(tokens, dtype=torch.long)
        self.n_chunks = (len(self.tokens) - 1) // seq_len

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start : start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def build_dataloader(file_paths, seq_len, batch_size, num_workers=4):
    ds = MythicDataset(file_paths, seq_len)
    return DataLoader(
        ds, batch_size=batch_size,
        shuffle=False,        # NUNCA embaralhar
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )


# Ordem mítica do corpus
# A sequência segue a ativação progressiva dos arquétipos:
# criação → herói → sombra → individuação → Self
CORPUS_ORDER = [
    "data/genesis.txt",              # Self, Grande Pai, Criança Divina
    "data/rigveda_hymns.txt",        # Espírito, quaternidade primordial
    "data/popol_vuh.txt",            # Grande Mãe, criação, Kore
    "data/iliad.txt",                # Herói, Sombra, Grande Pai
    "data/odyssey.txt",              # Herói, Anima, Trickster, Velho Sábio
    "data/gilgamesh.txt",            # Herói, Puer, amizade como Animus
    "data/homeric_hymns.txt",        # Kore (Deméter/Perséfone), Trickster (Hermes)
    "data/dante_commedia.txt",       # Individuação completa, Anima (Beatrice), Senex
    "data/grimm_tales.txt",          # todos os arquétipos em forma mínima
    "data/blake_prophetic.txt",      # Urizen (Senex), Los (Puer/Herói), Quaternidade
    "data/joyce_dubliners.txt",      # Persona, Senex, Sombra, paralisia
    "data/joyce_portrait.txt",       # Puer, individuação do ego, Anima (garota-pássaro)
    "data/joyce_ulysses.txt",        # todos os arquétipos no cotidiano
    "data/joyce_finnegans_wake.txt", # Self, ciclo, HCE/ALP como arquétipos puros
]
```

---

## 5. Treino — `train.py`

```python
import math
import os
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import wandb

from model import DesatencaoFormer
from loss import desatencao_loss
from dataset import build_dataloader, CORPUS_ORDER
from build_archetypes import build_archetype_tensors, ARCHETYPES


CONFIG = {
    # modelo
    "vocab_size":   50257,
    "d_model":      512,
    "n_heads":      8,
    "n_layers":     12,
    "d_ff":         2048,
    "seq_len":      1024,
    "dropout":      0.1,
    # treino
    "epochs":       10,
    "batch_size":   8,
    "lr":           3e-4,
    "weight_decay": 0.1,
    "warmup_steps": 500,
    # loss
    "alpha":        0.10,
    "beta":         0.05,
    "gamma":        0.02,
    # infra
    "log_every":    50,
    "save_every":   1000,
    "use_wandb":    True,
}


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Device: {device}")

    tensor_path = "archetype_tensors.pt"
    if os.path.exists(tensor_path):
        archetype_tensors = torch.load(tensor_path, map_location=device)
    else:
        archetype_tensors = build_archetype_tensors(
            ARCHETYPES, target_dim=config["d_model"], device=str(device)
        )
        torch.save(archetype_tensors, tensor_path)
    archetype_tensors = {k: v.to(device) for k, v in archetype_tensors.items()}

    model = DesatencaoFormer(
        vocab_size        = config["vocab_size"],
        d_model           = config["d_model"],
        n_heads           = config["n_heads"],
        n_layers          = config["n_layers"],
        d_ff              = config["d_ff"],
        max_seq_len       = config["seq_len"],
        archetype_tensors = archetype_tensors,
        dropout           = config["dropout"],
    ).to(device)
    print(f"Parâmetros treináveis: {model.count_parameters():,}")
    print(f"Arquétipos ativos: {list(archetype_tensors.keys())}")

    loader = build_dataloader(
        CORPUS_ORDER, seq_len=config["seq_len"], batch_size=config["batch_size"]
    )

    decay    = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = optim.AdamW(
        [{"params": decay, "weight_decay": config["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=config["lr"], betas=(0.9, 0.95),
    )

    total_steps = len(loader) * config["epochs"]

    def lr_lambda(step):
        if step < config["warmup_steps"]:
            return step / config["warmup_steps"]
        progress = (step - config["warmup_steps"]) / max(total_steps - config["warmup_steps"], 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = GradScaler(enabled=(device.type == "cuda"))

    if config["use_wandb"]:
        wandb.init(project="desatencaoformer", config=config)

    archetype_names = list(archetype_tensors.keys())
    os.makedirs("checkpoints", exist_ok=True)
    global_step = 0

    for epoch in range(config["epochs"]):
        model.train()
        for input_ids, targets in loader:
            input_ids = input_ids.to(device)
            targets   = targets.to(device)

            optimizer.zero_grad()
            with autocast(enabled=(device.type == "cuda")):
                out = model(input_ids, return_internals=True)
                losses = desatencao_loss(
                    logits            = out["logits"],
                    targets           = targets,
                    model             = model,
                    x_prefnn          = out["x_prefnn"],
                    archetype_weights = out["archetype_weights"],
                    alpha=config["alpha"], beta=config["beta"], gamma=config["gamma"],
                )

            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            if global_step % config["log_every"] == 0:
                dominant = archetype_names[out["archetype_weights"].mean(0).argmax().item()]
                log = {
                    "loss/total":        losses["loss"].item(),
                    "loss/ce":           losses["l_ce"].item(),
                    "loss/constelacao":  losses["l_const"].item(),
                    "loss/tensao":       losses["l_tension"].item(),
                    "loss/individuacao": losses["l_individ"].item(),
                    "archetype/dominant": dominant,
                    "lr":    scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "step":  global_step,
                }
                print(
                    f"ep={epoch} step={global_step} "
                    f"loss={log['loss/total']:.4f} ce={log['loss/ce']:.4f} "
                    f"arch={dominant} lr={log['lr']:.2e}"
                )
                if config["use_wandb"]:
                    wandb.log(log)

            if global_step % config["save_every"] == 0:
                torch.save({
                    "step": global_step, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "config": config,
                }, f"checkpoints/step_{global_step}.pt")

    return model


if __name__ == "__main__":
    train(CONFIG)
```

---

## 6. Inferência — `generate.py`

```python
import torch
import torch.nn.functional as F
import tiktoken
from model import DesatencaoFormer


@torch.inference_mode()
def generate(model, prompt, max_new_tokens=200, temperature=1.0, top_p=0.9, device="cuda"):
    """
    Geração com nucleus sampling.
    Retorna (texto_gerado, metadados_arquetípicos).
    archetype_log registra o arquétipo dominante por token gerado.
    """
    enc   = tiktoken.get_encoding("gpt2")
    names = list(model.archetypal_proj.archetype_names)
    model.eval()

    ids = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    log = []

    for _ in range(max_new_tokens):
        ids_ctx = ids[:, -model.pos_emb.num_embeddings:]
        out     = model(ids_ctx)
        logits  = out["logits"][:, -1, :] / temperature
        w       = out["archetype_weights"][0]

        log.append({
            "position": ids.size(1),
            "dominant": names[w.argmax().item()],
            "weights":  w.cpu().tolist(),
        })

        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove    = (cum_probs - F.softmax(sorted_logits, dim=-1)) > top_p
        sorted_logits[remove] = float('-inf')
        logits    = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)
        next_id   = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        ids       = torch.cat([ids, next_id], dim=1)

        if next_id.item() == enc.eot_token:
            break

    text = enc.decode(ids[0].tolist())
    dominant_overall = max(names, key=lambda n: sum(1 for e in log if e["dominant"] == n))
    return text, {"archetype_log": log, "dominant_overall": dominant_overall}
```

---

## 7. Estrutura de Arquivos

```
desatencaoformer/
├── build_archetypes.py
├── model.py
├── loss.py
├── dataset.py
├── train.py
├── generate.py
├── archetype_tensors.pt      # gerado por build_archetypes.py
├── checkpoints/
└── data/
    ├── genesis.txt
    ├── rigveda_hymns.txt
    ├── popol_vuh.txt
    ├── iliad.txt
    ├── odyssey.txt
    ├── gilgamesh.txt
    ├── homeric_hymns.txt
    ├── dante_commedia.txt
    ├── grimm_tales.txt
    ├── blake_prophetic.txt
    ├── joyce_dubliners.txt
    ├── joyce_portrait.txt
    ├── joyce_ulysses.txt
    └── joyce_finnegans_wake.txt
```

---

## 8. Tabela dos Arquétipos

| # | Nome | Polo + | Polo − | Em Joyce |
|---|---|---|---|---|
| 1 | `self` | totalidade/integração | inflação do ego | ciclo do FW |
| 2 | `sombra` | potencial recusado | projeção destrutiva | HCE / Phoenix Park |
| 3 | `anima` | inspiração/vida | possessão sentimental | Molly, ALP |
| 4 | `animus` | logos/clareza | dogmatismo/sentença | Stephen |
| 5 | `grande_mae` | nutrição/fertilidade | devoração/morte | ALP como rio |
| 6 | `grande_pai` | ordem/proteção | tirania/devoração do filho | HCE patriarca |
| 7 | `heroi` | transformação/coragem | hubris/destruição | Bloom/Odisseu |
| 8 | `velho_sabio` | conhecimento/orientação | feiticeiro enganador | Tirésias/Finnegan |
| 9 | `persona` | adaptação social | identificação total | Stephen artista |
| 10 | `puer_aeternus` | criatividade/liberdade | recusa de encarnar | Stephen/Ícaro |
| 11 | `senex` | estrutura/maturidade | paralisia/rigidez | Dublin de Dubliners |
| 12 | `trickster` | libertação/criatividade | caos/irresponsabilidade | Buck Mulligan |
| 13 | `kore` | potencialidade feminina | raptura/descida forçada | garota-pássaro |
| 14 | `crianca_divina` | renovação/potência pura | vulnerabilidade/abandono | o livro como nascimento |
| 15 | `espirito` | orientação/iluminação | possessão/fanatismo | fantasma de Hamlet |
| 16 | `quaternidade` | completude/integração | unilateralidade | estrutura do Ulysses |

---

## 9. Notas de Implementação

**Tensores arquetípicos:** construídos com `sentence-transformers` como bootstrap. Uma vez salvos em `archetype_tensors.pt`, são reutilizados. Se `d_model` mudar, reconstruir com o novo `target_dim`. Com 16 arquétipos e 10 passagens cada, o processo leva menos de 2 minutos em CPU.

**`shuffle=False`:** a escolha mais importante. O modelo deve ser exposto ao corpus na ordem mítica — criação antes do herói, herói antes da individuação, Joyce por último como síntese. Embaralhar destrói a contaminação arquetípica progressiva.

**Memória:** com `d_model=512`, `seq_len=1024`, `batch_size=8` e AMP, uso aproximado de 14–16 GB VRAM. Para experimentação inicial usar `batch_size=4`, `seq_len=512`.

**Diagnóstico durante treino:** o campo `archetype/dominant` no log é o indicador mais informativo. Se seguir a ordem do corpus (Grande Pai e Criança Divina em Genesis, Herói na Odisseia, Puer em Portrait, Self no Finnegans Wake), a projeção arquetípica está funcionando. Se colapsar para um único arquétipo em todos os batches, aumentar o número de passagens por arquétipo no `ARCHETYPES` ou reduzir `alpha`.

---

*"The Vico road goes round and round to meet where terms begin."*  
— Finnegans Wake, 452.21