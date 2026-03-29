"""
Build pre-computed archetypal curvature tensors from mythic passages.

Each archetype is represented by a covariance matrix C_k computed from
sentence embeddings of its canonical manifestations. These tensors have
no gradient -- they are the a priori structure that precedes learning.

The collective unconscious that precedes the individual.

16 Jungian archetypes, 10 passages each, covering both positive and
negative poles of each archetype.

Sources:
    Jung, C.G. -- The Archetypes and the Collective Unconscious (CW 9/1)
    Jung, C.G. -- Aion (CW 9/2)
    Jung, C.G. -- Psychology and Alchemy (CW 12)
    Von Franz, M-L. -- The Feminine in Fairy Tales
    Hillman, J. -- Re-Visioning Psychology
"""

import torch
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# JUNGIAN ARCHETYPES -- complete set (16)
#
# Each archetype contains passages from mythic, literary, and Joycean
# texts that instantiate it. The tensor C_k is built from the covariance
# of these manifestations -- captures the SHAPE of the archetypal field,
# not its average content.
#
# Polarities: each archetype has a positive and negative pole.
# Passages must cover BOTH poles -- the tension between them is what
# makes the tensor useful.
# ---------------------------------------------------------------------------

ARCHETYPES = {
    # -- 1. SELF -----------------------------------------------------------
    # The archetype of psychic totality. Distinct from the ego: precedes
    # and exceeds the ego. Center and circumference simultaneously.
    # Symbol: mandala, circle, quaternity, philosopher's stone, Christ as anthropos.
    # In Joyce: the cycle of Finnegans Wake that begins where it ends.
    # Polarity: integration <-> inflation (ego confused with Self = psychosis).
    "self": [
        "The mandala is the self made visible -- center and circumference at once",
        "That art thou: the drop knowing itself as ocean",
        "All rivers run into the sea; yet the sea is not full; unto the place from whence the rivers come thither they return",
        "The stone which the builders rejected has become the cornerstone",
        "What was before Abraham was -- I am",
        "The end of all our exploring will be to arrive where we started and know the place for the first time",
        "riverrun, past Eve and Adam's, from swerve of shore to bend of bay, brings us by a commodius vicus of recirculation back to Howth Castle and Environs",
        "The Self is not only the centre but also the whole circumference which embraces both conscious and unconscious",
        "As above, so below; as within, so without -- the whole in every part",
        "I am Alpha and Omega, the beginning and the end, the first and the last",
    ],
    # -- 2. SHADOW ---------------------------------------------------------
    # What the ego rejects and projects onto the other. Not evil itself --
    # it is the unintegrated. Can be positive (refused potential) or
    # negative (denied impulse). The Shadow is always of the same nature as the ego.
    # In Joyce: HCE and the unnamed crime of Phoenix Park.
    # Symbol: the double, the enemy brother, the pursuer in dreams.
    # Polarity: denial <-> integration; projection <-> recognition.
    "shadow": [
        "I am the spirit that always denies, and rightly so, for all that exists deserves to perish",
        "Better to reign in Hell than serve in Heaven",
        "Here Comes Everybody -- the man of the crowd, criminal and saint indistinguishable",
        "Set dismembered Osiris and scattered the pieces across the length of Egypt",
        "What the hand dare seize the fire? Did he who made the Lamb make thee?",
        "The thing I feared has come upon me; what I dreaded has happened to me",
        "In the cave of the heart the shadow waits, wearing your own face",
        "He that is without sin among you, let him cast the first stone",
        "Mr Hyde emerged from Dr Jekyll and could not be put back",
        "The unconscious is not a den of iniquity; it is simply the unknown",
    ],
    # -- 3. ANIMA ----------------------------------------------------------
    # The feminine in the masculine psyche. Not the real woman -- it is the
    # inner image of the feminine that projects and distorts perception.
    # Four levels: Eve (biological) -> Helen (aesthetic) ->
    # Mary (spiritual) -> Sophia (sapiential).
    # In Joyce: Molly Bloom ("Yes"), ALP, the girl on the beach in Portrait.
    # Polarity: muse <-> possession; inspiration <-> sentimental trap.
    "anima": [
        "Yes I said yes I will Yes -- the affirmation beyond all argument",
        "Penelope weaving and unweaving, faithful in betrayal, betraying in fidelity",
        "Beatrice appeared to me clothed in the colour of living flame, ancient and ageless",
        "She is the Muse who speaks through the poet and then abandons him to silence",
        "Anna Livia Plurabelle -- her waters are her words and her words are her waters",
        "A girl stood before him in midstream, alone and still, gazing out to sea",
        "The eternal feminine draws us onward and upward -- das Ewig-Weibliche",
        "She who must be obeyed -- the goddess who destroys those who cannot bear her presence",
        "I have been woman and man, young and old, plant and bird and silent fish",
        "The Anima is the archetype of life itself -- irrational, capricious, possessive, delightful",
    ],
    # -- 4. ANIMUS ---------------------------------------------------------
    # The masculine in the feminine psyche. Distinct from Anima -- operates
    # differently: where Anima is mood and irrationality, Animus is
    # opinion and logos. Four levels: physical power -> initiative ->
    # logos (word) -> spiritual meaning.
    # In Joyce: Stephen Dedalus as Animus of an epoch.
    # Polarity: clarity <-> dogmatism; logos <-> unyielding sentence.
    "animus": [
        "Non serviam -- I will not serve that in which I no longer believe",
        "The word is the deed; in the beginning was the logos and the logos was action",
        "He spoke with authority, and not as the scribes -- the voice that does not negotiate",
        "I go to encounter for the millionth time the reality of experience and to forge in the smithy of my soul the uncreated conscience of my race",
        "The masculine spirit in woman appears as an opinion disconnected from relation",
        "Athena springs fully armed from the head of Zeus -- thought that precedes feeling",
        "The animus says: it is so -- and closes the question before it opens",
        "He who argues with a woman is arguing with her animus, not with her",
        "The sword is the animus -- it cuts, divides, names, and kills what it names",
        "In the beginning was the Word -- and the Word was already a judgment",
    ],
    # -- 5. GREAT MOTHER ---------------------------------------------------
    # The archetype of the primordial feminine, prior to individuation.
    # Distinct from Anima: the Great Mother is collective and transpersonal.
    # Includes both nurturing AND devouring aspects -- both are the same figure.
    # In Joyce: ALP as river, as earth, as origin and destination.
    # Symbol: earth, cave, sea, moon, cauldron, vessel, night.
    # Polarity: nurturance <-> devouring; fertility <-> death.
    "great_mother": [
        "Earth mother searching for her lost daughter through barren winter fields -- Demeter's grief that made the world sterile",
        "Goddess gathering the scattered pieces of her beloved across Egypt to restore him to life -- Isis reassembling Osiris",
        "Dark goddess dancing on corpses, skull necklace rattling, destroyer and creatrix simultaneously -- Kali",
        "Mother holding her dead son, pieta of absolute sorrow and absolute grace inseparable",
        "She who kills her own children rather than yield them to another -- Medea, love as annihilation",
        "the earth was without form and void, and darkness was upon the face of the deep, and the spirit moved upon the waters",
        "riverrun, past Eve and Adam's -- ALP as the Liffey, the maternal river that is also the letter and the body",
        "The Great Mother is the earth that receives the dead and the womb that gives birth -- the same darkness",
        "Hecate at the crossroads, three-faced, the moon in all its phases -- maiden, mother, crone",
        "She swallowed her children whole so they could not surpass her -- Kronos in feminine form",
    ],
    # -- 6. GREAT FATHER ---------------------------------------------------
    # The archetype of the primordial and transpersonal masculine. Order, law,
    # spirit, collective logos. Distinct from the Wise Old Man -- the Great Father
    # is power and structure; the Wise Old Man is knowledge and initiation.
    # Symbol: sky, thunder, mountain, king, tablet of law, sword.
    # Polarity: protection/ordering <-> tyranny/devouring the son.
    "great_father": [
        "I am the Lord your God -- the voice from the whirlwind that does not explain itself",
        "Zeus hurling the thunderbolt from Olympus -- power that needs no justification",
        "The father who demands sacrifice: Abraham raising the knife over his son",
        "Kronos devouring his children to prevent being surpassed -- the father who devours the future",
        "The law is the law -- Creon who will not yield even when the law kills",
        "He descended to his fathers; in the city of David they buried him",
        "HCE -- Here Comes Everybody -- the fallen patriarch whose guilt structures the dream",
        "The king's two bodies: the mortal man and the immortal office that outlasts him",
        "Wotan wanders the world he can no longer govern, trading an eye for wisdom he cannot use",
        "The father's word is law -- and the word always arrives too late or too early",
    ],
    # -- 7. HERO -----------------------------------------------------------
    # The ego in development confronting the unconscious.
    # Not virtue -- it is transformation through danger.
    # The hero's journey: separation -> initiation -> return.
    # In Joyce: Bloom as everyday hero (Ulysses without Troy).
    # Symbol: sword, shield, dragon to be slain, descent to the underworld.
    # Polarity: courage/transformation <-> hubris/destruction.
    "hero": [
        "No man am I called, and No man is what my mother and father call me -- Odysseus in the cyclops' cave",
        "He who saw the deep, the country's foundation -- Gilgamesh after Enkidu's death",
        "Stately, plump Buck Mulligan -- but the true hero descends: Bloom walking through Dublin's underworld",
        "His soul swooned slowly as he heard the snow falling faintly through the universe -- the moment of dissolution before return",
        "Here am I -- the one who descends and does not return unchanged",
        "The hero's journey: road of trials, innermost cave, ordeal, reward, road back",
        "I have slain the Minotaur in the labyrinth and the thread leads me out -- but I forgot to change the sail",
        "Into that darkness peering, long I stood there, wondering, fearing, doubting",
        "He emptied himself, taking the form of a servant -- the hero who descends by choice",
        "The dragon hoards the gold: the hero's task is always to take what was locked away",
    ],
    # -- 8. WISE OLD MAN ---------------------------------------------------
    # The spirit as archetype -- not power (Great Father) but knowledge.
    # Appears as guide, initiator, bearer of meaning.
    # Distinct from the Great Father: does not command -- guides and questions.
    # In Joyce: the dead Finnegan who knows everything; Tiresias in Eliot.
    # Symbol: the elder, the wizard, the hermit, the blind seer.
    # Polarity: wisdom <-> deceiving sorcerer (negative aspect).
    "wise_old_man": [
        "I am Tiresias, blind and knowing both sexes and all futures, and I have foresuffered all",
        "What are you asking, Oedipus? You do not want to know the answer",
        "Thus Finn MacCool lies sleeping under the hill, and will rise when Ireland needs him",
        "The Fisher King waits in the wasteland for the one who asks the question that heals",
        "In the beginning was the Word -- and the Word was already a riddle requiring interpretation",
        "He who knows does not speak; he who speaks does not know -- the sage's paradox",
        "Prospero: I have bedimmed the noontide sun and by my art provoked the winds -- and then drowned the book",
        "The Merlin who knows Arthur's fate and cannot change it, can only accompany it",
        "Old age should burn and rave at close of day -- but the sage knows when to stop burning",
        "Do not go where the path may lead; go instead where there is no path and leave a trail",
    ],
    # -- 9. PERSONA --------------------------------------------------------
    # The social mask -- the face the ego presents to the world.
    # Not false by nature, but becomes pathological when the ego
    # identifies totally with it.
    # In Joyce: the Stephen who performs the artist; the Bloom who performs
    # the resigned husband.
    # Symbol: the mask, the uniform, the title, the social role.
    # Polarity: social adaptation <-> loss of Self; conformity <-> rigidity.
    "persona": [
        "All the world's a stage, and all the men and women merely players -- each with his entrance and his exit",
        "He wore his smile like a mask -- beneath it nothing, or everything, or the same smile again",
        "I am large, I contain multitudes -- but which multitude is presenting itself now?",
        "The uniform makes the man: remove the rank and what remains is a man afraid",
        "Stephen Dedalus, teacher, artist, Irishman -- each name a costume, none of them the wearer",
        "The persona is the individual's system of adaptation to the world -- and his prison",
        "He played the part so long he forgot there was a player",
        "Smile and smile and be a villain -- the persona as weapon",
        "Take off the mask: another mask beneath it. Take off that one too",
        "I have met them at close of day coming with vivid faces from counter or desk among grey eighteenth-century houses",
    ],
    # -- 10. PUER AETERNUS -------------------------------------------------
    # The eternal youth -- archetype of the divine son, of unrealized
    # possibility, of the refusal to incarnate and age.
    # Refuses limitation, lives in pure possibility.
    # In Joyce: Stephen Dedalus explicitly (Daedalus the artificer, Icarus the son).
    # Symbol: Icarus, Narcissus, Peter Pan, the prince who won't take the throne.
    # Polarity: creativity/freedom <-> refusal of commitment/premature death.
    "puer_aeternus": [
        "I will not serve that in which I no longer believe -- the Puer's manifesto",
        "Fabulous artificer -- the hawklike man -- the artist soaring above the world he refuses to inhabit",
        "He had daedalian wings and flew too near the sun -- Icarus as the type of all pure aspiration",
        "I shall not grow old -- Peter Pan's refusal which is also his death-in-life",
        "The boy who would not grow up lives in a country where no one ages and nothing has consequences",
        "Narcissus at the pool: in love with his own reflection, which is also his drowning",
        "Stephen looked at the tundish and thought: I am not what they made me. I will make myself",
        "The eternal youth has wings but no roots -- he can fly anywhere and land nowhere",
        "Dionysus torn apart by the Titans and reassembled -- the god who dies young always",
        "His gifts were extraordinary and his life was brief and that was the same thing",
    ],
    # -- 11. SENEX ---------------------------------------------------------
    # The old/Saturn pole -- opposite of the Puer. Where the Puer is freedom
    # and refusal, the Senex is order, weight, chronological law, melancholy.
    # Not the Wise Old Man -- the Senex is rigid and saturnine, not wise.
    # In Joyce: the paralyzed Dublin of Dubliners; Father Simon Dedalus.
    # Symbol: Saturn devouring the son, the king who won't abdicate, unyielding law.
    # Polarity: structure/maturity <-> rigidity/paralysis/devouring the new.
    "senex": [
        "The old men shall dream dreams -- but first they must stop dreaming that they are still young",
        "Saturnus eating his children: the old order consuming what would succeed it",
        "He had lived too long in one city; the streets had grown into him like roots",
        "The law is the law and I am its servant -- Creon, the Senex who cannot yield",
        "Dublin. I always knew I would leave and I never left -- the Senex city",
        "The paralysis of the living dead: not Lazarus raised but Lazarus who declined the resurrection",
        "He had his convictions and he had his position and between them there was no space for the world to enter",
        "Old Goriot giving everything to daughters who ceased to see him -- the Senex as tragic figure",
        "The weight of years is not wisdom; it is sometimes only weight",
        "He was not old; he had simply stopped moving -- and stillness looked like age",
    ],
    # -- 12. TRICKSTER -----------------------------------------------------
    # The trickster -- transgresses boundaries, violates taboos, provokes
    # transformation through chaos and laughter. Neither hero nor villain.
    # Liminal figure: lives at the edges of systems.
    # In Joyce: Buck Mulligan; Joyce himself as author.
    # Symbol: Hermes, Loki, Eshu, Coyote, the King's Fool.
    # Polarity: liberation/creativity <-> destruction/irresponsibility.
    "trickster": [
        "Buck Mulligan: the mocker, the usurper, the one who laughs at what the other holds sacred",
        "Loki who cut off Sif's hair for a joke -- and then had to save the world from the consequences",
        "Hermes: messenger, thief, guide of souls, inventor of the lyre stolen from Apollo",
        "Coyote stole fire and got burned and stole it again -- the trickster cannot stop",
        "The Fool in the Tarot: zero, the unnumbered card, the one who walks off cliffs",
        "Eshu at the crossroads, where all paths meet and all bargains are made -- and broken",
        "He who asks the wrong question at the right moment -- and everything changes",
        "The jester speaks truth to the king because no one takes him seriously enough to punish him",
        "Mercurius: the coincidence of opposites, the god who is simultaneously divine and demonic",
        "The trickster breaks the rule not to destroy the system but to reveal that the system was never the point",
    ],
    # -- 13. KORE / MAIDEN ------------------------------------------------
    # The young feminine -- not the mother, not the wise woman, but the
    # unrealized feminine potentiality. The maiden who descends, who is
    # abducted, who undergoes initiation.
    # Distinct from Anima: Kore is transpersonal and collective.
    # In Joyce: the bird-girl on the beach in Portrait; Gerty MacDowell.
    # Symbol: Persephone, Cinderella, the tower, the flower, the virgin.
    # Polarity: innocence/openness <-> abduction/forced descent/transformation.
    "kore": [
        "Persephone gathering flowers in the meadow when the earth opened and swallowed her -- the descent that was also a marriage",
        "A girl stood before him in midstream, gazing out to sea -- the epiphany on the strand",
        "She was like a wild bird, fearless and still -- the image of what art might become",
        "Gerty MacDowell on the rocks, arranging herself to be seen -- the Kore performing Kore",
        "Sleeping Beauty: the hundred-year sleep between childhood and womanhood",
        "Cinderella among the ashes, waiting for the transformation she does not yet know she carries",
        "The maiden in the tower has hair long enough to climb -- but she did not let it down for herself",
        "Proserpina in the underworld, eating the pomegranate seeds -- each seed a month of winter",
        "She was fourteen and she stood at the window and everything was still possible",
        "The virgin goddess: not innocent but undefeated, not cold but unclaimed",
    ],
    # -- 14. DIVINE CHILD --------------------------------------------------
    # The archetype of birth, of pure potentiality, of the beginning
    # that precedes all determination. Distinct from Puer -- the Divine Child
    # is prior to individuation, symbol of the emerging Self.
    # In Joyce: the book itself as child -- Finnegans Wake as birth that never ceases.
    # Symbol: the Christ child, infant Horus, infant Krishna, the young Fisher King.
    # Polarity: potency/renewal <-> vulnerability/abandonment/infanticide.
    "divine_child": [
        "And she brought forth her firstborn son and wrapped him in swaddling clothes and laid him in a manger",
        "The infant Horus hidden in the papyrus marsh while Set hunted him -- the divine child in danger",
        "The child Krishna stealing butter, dancing on the serpent, holding the world in his mouth",
        "A child is born to us, a son is given -- and the government shall be upon his shoulders",
        "Out of the mouth of babes and sucklings thou hast perfected praise",
        "The divine child is abandoned, exposed, threatened -- and survives because the world cannot yet bear him",
        "Every child arrives as a stranger from somewhere the adults have forgotten",
        "The boy in the library reading: the world has not yet closed around him",
        "The child who asks why -- before learning that why is not always a welcome question",
        "In my beginning is my end; in my end is my beginning -- the child as the Self not yet divided",
    ],
    # -- 15. SPIRIT --------------------------------------------------------
    # The archetype of spirit as such -- not a personal figure but
    # a presence that orients, illuminates, or disturbs.
    # Appears in fairy tales as the supernatural helper, the speaking
    # animal, the dwarf who knows the true name.
    # In Joyce: Hamlet's father's ghost (cited in Ulysses); the
    # spirit of the sleeping Finno.
    # Symbol: the wind, the flame, the voice in the desert, the Socratic daemon.
    # Polarity: orientation/illumination <-> possession/fanaticism.
    "spirit": [
        "The wind bloweth where it listeth, and thou hearest the sound thereof, but canst not tell whence it cometh",
        "I am thy father's spirit, doomed for a certain term to walk the night -- Hamlet's ghost",
        "The still small voice after the earthquake and the fire -- not in the spectacle but in the silence",
        "He was filled with the spirit and went into the wilderness -- the spirit that drives out, not in",
        "In the fairy tale the old man in the forest knows the path; he is the spirit appearing as helper",
        "Socrates heard his daimon: not a god, not a man, but the voice that said no when no should be said",
        "Pentecost: the spirit as fire distributed -- not hierarchy but simultaneous ignition",
        "The muse descends and the poet writes what he did not intend to write",
        "Geist: the spirit of an age that no individual chose but all embody",
        "The spirit appears as what is needed: guide, trickster, gift, demand",
    ],
    # -- 16. QUATERNITY ----------------------------------------------------
    # Jung's four functions (thinking, feeling, sensation, intuition)
    # structure the psyche as quaternity -- the number 4 as symbol of the
    # complete Self. The inferior function is always the most unconscious
    # and the gateway to the unconscious.
    # In Joyce: Ulysses divided into episodes corresponding to organs and arts --
    # a quaternity expanded into structure.
    # Symbol: the cross, the four elements, the four evangelists, the wheel.
    # Polarity: completeness <-> one-sidedness; integration <-> possessive inferior function.
    "quaternity": [
        "Four rivers went out from Eden -- and the four is always the completion of the three",
        "Earth, water, fire, air -- the four that compose all that decomposes",
        "The four evangelists: the man, the lion, the ox, the eagle -- four faces of one message",
        "Thinking, feeling, sensation, intuition: the cross on which the psyche is crucified and integrated",
        "The quaternity is the minimum structure of wholeness -- three is always incomplete",
        "Jung's four functions: the superior function we trust, the inferior function that betrays us",
        "The table has four legs -- remove one and it is no longer a table but a philosophical problem",
        "North, south, east, west -- the four directions that make a centre possible",
        "Ulysses: organ, art, colour, symbol -- the book as mandala of the human",
        "The inferior function: what we cannot do well and therefore fear and project and eventually must face",
    ],
}


def build_archetype_tensors(
    archetypes: dict,
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    target_dim: int = 256,
    device: str = "cpu",
) -> dict:
    """
    Build curvature tensors C_k for each archetype.

    Returns dict mapping archetype name to tensor of shape [target_dim, target_dim].
    Tensors have no gradient.
    """
    encoder = SentenceTransformer(encoder_name, device=device)
    encoder_dim = encoder.get_sentence_embedding_dimension()

    print(f"  Encoder dim: {encoder_dim}, target dim: {target_dim}")
    print(f"  Archetypes: {len(archetypes)}")

    tensors = {}
    for name, passages in archetypes.items():
        E = torch.tensor(
            encoder.encode(passages, normalize_embeddings=True),
            dtype=torch.float32,
            device=device,
        )  # [m, encoder_dim]

        E = E - E.mean(dim=0, keepdim=True)
        C_raw = (E.T @ E) / (len(passages) - 1)  # [encoder_dim, encoder_dim]

        # Resize to target_dim via fixed random projection
        # Johnson-Lindenstrauss: preserves distances with high probability
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
    print("Building archetypal tensors...")
    print(f"Total archetypes: {len(ARCHETYPES)}")
    tensors = build_archetype_tensors(ARCHETYPES, target_dim=256)
    torch.save(tensors, "archetype_tensors.pt")
    print("Saved: archetype_tensors.pt")
