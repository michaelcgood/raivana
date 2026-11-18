from raivana import nlp


def test_stem_is_consistent_across_inflections():
    assert nlp.stem("records") == nlp.stem("record")
    assert nlp.stem("approving") == nlp.stem("approve") == nlp.stem("approved")
    assert nlp.stem("policies") == nlp.stem("policy")
    assert nlp.stem("accesses") == nlp.stem("access")


def test_segment_negating_prefix():
    assert nlp.segment("non-employees")[0] == "non-"
    assert nlp.negating_prefix("unmanaged") == ("un", "managed")
    assert nlp.negating_prefix("non-employees") == ("non", "employees")


def test_pos_tags_modal_and_verb():
    toks = nlp.pos_tag(nlp.tokenize("Nurses can view records"), known_verbs={"view"})
    assert [t.pos for t in toks] == ["NOUN", "AUX", "VERB", "NOUN"]


def test_split_sentences_handles_bullets_and_money():
    sents = [s for s, _ in nlp.split_sentences("Policy\n- Managers can approve $1.5k.\n2. Nobody may delete logs.")]
    assert sents == ["Policy", "Managers can approve $1.5k.", "Nobody may delete logs."]


def test_detect_language():
    assert nlp.detect_language("Only doctors can prescribe medication and nurses may view records")[0] == "en"
    assert nlp.detect_language("Solo los médicos pueden ver los registros de la clínica")[0] == "es"
