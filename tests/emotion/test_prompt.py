from chatbot.emotion.config import load_taxonomy, CONFIG_ROOT

def test_examples_include_both_new_labels_and_selection_is_auditable():
    from chatbot.emotion.prompt import load_examples
    from chatbot.emotion.retrieval import select_examples
    examples=load_examples(CONFIG_ROOT/'emotion_examples.json',load_taxonomy())
    assert {'neutral','no_emotion'} <= {e['emotion'] for e in examples}
    selected=select_examples(examples,'convert file PDF',[],limit=4)
    assert len(selected)==4
    assert all('score' in e and 'reason' in e for e in selected)


def test_retrieval_limit_and_prior_are_configurable():
    from chatbot.emotion.retrieval import select_examples
    examples=[{'dialogue':'apple','emotion':'happy'},{'dialogue':'banana','emotion':'sad'}]
    assert select_examples(examples,'apple',['sad'],limit=0,prior_boost=0)==[]
    assert select_examples(examples,'apple',['sad'],limit=1,prior_boost=0)[0]['emotion']=='happy'
    assert select_examples(examples,'apple',['sad'],limit=1,prior_boost=10)[0]['emotion']=='sad'
