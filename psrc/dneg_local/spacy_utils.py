from typing import List, Any, Optional

import spacy
import os
from spacy.tokens.doc import Doc



BP = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../.."))

upos_dict = {'ADJ': 0, 'ADP': 1, 'ADV': 2, 'AUX': 3, 'CCONJ': 4, 'DET': 5, 'INTJ': 6, 'NOUN': 7,
             'NUM': 8, 'PART': 9, 'PRON': 10, 'PROPN': 11, 'PUNCT': 12, 'SCONJ': 13, 'SYM': 14,
             'VERB': 15, 'X': 16, 'SPACE': 17}

dep_dict = {"en_core_web_sm": {'ROOT': 0, 'acl': 1, 'acomp': 2, 'advcl': 3, 'advmod': 4, 'agent': 5, 'amod': 6, 'appos': 7,
                   'attr': 8, 'aux': 9, 'auxpass': 10, 'case': 11, 'cc': 12, 'ccomp': 13, 'compound': 14, 'conj': 15,
                   'csubj': 16, 'csubjpass': 17, 'dative': 18, 'dep': 19, 'det': 20, 'dobj': 21, 'expl': 22,
                   'intj': 23, 'mark': 24, 'meta': 25, 'neg': 26, 'nmod': 27, 'npadvmod': 28, 'nsubj': 29,
                   'nsubjpass': 30, 'nummod': 31, 'oprd': 32, 'parataxis': 33, 'pcomp': 34, 'pobj': 35, 'poss': 36,
                   'preconj': 37, 'predet': 38, 'prep': 39, 'prt': 40, 'punct': 41, 'quantmod': 42, 'relcl': 43,
                   'xcomp': 44},
            "de_core_news_sm": {'ROOT': 0, 'ac': 1, 'adc': 2, 'ag': 3, 'ams': 4, 'app': 5, 'avc': 6, 'cc': 7, 'cd': 8, 'cj': 9,
                   'cm': 10, 'cp': 11, 'cvc': 12, 'da': 13, 'dep': 14, 'dm': 15, 'ep': 16, 'ju': 17, 'mnr': 18,
                   'mo': 19, 'ng': 20, 'nk': 21, 'nmc': 22, 'oa': 23, 'oc': 24, 'og': 25, 'op': 26, 'par': 27,
                   'pd': 28, 'pg': 29, 'ph': 30, 'pm': 31, 'pnc': 32, 'punct': 33, 'rc': 34, 're': 35, 'rs': 36,
                   'sb': 37, 'sbp': 38, 'svp': 39, 'uc': 40, 'vo': 41},
            "nl_core_news_sm": {'ROOT': 0, 'acl': 1, 'acl:relcl': 2, 'advcl': 3, 'advmod': 4, 'amod': 5, 'appos': 6,
                                'aux': 7, 'aux:pass': 8, 'case': 9, 'cc': 10, 'ccomp': 11, 'compound:prt': 12,
                                'conj': 13, 'cop': 14, 'csubj': 15, 'dep': 16, 'det': 17, 'expl': 18, 'expl:pv': 19,
                                'fixed': 20, 'flat': 21, 'iobj': 22, 'mark': 23, 'nmod': 24, 'nmod:poss': 25,
                                'nsubj': 26, 'nsubj:pass': 27, 'nummod': 28, 'obj': 29, 'obl': 30, 'obl:agent': 31,
                                'orphan': 32, 'parataxis': 33, 'punct': 34, 'xcomp': 35},
            "sv_core_news_sm": {'ROOT': 0, 'acl': 1, 'acl:cleft': 2, 'acl:relcl': 3, 'advcl': 4, 'advmod': 5, 'amod': 6,
                                'appos': 7, 'aux': 8, 'aux:pass': 9, 'case': 10, 'cc': 11, 'ccomp': 12, 'compound:prt': 13,
                                'conj': 14, 'cop': 15, 'csubj': 16, 'dep': 17, 'det': 18, 'dislocated': 19, 'expl': 20,
                                'fixed': 21, 'flat:name': 22, 'iobj': 23, 'mark': 24, 'nmod': 25, 'nmod:poss': 26,
                                'nsubj': 27, 'nsubj:pass': 28, 'nummod': 29, 'obj': 30, 'obl': 31, 'obl:agent': 32,
                                'parataxis': 33, 'punct': 34, 'xcomp': 35},
            "da_core_news_sm": {'ROOT': 0, 'acl:relcl': 1, 'advcl': 2, 'advmod': 3, 'advmod:lmod': 4, 'amod': 5,
                                'appos': 6, 'aux': 7, 'case': 8, 'cc': 9, 'ccomp': 10, 'compound:prt': 11,
                                'conj': 12, 'cop': 13, 'dep': 14, 'det': 15, 'expl': 16, 'fixed': 17, 'flat': 18,
                                'iobj': 19, 'list': 20, 'mark': 21, 'nmod': 22, 'nmod:poss': 23, 'nsubj': 24,
                                'nummod': 25, 'obj': 26, 'obl': 27, 'obl:lmod': 28, 'obl:tmod': 29, 'punct': 30,
                                'xcomp': 31},
            "es_core_news_sm": {'ROOT': 0, 'acl': 1, 'advcl': 2, 'advmod': 3, 'amod': 4, 'appos': 5, 'aux': 6,
                                'case': 7, 'cc': 8, 'ccomp': 9, 'compound': 10, 'conj': 11, 'cop': 12, 'csubj': 13,
                                'dep': 14, 'det': 15, 'expl:impers': 16, 'expl:pass': 17, 'expl:pv': 18, 'fixed': 19,
                                'flat': 20, 'iobj': 21, 'mark': 22, 'nmod': 23, 'nsubj': 24, 'nummod': 25, 'obj': 26,
                                'obl': 27, 'parataxis': 28, 'punct': 29, 'xcomp': 30},
            "fr_core_news_sm": {'ROOT': 0, 'acl': 1, 'acl:relcl': 2, 'advcl': 3, 'advmod': 4, 'amod': 5, 'appos': 6,
                                'aux:pass': 7, 'aux:tense': 8, 'case': 9, 'cc': 10, 'ccomp': 11, 'conj': 12, 'cop': 13,
                                'dep': 14, 'det': 15, 'expl:comp': 16, 'expl:pass': 17, 'expl:subj': 18, 'fixed': 19,
                                'flat:foreign': 20, 'flat:name': 21, 'iobj': 22, 'mark': 23, 'nmod': 24, 'nsubj': 25,
                                'nsubj:pass': 26, 'nummod': 27, 'obj': 28, 'obl:agent': 29, 'obl:arg': 30, 'obl:mod': 31,
                                'parataxis': 32, 'punct': 33, 'vocative': 34, 'xcomp': 35},
            "it_core_news_sm": {'ROOT': 0, 'acl': 1, 'acl:relcl': 2, 'advcl': 3, 'advmod': 4, 'amod': 5, 'appos': 6,
                                'aux': 7, 'aux:pass': 8, 'case': 9, 'cc': 10, 'ccomp': 11, 'compound': 12, 'conj': 13,
                                'cop': 14, 'csubj': 15, 'dep': 16, 'det': 17, 'det:poss': 18, 'det:predet': 19,
                                'discourse': 20, 'expl': 21, 'expl:impers': 22, 'expl:pass': 23, 'fixed': 24,
                                'flat': 25, 'flat:foreign': 26, 'flat:name': 27, 'iobj': 28, 'mark': 29, 'nmod': 30,
                                'nsubj': 31, 'nsubj:pass': 32, 'nummod': 33, 'obj': 34, 'obl': 35, 'obl:agent': 36,
                                'parataxis': 37, 'punct': 38, 'vocative': 39, 'xcomp': 40},
            "ro_core_news_sm": {'ROOT': 0, 'acl': 1, 'advcl': 2, 'advcl:tcl': 3, 'advmod': 4, 'advmod:tmod': 5,
                                'amod': 6, 'appos': 7, 'aux': 8, 'aux:pass': 9, 'case': 10, 'cc': 11, 'cc:preconj': 12,
                                'ccomp': 13, 'ccomp:pmod': 14, 'compound': 15, 'conj': 16, 'cop': 17, 'csubj': 18,
                                'csubj:pass': 19, 'dep': 20, 'det': 21, 'expl': 22, 'expl:impers': 23, 'expl:pass': 24,
                                'expl:poss': 25, 'expl:pv': 26, 'fixed': 27, 'flat': 28, 'goeswith': 29, 'iobj': 30,
                                'mark': 31, 'nmod': 32, 'nmod:tmod': 33, 'nsubj': 34, 'nsubj:pass': 35, 'nummod': 36,
                                'obj': 37, 'obl': 38, 'obl:agent': 39, 'obl:pmod': 40, 'orphan': 41, 'parataxis': 42,
                                'punct': 43, 'vocative': 44, 'xcomp': 45},
            "zh_core_web_sm": {'ROOT': 0, 'acl': 1, 'advcl:loc': 2, 'advmod': 3, 'advmod:dvp': 4, 'advmod:loc': 5,
                               'advmod:rcomp': 6, 'amod': 7, 'amod:ordmod': 8, 'appos': 9, 'aux:asp': 10, 'aux:ba': 11,
                               'aux:modal': 12, 'aux:prtmod': 13, 'auxpass': 14, 'case': 15, 'cc': 16, 'ccomp': 17,
                               'compound:nn': 18, 'compound:vc': 19, 'conj': 20, 'cop': 21, 'dep': 22, 'det': 23,
                               'discourse': 24, 'dobj': 25, 'etc': 26, 'mark': 27, 'mark:clf': 28, 'name': 29, 'neg': 30,
                               'nmod': 31, 'nmod:assmod': 32, 'nmod:poss': 33, 'nmod:prep': 34, 'nmod:range': 35,
                               'nmod:tmod': 36, 'nmod:topic': 37, 'nsubj': 38, 'nsubj:xsubj': 39, 'nsubjpass': 40,
                               'nummod': 41, 'parataxis:prnmod': 42, 'punct': 43, 'xcomp': 44},
            "ja_core_news_sm": {'ROOT': 0, 'acl': 1, 'advcl': 2, 'advmod': 3, 'amod': 4, 'aux': 5, 'case': 6, 'cc': 7,
                                'ccomp': 8, 'compound': 9, 'cop': 10, 'csubj': 11, 'dep': 12, 'det': 13,
                                'dislocated': 14, 'fixed': 15, 'mark': 16, 'nmod': 17, 'nsubj': 18, 'nummod': 19,
                                'obj': 20, 'obl': 21, 'punct': 22},
            "ru_core_news_sm": {'ROOT': 0, 'acl': 1, 'acl:relcl': 2, 'advcl': 3, 'advmod': 4, 'amod': 5, 'appos': 6,
                                'aux': 7, 'aux:pass': 8, 'case': 9, 'cc': 10, 'ccomp': 11, 'compound': 12, 'conj': 13,
                                'cop': 14, 'csubj': 15, 'csubj:pass': 16, 'dep': 17, 'det': 18, 'discourse': 19,
                                'expl': 20, 'fixed': 21, 'flat': 22, 'flat:foreign': 23, 'flat:name': 24, 'iobj': 25,
                                'list': 26, 'mark': 27, 'nmod': 28, 'nsubj': 29, 'nsubj:pass': 30, 'nummod': 31,
                                'nummod:entity': 32, 'nummod:gov': 33, 'obj': 34, 'obl': 35, 'obl:agent': 36,
                                'orphan': 37, 'parataxis': 38, 'punct': 39, 'xcomp': 40},
            "hi_udpipe": {'acl': 0, 'acl:relcl': 1, 'advcl': 2, 'advmod': 3, 'amod': 4, 'aux': 5, 'aux:pass': 6,
                          'case': 7, 'cc': 8, 'ccomp': 9, 'compound': 10, 'conj': 11,'cop': 12, 'dep': 13, 'det': 14,
                          'dislocated': 15, 'iobj': 16, 'mark': 17, 'nmod': 18, 'nsubj': 19, 'nsubj:pass': 20,
                          'nummod': 21, 'obj': 22, 'obl': 23, 'punct': 24, 'root': 25, 'vocative': 26, 'xcomp': 27},
            "ar_udpipe": {'acl': 0, 'acl:relcl': 1, 'advcl': 2, 'advmod': 3, 'advmod:emph': 4, 'amod': 5, 'appos': 6,
                          'aux': 7, 'aux:pass': 8, 'case': 9, 'cc': 10, 'ccomp': 11, 'conj': 12, 'cop': 13, 'csubj': 14,
                          'csubj:pass': 15, 'dep': 16, 'det': 17, 'discourse': 18, 'dislocated': 19, 'fixed': 20,
                          'flat:foreign': 21, 'iobj': 22, 'mark': 23, 'nmod': 24, 'nsubj': 25, 'nsubj:pass': 26,
                          'nummod': 27, 'obj': 28, 'obl': 29, 'obl:arg': 30, 'orphan': 31, 'parataxis': 32, 'punct': 33,
                          'root': 34, 'xcomp': 35}
            }

lang_dict = {"": "en_core_web_sm",
             "en": "en_core_web_sm",
             "de": "de_core_news_sm",
             "nl": "nl_core_news_sm",
             "sv": "sv_core_news_sm",
             "da": "da_core_news_sm",
             "es": "es_core_news_sm",
             "fr": "fr_core_news_sm",
             "it": "it_core_news_sm",
             "ro": "ro_core_news_sm",
             "zh": "zh_core_web_sm",
             "hi": "hi_udpipe",
             "jap": "ja_core_news_sm",
             "ru": "ru_core_news_sm",
             "ar": "ar_udpipe"}

# Custom tokenizer for space-separated pre-tokenized input
class PreTokenizedTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        # Split space-separated string into tokens
        tokens = text.split("[&JOIN&]")
        return Doc(self.vocab, words=tokens)

def get_spacy_model(lang: str = "en") -> Any:
    if lang == "german":
        lang = "de"
    elif lang == "english":
        lang = "en"
    elif lang == "ja":
        lang = "jap"

    if lang in ['en', 'de', 'nl', 'sv', 'da', 'es', 'fr', 'it', 'ro', 'zh', 'jap', 'ru']:
        spacy_model = spacy.load(lang_dict[lang])
        # Replace default tokenizer
        spacy_model.tokenizer = PreTokenizedTokenizer(spacy_model.vocab)
        return spacy_model
    elif lang in ["hi", "ar"]:
        from .ud_pipe import UDPipeModel
        model_path_dict = {"hi": f"{BP}/data/UDPIPE/hindi-hdtb-ud-2.5-191206.udpipe",
                           "ar": f"{BP}/data/UDPIPE/arabic-padt-ud-2.5-191206.udpipe"}
        return UDPipeModel(model_path_dict[lang], lang)
    else:
        raise Exception(f"Unknown language: {lang}")

EDGE_PUNCT = ".,;:!?\"'()[]{}«»„“”‚‘’…–—"


def parser_form(token: str) -> str:
    """Whitespace tokens keep attached punctuation ("nicht.", "nee,"); strip it for the
    parser so tags/arcs are not skewed, while keeping a 1:1 token alignment."""
    stripped = token.strip(EDGE_PUNCT)
    return stripped if stripped else token


def process_sent_spacy(sent: List[str], spacy_model: Any):
    if isinstance(spacy_model, spacy.language.Language):
        doc = spacy_model("[&JOIN&]".join(parser_form(tok) for tok in sent))
        assert len(doc) == len(sent), (sent, [t.text for t in doc])
        upos_layer = [token.pos_ for token in doc]
        dep_layer = [token.dep_ for token in doc]
        sources = []
        targets = []
        for token in doc:
            if token.dep_ != "ROOT":
                # Edge from child (token.i) to head (token.head.i)
                sources.append(token.i)
                targets.append(token.head.i)
        # assert len(sent) == len(upos_layer) == len(dep_layer), print(*[sent, upos_layer, dep_layer])
    elif hasattr(spacy_model, "process_pretokenized"):  # UDPipeModel
        upos_layer, dep_layer, sources, targets = spacy_model.post_process(spacy_model.process_pretokenized(sent, tag=True, parse=True))
    else:
        raise Exception(f"Unknown model: {spacy_model}")


    return upos_layer, dep_layer, [sources, targets]


def get_possible_deprels(spacy_model):
    parser = spacy_model.get_pipe("parser")

    # Retrieve all dependency labels
    deprel_tags = parser.labels

    # Print the sorted list of dependency labels
    # print("All possible deprel tags:", sorted(deprel_tags))
    print({i: v for v, i in enumerate(deprel_tags)})


if __name__ == "__main__":
    # print(process_sent_spacy("This is a test .".split(" "), get_spacy_model("en")))

    # print(process_sent_spacy("यह एक परीक्षण है।".split(" "), get_spacy_model("hi")))
    import spacy_udpipe
    """rr = {
     "sv": "sv_core_news_sm",
     "da": "da_core_news_sm",
     "es": "es_core_news_sm",
     "fr": "fr_core_news_sm",
     "it": "it_core_news_sm",
     "ro": "ro_core_news_sm",
     "zh": "zh_core_web_sm",
     "jap": "ja_core_news_sm",
     "ru": "ru_core_news_sm"}
    for key in rr:
        print(rr[key])
        get_possible_deprels(spacy.load(rr[key]))"""

    rr = {
             "hi": f"{BP}/data/UDPIPE/hindi-hdtb-ud-2.5-191206.udpipe",
             "ar": f"{BP}/data/UDPIPE/arabic-padt-ud-2.5-191206.udpipe"}

    for key in rr:
        print(rr[key])
        get_possible_deprels(spacy_udpipe.load_from_path(lang=key, path=rr[key]))