import ufal.udpipe

class UDPipeModel:
    def __init__(self, path, lang):
        """Load given UDPipe model."""
        self.model = ufal.udpipe.Model.load(path)
        self.meta = {"lang": lang, "name": "udpipe"}
        if not self.model:
            raise Exception(f"Cannot load UDPipe model from file '{path}'")

    def process_pretokenized(self, tokens, tag=True, parse=True):
        """Process pretokenized input (list of tokens) for tagging and/or parsing."""
        # Create a new sentence
        sentence = ufal.udpipe.Sentence()

        # Add tokens to the sentence
        for i, token in enumerate(tokens, 1):
            sentence.addWord(token)  # Add token as string
            sentence.words[-1].id = i  # Set ID for the added word

        # Apply tagging if requested
        if tag:
            self.model.tag(sentence, self.model.DEFAULT)

        # Apply parsing if requested
        if parse:
            self.model.parse(sentence, self.model.DEFAULT)

        return sentence

    def post_process(self, sentence: str):
        results = []
        for word in sentence.words[1:]:  # Skip root (words[0])
            results.append({
                "token": word.form,
                "upostag": word.upostag,
                "deprel": word.deprel,
                "head": word.head
            })
        upos = []
        deps = []
        sources = []
        targets = []
        for idx, word in enumerate(results):
            upos.append(word["upostag"])
            deps.append(word["deprel"])
            if word["deprel"].lower().strip() != "root":
                sources.append(idx)
                targets.append(word["head"] - 1)
        return upos, deps, sources, targets

    def write(self, sentence, out_format="conllu"):
        """Write sentence in the required format (conllu|horizontal|vertical)."""
        output_format = ufal.udpipe.OutputFormat.newOutputFormat(out_format)
        output = output_format.writeSentence(sentence)
        return output

if __name__ == "__main__":
    # Example usage
    model_path = "/home/leon/Downloads/hindi-hdtb-ud-2.5-191206.udpipe"  # Path to Hindi model
    udpipe_model = UDPipeModel(model_path, "hi")

    # Pretokenized input (e.g., from spaCy or indicnlp)
    tokens = ['यह', 'एक', 'परीक्षणवाक्य', 'है', '।']

    # Process tokens
    sentence = udpipe_model.process_pretokenized(tokens, tag=True, parse=True)

    # Output results in CoNLL-U format
    print(udpipe_model.write(sentence, "conllu"))
    print(*udpipe_model.post_process(sentence), sep="\n")
