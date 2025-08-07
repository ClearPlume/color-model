import sentencepiece as spm

spm.SentencePieceTrainer.Train(
    input='../data/color_words.txt',
    model_prefix='color_unigram',
    vocab_size=283 * 2,
    model_type='unigram',
    character_coverage=1.0,
    input_sentence_size=1000000,
    shuffle_input_sentence=True,
    hard_vocab_limit=False,
)
