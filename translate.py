import onmt.translate
import torch
from onmt.translate.translator import Translator
from onmt.translate import TranslationBuilder
from onmt import inputters
from onmt.inputters import Dataset, OrderedIterator
import os
import glob
import argparse

import torch.nn as nn
import onmt
import onmt.inputters
import onmt.modules
import onmt.utils
import numpy as np


def translate(model_path='models/deen/char2char/_step_190000.pt',
              src_shards="/home/nbanar/pycharmProjects/data/deen/dev/char/newstest2013.de.tok",
              tgt_shards="/home/nbanar/pycharmProjects/data/deen/dev/char/newstest2013.en.tok",
              batch_size=1):
    src_reader = onmt.inputters.str2reader["text"]()
    tgt_reader = onmt.inputters.str2reader["text"]()

    scorer = onmt.translate.GNMTGlobalScorer(alpha=0.,
                                             beta=0.,
                                             length_penalty="avg",
                                             coverage_penalty="none")

    check_point = torch.load(model_path, map_location=torch.device("cuda:0"))

    vocab_fields = check_point["vocab"]

    src_text_field = vocab_fields["src"].base_field
    src_vocab = src_text_field.vocab
    src_padding = src_vocab.stoi[src_text_field.pad_token]

    tgt_text_field = vocab_fields['tgt'].base_field
    tgt_vocab = tgt_text_field.vocab
    tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]

    model_dict = check_point["model"]
    # print("onmt dict", model_dict.keys())

    model_dict["generator.0.weight"] = check_point["generator"]["0.weight"]
    model_dict["generator.0.bias"] = check_point["generator"]["0.bias"]

    # print(model_dict.keys())

    model = transformer(src_vocab, tgt_vocab, src_padding, tgt_padding)
    # print("my dict", model.state_dict().keys())
    model = model.to(torch.device("cuda:0"))
    model.load_state_dict(model_dict)

    translator = Translator(model=model, fields=vocab_fields, src_reader=src_reader, tgt_reader=tgt_reader,
                            global_scorer=scorer, max_length=500, gpu=0,
                            beam_size=25)

    data = Dataset(vocab_fields,
                   readers=([src_reader, tgt_reader]),
                   data=[("src", src_shards), ("tgt", tgt_shards)],
                   dirs=[None, None],
                   sort_key=inputters.str2sortkey["text"],
                   filter_pred=None)

    builder = TranslationBuilder(data=data, fields=vocab_fields, has_tgt=True)

    data_iter = OrderedIterator(dataset=data, device=torch.device("cuda:0"), batch_size=batch_size, batch_size_fn=None,
                                train=False, sort=False, sort_within_batch=True, shuffle=False)
    attentions = []
    target = []
    for batch in data_iter:
        trans_batch = translator.translate_batch(batch=batch, src_vocabs=[src_vocab], attn_debug=True)
        translations = builder.from_batch(trans_batch)
        print('================')
        print("src:", " ".join(translations[0].src_raw))
        print("tgt:", " ".join(translations[0].pred_sents[0]))
        print('================')
        for trans in translations:
            target.append(detokenize(trans.pred_sents[0]))
            attentions.append(trans.attns[0])

            # print(detokenize(trans.gold_sent))
            # ref.append(detokenize(trans.gold_sent))
            # print(detokenize(trans.pred_sents[0]))

    return target, attentions


def transformer(src_voc, tgt_voc, src_pad, tgt_pad):
    encoder_embeddings = onmt.modules.Embeddings(512, len(src_voc),
                                                 word_padding_idx=src_pad, position_encoding=True)

    encoder = onmt.encoders.transformer.TransformerEncoder(num_layers=4, d_model=512, heads=8, d_ff=2048, dropout=0,
                                                           attention_dropout=0, embeddings=encoder_embeddings,
                                                           max_relative_positions=0)

    decoder_embeddings = onmt.modules.Embeddings(512, len(tgt_voc),
                                                 word_padding_idx=tgt_pad, position_encoding=True)

    decoder = onmt.decoders.transformer.TransformerDecoder(num_layers=4, d_model=512, heads=8, d_ff=2048,
                                                           copy_attn=False, self_attn_type="scaled-dot", dropout=0,
                                                           attention_dropout=0,
                                                           embeddings=decoder_embeddings, max_relative_positions=0,
                                                           aan_useffn=False)

    merged_model = onmt.models.model.NMTModel(encoder, decoder)

    merged_model.generator = nn.Sequential(
        nn.Linear(512, len(tgt_voc)),
        nn.LogSoftmax(dim=-1))

    return merged_model


def detokenize(string):
    out = ''

    if type(string) is str:
        string = string.split(' ')

    for i, s in enumerate(string):
        if s != "<s>":
            out += s
        elif s == "<s>":
            out += " "
    return out


def arguments():
    parser = argparse.ArgumentParser(description='CharNMT arguments')
    parser.add_argument('--model', type=str, help='name', default=None)
    parser.add_argument('--src', type=str, help='name', default=None)
    parser.add_argument('--tgt', type=str, help='name', default=None)
    parser.add_argument('--save', type=str, help='name', default=None)
    parser.add_argument("--test", default=False, action="store_true", help="Flag to do something")
    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()

    trt, attn = translate(model_path=args.model,
                          src_shards=args.src,
                          tgt_shards=args.tgt,
                          batch_size=1)

    if not os.path.isdir(os.path.dirname(args.save + '/attn/')):
        os.makedirs(os.path.dirname(args.save + '/attn/'))

    if args.test:

        with open(args.save + '/pred.txt', 'w') as f:
            for line in trt:
                f.write(line + '\n')

        for i, a in enumerate(attn):
            np.save(args.save + '/attn/' + f'{i}.npy', a.cpu())
    else:
        with open(args.save + '/dev.txt', 'w') as f:
            for line in trt:
                f.write(line + '\n')

    print(f"{args.save} is saved")
