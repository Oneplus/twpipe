twpipe
======

Twpipe is a pipeline toolkit that parses raw tweets into universal
dependencies. Twpipe contains the following components:
* a joint sentence sentence segmentor and tokenizer;
* a POS tagger;
* a transition-based parser (with parser ensemble and distillation supports).

For technique details, please refer our NAACL 2018 paper
(for pipeline without sentence segmentor) and ACL 2018 paper
(for parser ensemble and distillation).

## Pre-requirements

* GCC (version greater than 5): `json.hpp` requires syntax that is not
supports by GCC 4.X.
* MSVC 18.x: I've successfully compiled `twpipe` on MSVC 18. If you got
any problem, please let me know.
* boost: we use its `program_options`.

## Compiling

In the project root directory, run the following commands

```
git submodule init
git submodule update
mkdir build
cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/your/eigen3/
```

## Testing on Raw Data
First unzip the released model with
```
bunzip2 -k model/en_ewt_en_tweebank_train.model.json.bz2
```

Then go back to the root directory.
Suppose your input tweets are stored in the `input_file' with
each tweet in one line.
Run the following commands:
```
./bin/twpipe --segment-and-tokenize --postag --parse \
    --model model/en_ewt_en_tweebank_train.model.json input_file
```

The conllu-formatted output dumped to `stdout`.

### Important Notes

1. The postagger we shipped in `twpipe` is a naive bidirectional
 LSTM sequence tagger which performs worse than that of
 Owoputi et al. (2015). We suggest using theirs instead.
2. We found that also doing sentence segmentation leads to
 better parsing performance.
3. Specifying word embeddings with `--embedding ./data/glove.twitter.27B.100d.txt`
 will lead better performance.


## Training on Tweebank

```
./bin/twpipe \
    --dynet-seed 1 \
    --train \
    --heldout ./data/en-ud-tweebank-train.conllu \
    --train-parser true \
    --train-postagger true \
    --train-segmentor-and-tokenizer true \
    --optimizer adam \
    --optimizer-enable-clipping true \
    --model model.twpipe \
    --max-iter 10 \
    ./data/en-ud-tweebank-train.conllu
```

## Training on Ensemble and Distillation

We found the transition-based parser training is sensitive to initialization.
To this remedy, we can train several parsers, ensemble them, then doing
knowledge distillation.

To generate training data for the distillation models, you
need first generating a set of baseline parsers (by varying
the random seed with `--dynet-seed`). Then generate the ensemble
of the parsers as:

```
./bin/generate_parse_ensemble_data \
    --models ./model1.twpipe,./model2.twpipe./model3.twpipe \
    ./data/en-ud-tweebank-train.conllu > en-ud-tweebank-train.actions
```
where different models are separated by comma.
It will dump a json formatted data into `stdout`.
And you can learn a single parser from this data using
the following commands.
```
./bin/twpipe \
    --dynet-seed 1 \
    --train \
    --heldout ./data/en-ud-tweebank-train.conllu \
    --train-distill-parser true \
    --parse-ensemble-data ./en-ud-tweebank-train.actions \
    --optimizer adam \
    --optimizer-enable-clipping true \
    --model model.twpipe \
    --max-iter 10 \
    ./data/en-ud-tweebank-train.conllu
```

## Treebank Concatenation

Our results for the NAACL 2018 paper was obtained by concatenating
the `en-ud-ewt-train.conllu` and `en-ud-tweebank-train.conllu`.
Try it please!

## TODO

* Shipping Owoputi's Tagger:
 as pointed in our paper, Owoputi's Tagger is the SOTA twitter parser.
 We would like to ship their tagger within twpipe in the future.

## References

* Olutobi Owoputi, Brendan O’Connor, Chris Dyer, Kevin Gimpel, Nathan Schneider, and Noah A. Smith. 2013. Improved part-of-speech tagging for online conversational text with word clusters. In Proc. of NAACL.
* Yijia Liu, Yi Zhu, Wanxiang Che, Bing Qin, Nathan Schneider, and Noah A. Smith. 2018. Parsing Tweets into Universal Dependency. In Proc. of NAACL.
* Yijia Liu, Wanxiang Che, Huaipeng Zhao, Bing Qin, and Ting Liu. 2018. Distilling Knowledge for Search-based Structured Prediction. In Proc. of ACL.
## License

Please see the `LICENSE.md` file.
