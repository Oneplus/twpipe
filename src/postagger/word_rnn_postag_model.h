#ifndef __TWPIPE_WORD_POSTAG_MODEL_H__
#define __TWPIPE_WORD_POSTAG_MODEL_H__

#include "postag_model.h"
#include "twpipe/layer.h"
#include "twpipe/logging.h"
#include "twpipe/alphabet_collection.h"
#include "twpipe/embedding.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"

namespace twpipe {

template <class RNNBuilderType>
struct WordRNNPostagModel : public PostagModel {
  const static char* name;
  BiRNNLayer<RNNBuilderType> word_rnn;
  SymbolEmbedding word_embed;
  SymbolEmbedding pos_embed;
  InputLayer embed_input;
  Merge3Layer merge;
  DenseLayer dense;

  unsigned word_size;
  unsigned word_dim;
  unsigned word_hidden_dim;
  unsigned word_n_layers;
  unsigned pos_dim;
  unsigned root_pos_id;

  WordRNNPostagModel(dynet::ParameterCollection & model,
                     unsigned word_size,
                     unsigned word_dim,
                     unsigned embed_dim,
                     unsigned word_hidden_dim,
                     unsigned word_n_layers,
                     unsigned pos_dim) :
    PostagModel(model),
    word_rnn(model, word_n_layers, word_dim + embed_dim, word_hidden_dim),
    word_embed(model, word_size, word_dim),
    pos_embed(model, AlphabetCollection::get()->pos_map.size(), pos_dim),
    embed_input(embed_dim),
    merge(model, word_hidden_dim, word_hidden_dim, pos_dim, word_hidden_dim),
    dense(model, word_hidden_dim, AlphabetCollection::get()->pos_map.size()),
    word_dim(word_dim),
    word_hidden_dim(word_hidden_dim),
    word_n_layers(word_n_layers),
    pos_dim(pos_dim) {
    _INFO << "[postag|model] name = " << name;
    _INFO << "[postag|model] number of word types = " << word_size;
    _INFO << "[postag|model] word dimension = " << word_dim;
    _INFO << "[postag|model] pre-trained word embedding dimension = " << embed_dim;
    _INFO << "[postag|model] word rnn hidden dimension = " << word_hidden_dim;
    _INFO << "[postag|model] word rnn number layers = " << word_n_layers;
    _INFO << "[postag|model] postag hidden dimension = " << pos_dim;

    root_pos_id = AlphabetCollection::get()->pos_map.get(Corpus::ROOT);
  }

  void new_graph(dynet::ComputationGraph & cg) override {
    word_rnn.new_graph(cg);
    word_embed.new_graph(cg);
    pos_embed.new_graph(cg);
    embed_input.new_graph(cg);
    merge.new_graph(cg);
    dense.new_graph(cg);
  }

  void decode(const std::vector<std::string> & words, std::vector<std::string> & tags) override {
    Alphabet & char_map = AlphabetCollection::get()->char_map;
    Alphabet & pos_map = AlphabetCollection::get()->pos_map;

    std::vector<std::vector<float>> embeddings;
    WordEmbedding::get()->render(words, embeddings);

    unsigned n_words = words.size();
    std::vector<dynet::Expression> word_exprs(n_words);

    unsigned unk = AlphabetCollection::get()->word_map.get(Corpus::UNK);
    for (unsigned i = 0; i < n_words; ++i) {
      std::string word = words[i];
      unsigned wid = unk;
      if (AlphabetCollection::get()->word_map.contains(word)) {
        wid = AlphabetCollection::get()->word_map.get(word);
      }
      word_exprs[i] = dynet::concatenate({ word_embed.embed(wid), embed_input.get_output(embeddings[i]) });
    }

    word_rnn.add_inputs(word_exprs);
    tags.resize(n_words);
    unsigned prev_label = root_pos_id;
    for (unsigned i = 0; i < n_words; ++i) {
      auto payload = word_rnn.get_output(i);
      dynet::Expression logits = dense.get_output(dynet::rectify(
        merge.get_output(payload.first, payload.second, pos_embed.embed(prev_label))
      ));
      std::vector<float> scores = dynet::as_vector((word_embed.cg)->get_value(logits));
      unsigned label = std::max_element(scores.begin(), scores.end()) - scores.begin();

      tags[i] = pos_map.get(label);
      prev_label = label;
    }
  }

  dynet::Expression objective(const Instance & inst) override {
    // embeddings counting w/o pseudo root.
    std::vector<std::vector<float>> embeddings;
    std::vector<std::string> words;
    for (unsigned i = 1; i < inst.input_units.size(); ++i) {
      words.push_back(inst.input_units[i].word);
    }
    WordEmbedding::get()->render(words, embeddings);

    const InputUnits & input_units = inst.input_units;
    unsigned n_words_w_root = input_units.size();
    unsigned n_words = n_words_w_root - 1;
    BOOST_ASSERT_MSG(n_words == embeddings.size(), "[postag|model] number of dimension not equal.");

    std::vector<dynet::Expression> word_exprs(n_words);
    std::vector<unsigned> labels(n_words);

    for (unsigned i = 1; i < n_words_w_root; ++i) {
      word_exprs[i - 1] = dynet::concatenate({ word_embed.embed(input_units[i].wid), embed_input.get_output(embeddings[i - 1]) });
      labels[i - 1] = input_units[i].pid;
    }

    word_rnn.add_inputs(word_exprs);
    std::vector<dynet::Expression> losses(n_words);
    unsigned prev_label = root_pos_id;
    for (unsigned i = 0; i < n_words; ++i) {
      auto payload = word_rnn.get_output(i);
      dynet::Expression logits = dense.get_output(dynet::rectify(
        merge.get_output(payload.first, payload.second, pos_embed.embed(prev_label))
      ));
      losses[i] = dynet::pickneglogsoftmax(logits, labels[i]);
      prev_label = labels[i];
    }

    return dynet::sum(losses);
  }
};

typedef WordRNNPostagModel<dynet::GRUBuilder> WordGRUPostagModel;
typedef WordRNNPostagModel<dynet::CoupledLSTMBuilder> WordLSTMPostagModel;

}

#endif  //  end for __TWPIPE_CHAR_POSTAG_MODEL_H__
