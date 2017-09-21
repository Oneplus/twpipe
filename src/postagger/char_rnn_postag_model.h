#ifndef __TWPIPE_CHAR_POSTAG_MODEL_H__
#define __TWPIPE_CHAR_POSTAG_MODEL_H__

#include "postag_model.h"
#include "twpipe/logging.h"
#include "twpipe/alphabet_collection.h"
#include "twpipe/embedding.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet_layer/layer.h"

namespace twpipe {

template <class RNNBuilderType>
struct CharacterRNNPostagModel : public PostagModel {
  const static char* name;
  BiRNNLayer<RNNBuilderType> char_rnn;
  BiRNNLayer<RNNBuilderType> word_rnn;
  SymbolEmbedding char_embed;
  SymbolEmbedding pos_embed;
  InputLayer embed_input;
  DenseLayer dense1;
  DenseLayer dense2;

  unsigned char_size;
  unsigned char_dim;
  unsigned char_hidden_dim;
  unsigned char_n_layers;
  unsigned word_hidden_dim;
  unsigned word_n_layers;
  unsigned pos_dim;
  unsigned root_pos_id;

  CharacterRNNPostagModel(dynet::ParameterCollection & model,
                          unsigned char_size,
                          unsigned char_dim,
                          unsigned char_hidden_dim,
                          unsigned char_n_layers,
                          unsigned embed_dim,
                          unsigned word_hidden_dim,
                          unsigned word_n_layers,
                          unsigned pos_dim) :
    PostagModel(model),
    char_rnn(model, char_n_layers, char_dim, char_hidden_dim, false),
    word_rnn(model, word_n_layers, char_hidden_dim + char_hidden_dim + embed_dim, word_hidden_dim),
    char_embed(model, char_size, char_dim),
    pos_embed(model, AlphabetCollection::get()->pos_map.size(), pos_dim),
    embed_input(embed_dim),
    dense1(model, word_hidden_dim + word_hidden_dim + pos_dim, word_hidden_dim),
    dense2(model, word_hidden_dim, AlphabetCollection::get()->pos_map.size()),
    char_size(char_size),
    char_dim(char_dim),
    char_hidden_dim(char_hidden_dim),
    char_n_layers(char_n_layers),
    word_hidden_dim(word_hidden_dim),
    word_n_layers(word_n_layers),
    pos_dim(pos_dim) {
    _INFO << "[postag|model] name = " << name;
    _INFO << "[postag|model] number of character types = " << char_size;
    _INFO << "[postag|model] character dimension = " << char_dim;
    _INFO << "[postag|model] character rnn hidden dimension = " << char_hidden_dim;
    _INFO << "[postag|model] character rnn number layers = " << char_n_layers;
    _INFO << "[postag|model] pre-trained word embedding dimension = " << embed_dim;
    _INFO << "[postag|model] word rnn hidden dimension = " << word_hidden_dim;
    _INFO << "[postag|model] word rnn number layers = " << word_n_layers;
    _INFO << "[postag|model] postag hidden dimension = " << pos_dim;

    root_pos_id = AlphabetCollection::get()->pos_map.get(Corpus::ROOT);
  }

  void new_graph(dynet::ComputationGraph & cg) override {
    char_rnn.new_graph(cg);
    word_rnn.new_graph(cg);
    char_embed.new_graph(cg);
    pos_embed.new_graph(cg);
    embed_input.new_graph(cg);
    dense1.new_graph(cg);
    dense2.new_graph(cg);
  }

  void initialize(const std::vector<std::string> & words) override {
    Alphabet & char_map = AlphabetCollection::get()->char_map;

    std::vector<std::vector<float>> embeddings;
    WordEmbedding::get()->render(words, embeddings);

    unsigned n_words = words.size();
    std::vector<dynet::Expression> word_reprs(n_words);

    for (unsigned i = 0; i < n_words; ++i) {
      std::string word = words[i];
      unsigned len = 0;
      std::vector<unsigned> cids;
      for (unsigned j = 0; j < word.size(); j += len) {
        len = utf8_len(word[j]);
        std::string ch = word.substr(j, len);
        unsigned cid = (char_map.contains(ch) ? char_map.get(ch) : char_map.get(Corpus::UNK));
        cids.push_back(cid);
      }

      unsigned n_chars = cids.size();
      std::vector<dynet::Expression> char_exprs(n_chars);
      for (unsigned j = 0; j < n_chars; ++j) {
        char_exprs[j] = char_embed.embed(cids[j]);
      }
      char_rnn.add_inputs(char_exprs);
      auto payload = char_rnn.get_final();
      word_reprs[i] = dynet::concatenate({ payload.first, payload.second, embed_input.get_output(embeddings[i]) });
    }

    word_rnn.add_inputs(word_reprs);
  }

  dynet::Expression get_emit_score(dynet::Expression & word_repr) override {
    dynet::Expression logits = dense2.get_output(dynet::rectify(dense1.get_output(word_repr)));
    return logits;
  }

  dynet::Expression get_feature(unsigned i, unsigned prev_tag) override {
    auto payload = word_rnn.get_output(i);
    dynet::Expression feature = dynet::concatenate({ payload.first, payload.second, pos_embed.embed(prev_tag) });
    return feature;
  }

  void decode(const std::vector<std::string> & words,
              std::vector<std::string> & tags) override {
    Alphabet & pos_map = AlphabetCollection::get()->pos_map;

    unsigned n_words = words.size();
    initialize(words);

    tags.resize(n_words);
    unsigned prev_label = root_pos_id;
    std::vector<float> temp_scores;
    for (unsigned i = 0; i < n_words; ++i) {
      dynet::Expression feature = get_feature(i, prev_label);
      dynet::Expression logits = get_emit_score(feature);
      std::vector<float> scores = dynet::as_vector((char_embed.cg)->get_value(logits));
      unsigned label = std::max_element(scores.begin(), scores.end()) - scores.begin();

      tags[i] = pos_map.get(label);
      prev_label = label;
    }
  }

  dynet::Expression objective(const Instance & inst) override {
    // embeddings counting w/o pseudo root.
    unsigned n_words = inst.input_units.size() - 1;
    std::vector<std::string> words(n_words);
    std::vector<unsigned> labels(n_words);
    for (unsigned i = 1; i < inst.input_units.size(); ++i) {
      words[i - 1] = inst.input_units[i].word;
      labels[i - 1] = inst.input_units[i].pid;
    }
    initialize(words);
    std::vector<dynet::Expression> losses(n_words);
    unsigned prev_label = root_pos_id;
    for (unsigned i = 0; i < n_words; ++i) {
      dynet::Expression feature = get_feature(i, prev_label);
      dynet::Expression logits = get_emit_score(feature);
      losses[i] = dynet::pickneglogsoftmax(logits, labels[i]);
      prev_label = labels[i];
    }

    return dynet::sum(losses);
  }

  dynet::Expression l2() override {
    std::vector<dynet::Expression> ret;
    for (auto & e : char_rnn.get_params()) { ret.push_back(dynet::squared_norm(e)); }
    for (auto & e : word_rnn.get_params()) { ret.push_back(dynet::squared_norm(e)); }
    for (auto & e : dense1.get_params()) { ret.push_back(dynet::squared_norm(e)); }
    for (auto & e : dense2.get_params()) { ret.push_back(dynet::squared_norm(e)); }
    return dynet::sum(ret);
  }
};

typedef CharacterRNNPostagModel<dynet::GRUBuilder> CharacterGRUPostagModel;
typedef CharacterRNNPostagModel<dynet::CoupledLSTMBuilder> CharacterLSTMPostagModel;

}

#endif  //  end for __TWPIPE_CHAR_POSTAG_MODEL_H__
