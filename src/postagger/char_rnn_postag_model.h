#ifndef __TWPIPE_CHAR_POSTAG_MODEL_H__
#define __TWPIPE_CHAR_POSTAG_MODEL_H__

#include "postag_model.h"
#include "twpipe/layer.h"
#include "twpipe/logging.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"

namespace twpipe {

template <class RNNBuilderType>
struct CharacterRNNPostagModel : public PostagModel {
  const static char* name;
  BiRNNLayer<RNNBuilderType> char_rnn;
  BiRNNLayer<RNNBuilderType> word_rnn;
  SymbolEmbedding char_embed;
  SymbolEmbedding pos_embed;
  InputLayer embed_input;
  Merge3Layer merge_input;
  Merge3Layer merge;
  DenseLayer dense;

  unsigned char_size;
  unsigned char_dim;
  unsigned char_hidden_dim;
  unsigned char_n_layers;
  unsigned word_dim;
  unsigned word_hidden_dim;
  unsigned word_n_layers;
  unsigned pos_dim;
  unsigned root_pos_id;

  const Alphabet & char_map;

  CharacterRNNPostagModel(dynet::ParameterCollection & model,
                          unsigned char_size,
                          unsigned char_dim,
                          unsigned char_hidden_dim,
                          unsigned char_n_layers,
                          unsigned embed_dim,
                          unsigned word_dim,
                          unsigned word_hidden_dim,
                          unsigned word_n_layers,
                          unsigned pos_dim,
                          const Alphabet & char_map,
                          const Alphabet & pos_map) :
    PostagModel(model, pos_map),
    char_rnn(model, char_n_layers, char_dim, char_hidden_dim, false),
    word_rnn(model, word_n_layers, word_dim, word_hidden_dim),
    char_embed(model, char_size, char_dim),
    pos_embed(model, pos_map.size(), pos_dim),
    embed_input(embed_dim),
    merge_input(model, char_hidden_dim, char_hidden_dim, embed_dim, word_dim),
    merge(model, word_hidden_dim, word_hidden_dim, pos_dim, word_hidden_dim),
    dense(model, word_hidden_dim, pos_map.size()),
    char_size(char_size),
    char_dim(char_dim),
    char_hidden_dim(char_hidden_dim),
    char_n_layers(char_n_layers),
    word_dim(word_dim),
    word_hidden_dim(word_hidden_dim),
    word_n_layers(word_n_layers),
    pos_dim(pos_dim),
    char_map(char_map) {
    _INFO << "[postag|model] name = " << name;
    _INFO << "[postag|model] number of character types = " << char_size;
    _INFO << "[postag|model] character dimension = " << char_dim;
    _INFO << "[postag|model] character rnn hidden dimension = " << char_hidden_dim;
    _INFO << "[postag|model] character rnn number layers = " << char_n_layers;
    _INFO << "[postag|model] pre-trained word embedding dimension = " << embed_dim;
    _INFO << "[postag|model] word dimension = " << word_dim;
    _INFO << "[postag|model] word rnn hidden dimension = " << word_hidden_dim;
    _INFO << "[postag|model] word rnn number layers = " << word_n_layers;
    _INFO << "[postag|model] postag hidden dimension = " << pos_dim;

    root_pos_id = pos_map.get(Corpus::ROOT);
  }

  void new_graph(dynet::ComputationGraph & cg) {
    char_rnn.new_graph(cg);
    word_rnn.new_graph(cg);
    char_embed.new_graph(cg);
    pos_embed.new_graph(cg);
    embed_input.new_graph(cg);
    merge_input.new_graph(cg);
    merge.new_graph(cg);
    dense.new_graph(cg);
  }

  void decode(const std::vector<std::string> & words,
              const std::vector<std::vector<float>> & embeddings,
              std::vector<std::string> & tags) {
    unsigned n_words = words.size();
    std::vector<dynet::Expression> word_exprs(n_words);

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
      word_exprs[i] = merge_input.get_output(payload.first,
                                             payload.second,
                                             embed_input.get_output(embeddings[i]));
    }

    word_rnn.add_inputs(word_exprs);
    tags.resize(n_words);
    unsigned prev_label = root_pos_id;
    for (unsigned i = 0; i < n_words; ++i) {
      auto payload = word_rnn.get_output(i);
      dynet::Expression logits = dense.get_output(dynet::rectify(
        merge.get_output(payload.first, payload.second, pos_embed.embed(prev_label))
      ));
      std::vector<float> scores = dynet::as_vector((char_embed.cg)->get_value(logits));
      unsigned label = std::max_element(scores.begin(), scores.end()) - scores.begin();

      tags[i] = pos_map.get(label);
      prev_label = label;
    }
  }

  dynet::Expression objective(const Instance & inst,
                              const std::vector<std::vector<float>> & embeddings) override {
    const InputUnits & input_units = inst.input_units;
    unsigned n_words_w_root = input_units.size();
    unsigned n_words = n_words_w_root - 1;
    BOOST_ASSERT_MSG(n_words == embeddings.size(), "[postag|model] number of dimension not equal.");

    std::vector<dynet::Expression> word_exprs(n_words);
    std::vector<unsigned> labels(n_words);

    for (unsigned i = 1; i < n_words_w_root; ++i) {
      unsigned n_chars = input_units[i].cids.size();
      std::vector<dynet::Expression> char_exprs(n_chars);
      for (unsigned j = 0; j < n_chars; ++j) {
        char_exprs[j] = char_embed.embed(input_units[i].cids[j]);
      }
      char_rnn.add_inputs(char_exprs);
      auto payload = char_rnn.get_final();
      word_exprs[i - 1] = merge_input.get_output(payload.first,
                                                 payload.second,
                                                 embed_input.get_output(embeddings[i - 1]));
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

typedef CharacterRNNPostagModel<dynet::GRUBuilder> CharacterGRUPostagModel;
typedef CharacterRNNPostagModel<dynet::LSTMBuilder> CharacterLSTMPostagModel;

}

#endif  //  end for __TWPIPE_CHAR_POSTAG_MODEL_H__
