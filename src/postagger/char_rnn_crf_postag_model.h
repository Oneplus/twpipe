#ifndef __TWPIPE_CHAR_POSTAG_CRF_MODEL_H__
#define __TWPIPE_CHAR_POSTAG_CRF_MODEL_H__

#include "postag_model.h"
#include "twpipe/logging.h"
#include "twpipe/embedding.h"
#include "twpipe/alphabet_collection.h"
#include "twpipe/corpus.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet_layer/layer.h"

namespace twpipe {

template <class RNNBuilderType>
struct CharacterRNNCRFPostagModel : public PostagModel {
  typedef std::vector<dynet::Expression> ExpressionRow;
  const static char* name;
  BiRNNLayer<RNNBuilderType> char_rnn;
  BiRNNLayer<RNNBuilderType> word_rnn;
  SymbolEmbedding char_embed;
  SymbolEmbedding pos_embed;
  SymbolEmbedding tran_embed;
  InputLayer embed_input;
  DenseLayer dense1;
  DenseLayer dense2;

  unsigned char_size;
  unsigned char_dim;
  unsigned char_hidden_dim;
  unsigned char_n_layers;
  unsigned word_dim;
  unsigned word_hidden_dim;
  unsigned word_n_layers;
  unsigned pos_dim;
  unsigned root_pos_id;

  CharacterRNNCRFPostagModel(dynet::ParameterCollection & model,
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
    tran_embed(model, AlphabetCollection::get()->pos_map.size() * AlphabetCollection::get()->pos_map.size(), 1),
    embed_input(embed_dim),
    dense1(model, word_hidden_dim + word_hidden_dim + pos_dim, word_hidden_dim),
    dense2(model, word_hidden_dim, 1),
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
    tran_embed.new_graph(cg);
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

  void decode(const std::vector<std::string> & words, std::vector<std::string> & tags) override {
    Alphabet & pos_map = AlphabetCollection::get()->pos_map;

    unsigned n_words = words.size();
    initialize(words);
    std::vector<dynet::Expression> losses(n_words);
    std::vector<std::vector<float>> emit_matrix(n_words, std::vector<float>(pos_size));
    std::vector<std::vector<float>> tran_matrix(pos_size, std::vector<float>(pos_size));
    for (unsigned t = 0; t < pos_size; ++t) {
      for (unsigned pt = 0; pt < pos_size; ++pt) {
        dynet::Expression tran_score = tran_embed.embed(pt * pos_size + t);
        tran_matrix[pt][t] = dynet::as_scalar(tran_score.value());
      }
    }

    for (unsigned i = 0; i < n_words; ++i) {
      for (unsigned t = 0; t < pos_size; ++t) {
        dynet::Expression feature = get_feature(i, t);
        dynet::Expression emit_score = get_emit_score(feature);
        emit_matrix[i][t] = dynet::as_scalar(emit_score.value());
      }
    }

    std::vector<std::vector<float>> alpha(n_words, std::vector<float>(pos_size, -1e10f));
    std::vector<std::vector<unsigned>> path(n_words, std::vector<unsigned>(pos_size, root_pos_id));

    for (unsigned i = 0; i < n_words; ++i) {
      for (unsigned t = 0; t < pos_size; ++t) {
        if (i == 0) {
          alpha[i][t] = emit_matrix[i][t] + tran_matrix[root_pos_id][t];
          path[i][t] = root_pos_id;
          continue;
        }
        alpha[i][t] = alpha[i - 1][0] + emit_matrix[i][t] + tran_matrix[0][t];
        for (unsigned pt = 1; pt < pos_size; ++pt) {
          float score = alpha[i - 1][pt] + emit_matrix[i][t] + tran_matrix[pt][t];
          if (score > alpha[i][t]) {
            alpha[i][t] = score;
            path[i][t] = pt;
          }
        }
      }
    }
    unsigned best = 0;
    float best_score = alpha[n_words - 1][0];
    for (unsigned t = 1; t < pos_size; ++t) {
      if (best_score < alpha[n_words - 1][t]) { best = t; best_score = alpha[n_words - 1][t]; }
    }
    tags.clear();
    tags.push_back(pos_map.get(best));
    for (unsigned i = n_words - 1; i > 0; --i) {
      best = path[i][best];
      tags.push_back(pos_map.get(best));
    }
    std::reverse(tags.begin(), tags.end());
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

    std::vector<ExpressionRow> emit_matrix(n_words, ExpressionRow(pos_size));
    std::vector<ExpressionRow> tran_matrix(pos_size, ExpressionRow(pos_size));
    for (unsigned t = 0; t < pos_size; ++t) {
      for (unsigned pt = 0; pt < pos_size; ++pt) {
        tran_matrix[pt][t] = tran_embed.embed(pt * pos_size + t);
      }
    }

    for (unsigned i = 0; i < n_words; ++i) {
      for (unsigned t = 0; t < pos_size; ++t) {
        dynet::Expression feature = get_feature(i, t);
        emit_matrix[i][t] = get_emit_score(feature);
      }
    }

    std::vector<ExpressionRow> alpha(n_words, ExpressionRow(pos_size));
    std::vector<dynet::Expression> path(n_words);

    for (unsigned i = 0; i < n_words; ++i) {
      for (unsigned t = 0; t < pos_size; ++t) {
        std::vector<dynet::Expression> f;
        if (i == 0) {
          f.push_back(emit_matrix[i][t] + tran_matrix[root_pos_id][t]);
          if (t == labels[i]) {
            path[i] = emit_matrix[i][t] + tran_matrix[root_pos_id][t];
          }
        } else {
          for (unsigned pt = 0; pt < pos_size; ++pt) {
            f.push_back(alpha[i - 1][pt] + emit_matrix[i][t] + tran_matrix[pt][t]);
            if (pt == labels[i - 1] && t == labels[i]) {
              path[i] = path[i - 1] + emit_matrix[i][t] + tran_matrix[pt][t];
            }
          }
        }
        alpha[i][t] = dynet::logsumexp(f);
      }
    }

    std::vector<dynet::Expression> f;
    for (unsigned t = 0; t < pos_size; ++t) {
      f.push_back(alpha[n_words - 1][t]);
    }
    return dynet::logsumexp(f) - path.back();
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

typedef CharacterRNNCRFPostagModel<dynet::GRUBuilder> CharacterGRUCRFPostagModel;
typedef CharacterRNNCRFPostagModel<dynet::CoupledLSTMBuilder> CharacterLSTMCRFPostagModel;

}

#endif  //  end for __TWPIPE_CHAR_POSTAG_CRF_MODEL_H__
