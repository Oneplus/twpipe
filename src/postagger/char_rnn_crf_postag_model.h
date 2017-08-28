#ifndef __TWPIPE_CHAR_POSTAG_CRF_MODEL_H__
#define __TWPIPE_CHAR_POSTAG_CRF_MODEL_H__

#include "postag_model.h"
#include "twpipe/layer.h"
#include "twpipe/logging.h"
#include "twpipe/embedding.h"
#include "twpipe/alphabet_collection.h"
#include "twpipe/corpus.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"

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
    merge(model, word_hidden_dim, word_hidden_dim, pos_dim, word_hidden_dim),
    dense(model, word_hidden_dim, 1),
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

  void new_graph(dynet::ComputationGraph & cg) {
    char_rnn.new_graph(cg);
    word_rnn.new_graph(cg);
    char_embed.new_graph(cg);
    pos_embed.new_graph(cg);
    tran_embed.new_graph(cg);
    embed_input.new_graph(cg);
    merge.new_graph(cg);
    dense.new_graph(cg);
  }

  void decode(const std::vector<std::string> & words, std::vector<std::string> & tags) {
    Alphabet & char_map = AlphabetCollection::get()->char_map;
    Alphabet & pos_map = AlphabetCollection::get()->pos_map;

    std::vector<std::vector<float>> embeddings;
    WordEmbedding::get()->render(words, embeddings);

    unsigned n_words = words.size();
    std::vector<std::vector<float>> emit_matrix(n_words, std::vector<float>(pos_size));
    std::vector<std::vector<float>> tran_matrix(pos_size, std::vector<float>(pos_size));
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
      word_exprs[i] = dynet::concatenate({ payload.first, payload.second, embed_input.get_output(embeddings[i]) });
    }
    word_rnn.add_inputs(word_exprs);

    std::vector<dynet::Expression> uni_labels(pos_size);
    for (unsigned t = 0; t < pos_size; ++t) {
      uni_labels[t] = pos_embed.embed(t);
      for (unsigned pt = 0; pt < pos_size; ++pt) {
        tran_matrix[pt][t] = dynet::as_scalar(
          char_embed.cg->get_value(tran_embed.embed(pt * pos_size + t))
        );
      }
    }

    for (unsigned i = 0; i < n_words; ++i) {
      auto payload = word_rnn.get_output(i);
      for (unsigned t = 0; t < pos_size; ++t) {
        emit_matrix[i][t] = dynet::as_scalar(char_embed.cg->get_value(
          dense.get_output(dynet::rectify(merge.get_output(payload.first, payload.second, pos_embed.embed(t))))
        ));
      }
    }

    std::vector<std::vector<float>> alpha(n_words, std::vector<float>(pos_size));
    std::vector<std::vector<unsigned>> path(n_words, std::vector<unsigned>(pos_size));

    for (unsigned i = 0; i < n_words; ++i) {
      for (unsigned t = 0; t < pos_size; ++t) {
        if (i == 0) {
          alpha[i][t] = emit_matrix[i][t];
          path[i][t] = root_pos_id;
          continue;
        }

        for (unsigned pt = 0; pt < pos_size; ++pt) {
          float score = alpha[i - 1][pt] + emit_matrix[i][t] + tran_matrix[pt][t];
          if (pt == 0 || score > alpha[i][t]) {
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
      unsigned n_chars = input_units[i].cids.size();
      std::vector<dynet::Expression> char_exprs(n_chars);
      for (unsigned j = 0; j < n_chars; ++j) {
        char_exprs[j] = char_embed.embed(input_units[i].cids[j]);
      }
      char_rnn.add_inputs(char_exprs);
      auto payload = char_rnn.get_final();
      word_exprs[i - 1] = dynet::concatenate({ payload.first, payload.second, embed_input.get_output(embeddings[i - 1]) });
      labels[i - 1] = input_units[i].pid;
    }

    word_rnn.add_inputs(word_exprs);
    std::vector<BiRNNOutput> payloads;
    word_rnn.get_outputs(payloads);

    std::vector<ExpressionRow> emit_matrix(n_words, ExpressionRow(pos_size));
    std::vector<ExpressionRow> tran_matrix(pos_size, ExpressionRow(pos_size));
    std::vector<dynet::Expression> uni_labels(pos_size);
    for (unsigned t = 0; t < pos_size; ++t) {
      uni_labels[t] = pos_embed.embed(t);
      for (unsigned pt = 0; pt < pos_size; ++pt) {
        tran_matrix[pt][t] = tran_embed.embed(pt * pos_size + t);
      }
    }

    for (unsigned i = 0; i < n_words; ++i) {
      for (unsigned t = 0; t < pos_size; ++t) {
        emit_matrix[i][t] = dense.get_output(
          dynet::rectify(merge.get_output(payloads[i].first, payloads[i].second, uni_labels[t]))
        );
      }
    }

    std::vector<ExpressionRow> alpha(n_words, ExpressionRow(pos_size));
    std::vector<dynet::Expression> path(n_words);

    for (unsigned i = 0; i < n_words; ++i) {
      for (unsigned t = 0; t < pos_size; ++t) {
        std::vector<dynet::Expression> f;
        if (i == 0) {
          f.push_back(emit_matrix[i][t]);
          if (t == labels[i]) {
            path[i] = emit_matrix[i][t];
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
};

typedef CharacterRNNCRFPostagModel<dynet::GRUBuilder> CharacterGRUCRFPostagModel;
typedef CharacterRNNCRFPostagModel<dynet::CoupledLSTMBuilder> CharacterLSTMCRFPostagModel;

}

#endif  //  end for __TWPIPE_CHAR_POSTAG_CRF_MODEL_H__
