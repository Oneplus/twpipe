#ifndef __TWPIPE_LINEAR_RNN_TOKENIZE_MODEL_H__
#define __TWPIPE_LINEAR_RNN_TOKENIZE_MODEL_H__

#include <regex>
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "twpipe/layer.h"
#include "twpipe/logging.h"
#include "twpipe/alphabet_collection.h"
#include "tokenize_model.h"

namespace twpipe {

template <class RNNBuilderType>
struct LinearRNNTokenizeModel : public TokenizeModel {
  const static unsigned kB;
  const static unsigned kI;
  const static unsigned kO;
  const static char* name;

  BiRNNLayer<RNNBuilderType> bi_rnn;
  SymbolEmbedding char_embed;
  Merge2Layer merge;
  DenseLayer dense;

  unsigned char_size;
  unsigned char_dim;
  unsigned hidden_dim;
  unsigned n_layers;

  std::regex one_more_space_regex;

  LinearRNNTokenizeModel(dynet::ParameterCollection & model,
                         unsigned char_size,
                         unsigned char_dim,
                         unsigned hidden_dim,
                         unsigned n_layers) :
    TokenizeModel(model),
    bi_rnn(model, n_layers, char_dim, hidden_dim),
    char_embed(model, char_size, char_dim),
    merge(model, hidden_dim, hidden_dim, hidden_dim),
    dense(model, hidden_dim, 3),
    char_size(char_size),
    char_dim(char_dim), 
    hidden_dim(hidden_dim), 
    n_layers(n_layers),
    one_more_space_regex("[ ]{2,}") {

    // Logging stat.
    _INFO << "[tokenize|model] name = " << name;
    _INFO << "[tokenize|model] number of character types = " << char_size;
    _INFO << "[tokenize|model] character dimension = " << char_dim;
    _INFO << "[tokenize|model] hidden dimension = " << hidden_dim;
    _INFO << "[tokenize|model] number of rnn layers = " << n_layers;
  }

  void new_graph(dynet::ComputationGraph & cg) {
    bi_rnn.new_graph(cg);
    char_embed.new_graph(cg);
    merge.new_graph(cg);
    dense.new_graph(cg);
  }

  void decode(const std::string & input, std::vector<std::string> & output) {
    // First, replace multiple space with one space.
    Alphabet & char_map = AlphabetCollection::get()->char_map;

    std::string clean_input = std::regex_replace(input, one_more_space_regex, " ");
    std::vector<unsigned> cids;
    std::vector<std::string> chars;
    
    unsigned len = 0;
    for (unsigned i = 0; i < clean_input.size(); i += len) {
      len = utf8_len(clean_input[i]);
      std::string ch = clean_input.substr(i, len);
      chars.push_back(ch);
      unsigned cid = (char_map.contains(ch) ? char_map.get(ch) : char_map.get(Corpus::UNK));
      cids.push_back(cid);
    }
  
    unsigned n_chars = cids.size();
    std::vector<dynet::Expression> ch_exprs(n_chars);
    for (unsigned i = 0; i < n_chars; ++i) {
      ch_exprs[i] = char_embed.embed(cids[i]);
    }
    bi_rnn.add_inputs(ch_exprs);
    std::vector<unsigned> labels(n_chars);
    for (unsigned i = 0; i < n_chars; ++i) {
      auto payload = bi_rnn.get_output(i);
      dynet::Expression logits = dense.get_output(dynet::rectify(merge.get_output(payload.first, payload.second)));
      std::vector<float> scores = dynet::as_vector((char_embed.cg)->get_value(logits));
      labels[i] = std::max_element(scores.begin(), scores.end()) - scores.begin();
    }

    std::string form = "";
    for (unsigned i = 0; i < n_chars; ++i) {
      if (labels[i] == kO) {
        output.push_back(form);
        form = "";
      } else if (labels[i] == kB) {
        if (form != "") { output.push_back(form); }
        form = chars[i];
      } else {
        form += chars[i];
      }
    }
    if (form != "") { output.push_back(form); }
  }

  dynet::Expression objective(const Instance & inst) {
    Alphabet & char_map = AlphabetCollection::get()->char_map;
    const InputUnits & input_units = inst.input_units;
    std::string clean_input = std::regex_replace(inst.raw_sentence, one_more_space_regex, " ");
    std::vector<unsigned> cids;
    std::vector<unsigned> labels;

    unsigned len = 0;
    unsigned j = 1, k = 0; // j start from 1 because the first one is dummy root.
    for (unsigned i = 0; i < clean_input.size(); i += len) {
      len = utf8_len(clean_input[i]);
      unsigned cid = char_map.get(clean_input.substr(i, len));
      cids.push_back(cid);
      labels.push_back(cid == space_cid ? kO : (k == 0 ? kB : kI));
      if (cid != space_cid) {
        ++k;
        if (k == input_units[j].cids.size()) { k = 0; ++j; }
      }
    }

    unsigned n_chars = cids.size();
    std::vector<dynet::Expression> ch_exprs(n_chars);
    for (unsigned i = 0; i < n_chars; ++i) {
      ch_exprs[i] = char_embed.embed(cids[i]);
    }
    bi_rnn.add_inputs(ch_exprs);
    std::vector<dynet::Expression> losses(n_chars);
    for (unsigned i = 0; i < n_chars; ++i) {
      auto payload = bi_rnn.get_output(i);
      dynet::Expression logits = dense.get_output(dynet::rectify(merge.get_output(payload.first, payload.second)));
      losses[i] = dynet::pickneglogsoftmax(logits, labels[i]);
    }
    return dynet::sum(losses);
  }
};

typedef LinearRNNTokenizeModel<dynet::GRUBuilder> LinearGRUTokenizeModel;
typedef LinearRNNTokenizeModel<dynet::CoupledLSTMBuilder> LinearLSTMTokenizeModel;

}

#endif  //  end for __TWPIPE_LINEAR_RNN_TOKENIZE_MODEL_H__
