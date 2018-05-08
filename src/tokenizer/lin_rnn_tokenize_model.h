#ifndef __TWPIPE_LINEAR_RNN_TOKENIZE_MODEL_H__
#define __TWPIPE_LINEAR_RNN_TOKENIZE_MODEL_H__

#include <regex>
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet_layer/layer.h"
#include "twpipe/logging.h"
#include "twpipe/alphabet_collection.h"
#include "tokenize_model.h"

namespace twpipe {

struct CharactersTokenizeModel {
  void get_chars(const std::string & clean_input, std::vector<unsigned> & cids,
                 Alphabet & char_map, std::vector<std::string> * chars);

  void get_chars_and_char_categories(const std::string & clean_input,
                                     std::vector<unsigned> & cids,
                                     std::vector<unsigned> & ctids,
                                     Alphabet & char_map, std::vector<std::string> * chars);
};

struct LinearTokenizeModel : public TokenizeModel, CharactersTokenizeModel {
  const static unsigned kB;
  const static unsigned kI;
  const static unsigned kO;

  std::regex one_more_space_regex;

  LinearTokenizeModel(dynet::ParameterCollection & model);

  void get_gold_labels(const twpipe::Instance &inst,
                       const std::string &clean_input,
                       std::vector<unsigned> &labels);
};

template <class RNNBuilderType>
struct LinearRNNTokenizeModel : public LinearTokenizeModel {
  const static char* name;

  BiRNNLayer<RNNBuilderType> bi_rnn;
  SymbolEmbedding char_embed;
  SymbolEmbedding char_category_embed;
  Merge2Layer merge;
  DenseLayer dense;

  unsigned char_size;
  unsigned char_dim;
  unsigned hidden_dim;
  unsigned n_layers;

  LinearRNNTokenizeModel(dynet::ParameterCollection & model,
                         unsigned char_size,
                         unsigned char_dim,
                         unsigned hidden_dim,
                         unsigned n_layers) :
    LinearTokenizeModel(model),
    bi_rnn(model, n_layers, char_dim + 8, hidden_dim),
    char_embed(model, char_size, char_dim),
    char_category_embed(model, 64, 8),
    merge(model, hidden_dim, hidden_dim, hidden_dim),
    dense(model, hidden_dim, kO + 1),
    char_size(char_size),
    char_dim(char_dim), 
    hidden_dim(hidden_dim), 
    n_layers(n_layers) {

    // Logging stat.
    _INFO << "[tokenize|model] name = " << name;
    _INFO << "[tokenize|model] number of character types = " << char_size;
    _INFO << "[tokenize|model] character dimension = " << char_dim;
    _INFO << "[tokenize|model] hidden dimension = " << hidden_dim;
    _INFO << "[tokenize|model] number of rnn layers = " << n_layers;
  }

  void new_graph(dynet::ComputationGraph & cg) override {
    bi_rnn.new_graph(cg);
    char_embed.new_graph(cg);
    char_category_embed.new_graph(cg);
    merge.new_graph(cg);
    dense.new_graph(cg);
  }

  void decode(const std::vector<unsigned> & cids, std::vector<unsigned> & ctids, std::vector<unsigned> & output) {
    unsigned n_chars = cids.size();
    std::vector<dynet::Expression> ch_exprs(n_chars);
    for (unsigned i = 0; i < n_chars; ++i) {
      ch_exprs[i] = dynet::concatenate({char_embed.embed(cids[i]), char_category_embed.embed(ctids[i])});
    }
    bi_rnn.add_inputs(ch_exprs);
    output.resize(n_chars);
    for (unsigned i = 0; i < n_chars; ++i) {
      auto payload = bi_rnn.get_output(i);
      dynet::Expression logits = dense.get_output(dynet::rectify(merge.get_output(payload.first, payload.second)));
      std::vector<float> scores = dynet::as_vector((char_embed.cg)->get_value(logits));
      output[i] = std::max_element(scores.begin(), scores.end()) - scores.begin();
    }
  }

  void decode(const std::string & input, std::vector<std::string> & output) override {
    Alphabet & char_map = AlphabetCollection::get()->char_map;

    std::string clean_input = std::regex_replace(input, one_more_space_regex, " ");
    std::vector<unsigned> cids;
    std::vector<unsigned> ctids;
    std::vector<std::string> chars;

    get_chars_and_char_categories(clean_input, cids, ctids, char_map, &chars);
    unsigned n_chars = cids.size();
    std::vector<unsigned> labels;

    decode(cids, ctids, labels);

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

  dynet::Expression objective(const Instance & inst) override {
    Alphabet & char_map = AlphabetCollection::get()->char_map;
    std::string clean_input = std::regex_replace(inst.raw_sentence, one_more_space_regex, " ");
    std::vector<unsigned> cids;
    std::vector<unsigned> ctids;
    std::vector<unsigned> labels;

    get_chars_and_char_categories(clean_input, cids, ctids, char_map, nullptr);
    get_gold_labels(inst, clean_input, labels);

    unsigned n_chars = cids.size();
    std::vector<dynet::Expression> ch_exprs(n_chars);
    for (unsigned i = 0; i < n_chars; ++i) {
      ch_exprs[i] = dynet::concatenate({char_embed.embed(cids[i]), char_category_embed.embed(ctids[i])});
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

  dynet::Expression l2() override {
    std::vector<dynet::Expression> ret;
    for (auto & e : bi_rnn.get_params()) { ret.push_back(dynet::squared_norm(e)); }
    for (auto & e : merge.get_params()) { ret.push_back(dynet::squared_norm(e)); }
    for (auto & e : dense.get_params()) { ret.push_back(dynet::squared_norm(e)); }
    return dynet::sum(ret);
  }
};

struct LinearSentenceSegmentAndTokenizeModel : public SentenceSegmentAndTokenizeModel, CharactersTokenizeModel {
  const static unsigned kB;
  const static unsigned kB1;
  const static unsigned kI;
  const static unsigned kO;

  std::regex one_more_space_regex;

  LinearSentenceSegmentAndTokenizeModel(dynet::ParameterCollection & model);

  void get_colored(const std::vector<std::vector<unsigned>>& tree,
                   unsigned now,
                   unsigned target,
                   std::vector<unsigned> & colors);

  void get_colored(const Instance & inst, std::vector<unsigned> & colors);

  void get_gold_labels(const Instance & inst, const std::string & clean_input,
                       std::vector<unsigned> & labels);
};

template <class RNNBuilderType>
struct LinearRNNSentenceSegmentAndTokenizeModel : public LinearSentenceSegmentAndTokenizeModel {
  const static char* name;

  BiRNNLayer<RNNBuilderType> bi_rnn;
  SymbolEmbedding char_embed;
  SymbolEmbedding char_category_embed;
  Merge2Layer merge;
  DenseLayer dense;

  unsigned char_size;
  unsigned char_dim;
  unsigned hidden_dim;
  unsigned n_layers;

  LinearRNNSentenceSegmentAndTokenizeModel(dynet::ParameterCollection & model,
                                           unsigned char_size,
                                           unsigned char_dim,
                                           unsigned hidden_dim,
                                           unsigned n_layers) :
    LinearSentenceSegmentAndTokenizeModel(model),
    bi_rnn(model, n_layers, char_dim + 8, hidden_dim),
    char_embed(model, char_size, char_dim),
    char_category_embed(model, 64, 8),
    merge(model, hidden_dim, hidden_dim, hidden_dim),
    dense(model, hidden_dim, kO + 1),
    char_size(char_size),
    char_dim(char_dim),
    hidden_dim(hidden_dim),
    n_layers(n_layers) {

    // Logging stat.
    _INFO << "[tokenize|model] name = " << name;
    _INFO << "[tokenize|model] number of character types = " << char_size;
    _INFO << "[tokenize|model] character dimension = " << char_dim;
    _INFO << "[tokenize|model] hidden dimension = " << hidden_dim;
    _INFO << "[tokenize|model] number of rnn layers = " << n_layers;
  }

  void new_graph(dynet::ComputationGraph & cg) override {
    bi_rnn.new_graph(cg);
    char_embed.new_graph(cg);
    char_category_embed.new_graph(cg);
    merge.new_graph(cg);
    dense.new_graph(cg);
  }

  void decode(const std::vector<unsigned> & cids, const std::vector<unsigned> & ctids, std::vector<unsigned> & output) {
    unsigned n_chars = cids.size();
    std::vector<dynet::Expression> ch_exprs(n_chars);
    for (unsigned i = 0; i < n_chars; ++i) {
      ch_exprs[i] = dynet::concatenate({char_embed.embed(cids[i]), char_category_embed.embed(ctids[i])});
    }
    bi_rnn.add_inputs(ch_exprs);
    output.resize(n_chars);
    for (unsigned i = 0; i < n_chars; ++i) {
      auto payload = bi_rnn.get_output(i);
      dynet::Expression logits = dense.get_output(dynet::rectify(merge.get_output(payload.first, payload.second)));
      std::vector<float> scores = dynet::as_vector((char_embed.cg)->get_value(logits));
      output[i] = std::max_element(scores.begin(), scores.end()) - scores.begin();
    }
  }

  void decode(const std::string & input, std::vector<std::vector<std::string>> & output) override {
    Alphabet & char_map = AlphabetCollection::get()->char_map;

    std::string clean_input = std::regex_replace(input, one_more_space_regex, " ");
    std::vector<unsigned> cids;
    std::vector<unsigned> ctids;
    std::vector<std::string> chars;

    get_chars_and_char_categories(clean_input, cids, ctids, char_map, &chars);
    unsigned n_chars = cids.size();
    std::vector<unsigned> labels;

    decode(cids, ctids, labels);

    std::vector<std::string> sentence;
    std::string form = "";
    for (unsigned i = 0; i < n_chars; ++i) {
      if (labels[i] == kO) {
        sentence.push_back(form);
        form = "";
      } else if (labels[i] == kB1) {
        if (form != "") { sentence.push_back(form); }
        if (!sentence.empty()) {
          output.push_back(sentence);
        }
        sentence.clear();
        form = chars[i];
      } else if (labels[i] == kB) {
        if (form != "") { sentence.push_back(form); }
        form = chars[i];
      } else {
        form += chars[i];
      }
    }
    if (form != "") { sentence.push_back(form); }
    if (sentence.size() > 0) { output.push_back(sentence); }
  }

  dynet::Expression objective(const Instance & inst) override {
    Alphabet & char_map = AlphabetCollection::get()->char_map;
    std::string clean_input = std::regex_replace(inst.raw_sentence, one_more_space_regex, " ");
    std::vector<unsigned> cids;
    std::vector<unsigned> ctids;
    std::vector<unsigned> labels;

    get_chars_and_char_categories(clean_input, cids, ctids, char_map, nullptr);
    get_gold_labels(inst, clean_input, labels);

    unsigned n_chars = cids.size();
    std::vector<dynet::Expression> ch_exprs(n_chars);
    for (unsigned i = 0; i < n_chars; ++i) {
      ch_exprs[i] = dynet::concatenate({char_embed.embed(cids[i]), char_category_embed.embed(ctids[i])});
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

  dynet::Expression l2() override {
    std::vector<dynet::Expression> ret;
    for (auto & e : bi_rnn.get_params()) { ret.push_back(dynet::squared_norm(e)); }
    for (auto & e : merge.get_params()) { ret.push_back(dynet::squared_norm(e)); }
    for (auto & e : dense.get_params()) { ret.push_back(dynet::squared_norm(e)); }
    return dynet::sum(ret);
  }
};


typedef LinearRNNTokenizeModel<dynet::GRUBuilder> LinearGRUTokenizeModel;
typedef LinearRNNTokenizeModel<dynet::CoupledLSTMBuilder> LinearLSTMTokenizeModel;
typedef LinearRNNSentenceSegmentAndTokenizeModel<dynet::GRUBuilder> LinearGRUSentenceSplitAndTokenizeModel;
typedef LinearRNNSentenceSegmentAndTokenizeModel<dynet::CoupledLSTMBuilder> LinearLSTMSentenceSplitAndTokenizeModel;

}

#endif  //  end for __TWPIPE_LINEAR_RNN_TOKENIZE_MODEL_H__
