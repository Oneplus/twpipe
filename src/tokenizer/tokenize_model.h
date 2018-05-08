#ifndef __TOKENIZER_TOKENIZE_MODEL_H__
#define __TOKENIZER_TOKENIZE_MODEL_H__

#include <boost/program_options.hpp>
#include <tuple>
#include "twpipe/corpus.h"
#include "twpipe/model.h"
#include "dynet/expr.h"

namespace po = boost::program_options;

namespace twpipe {

struct AbstractTokenizeModel {
  static po::options_description get_options();

  dynet::ParameterCollection & model;
  unsigned space_cid;

  AbstractTokenizeModel(dynet::ParameterCollection & model);

  virtual void new_graph(dynet::ComputationGraph & cg) = 0;

  virtual dynet::Expression objective(const Instance & inst) = 0;

  virtual dynet::Expression l2() = 0;

  virtual std::tuple<float, float, float> evaluate(const Instance & inst) = 0;

  std::tuple<float, float, float> fscore(const std::vector<std::string> & gold,
                                         const std::vector<std::string> & prediction);
};

struct TokenizeModel : public AbstractTokenizeModel {
  TokenizeModel(dynet::ParameterCollection & model);

  virtual void decode(const std::string & input, std::vector<std::string> & result) = 0;

  void tokenize(const std::string & input);

  void tokenize(const std::string & input, std::vector<std::string> & result);

  std::tuple<float, float, float> evaluate(const Instance & inst) override;
};

struct SentenceSegmentAndTokenizeModel : public AbstractTokenizeModel {
  SentenceSegmentAndTokenizeModel(dynet::ParameterCollection & model);

  virtual void decode(const std::string & input, std::vector<std::vector<std::string>> & result) = 0;

  void sentsegment_and_tokenize(const std::string &input);

  void sentsegment_and_tokenize(const std::string &input, std::vector<std::vector<std::string>> &result);

  std::tuple<float, float, float> evaluate(const Instance & inst) override;
};

}

#endif  //  end for __TOKENIZER_TOKENIZE_MODEL_H__