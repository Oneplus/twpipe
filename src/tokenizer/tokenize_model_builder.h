#ifndef __TWPIPE_TOKENIZE_MODEL_BUILDER_H__
#define __TWPIPE_TOKENIZE_MODEL_BUILDER_H__

#include <boost/program_options.hpp>
#include "lin_rnn_tokenize_model.h"
#include "seg_rnn_tokenize_model.h"

namespace po = boost::program_options;

namespace twpipe {

struct TokenizeModelBuilder {
  enum ModelType { 
    kLinearGRUTokenizeModel,
    kLinearLSTMTokenizeModel,
    kSegmentalGRUTokenizeModel,
    kSegmentalLSTMTokenizeModel
  };

  ModelType model_type;
  std::string model_name;
  const Alphabet & char_map;
  unsigned char_size;
  unsigned char_dim;
  unsigned hidden_dim;
  unsigned n_layers;

  TokenizeModelBuilder(po::variables_map & conf, const Alphabet & char_map);

  TokenizeModel * build(dynet::ParameterCollection & model);

  void to_json();

  TokenizeModel * from_json(dynet::ParameterCollection & model);
};

}

#endif  //  end for __TWPIPE_TOKENIZE_MODEL_BUILDER_H__
