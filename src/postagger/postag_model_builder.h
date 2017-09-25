#ifndef __TWPIPE_POSTAG_MODEL_BUILDER_H__
#define __TWPIPE_POSTAG_MODEL_BUILDER_H__

#include <boost/program_options.hpp>
#include "postag_model.h"

namespace po = boost::program_options;

namespace twpipe {

struct PostagModelBuilder {
  enum ModelType {
    kCharacterGRUPostagModel,
    kCharacterLSTMPostagModel,
    kCharacterGRUPostagCRFModel,
    kCharacterLSTMPostagCRFModel,
    kCharacterCNNGRUPostagModel,
    kCharacterCNNLSTMPostagModel,
    kCharacterClusterGRUPostagModel,
    kCharacterClusterLSTMPostagModel,
    kWordGRUPostagModel,
    kWordLSTMPostagModel,
    kWordCharacterGRUPostagModel,
    kWordCharacterLSTMPostagModel
  };

  ModelType model_type;
  std::string model_name;
  unsigned word_size;
  unsigned word_dim;
  unsigned char_size;
  unsigned char_dim;
  unsigned char_hidden_dim;
  unsigned char_n_layers;
  unsigned word_hidden_dim;
  unsigned word_n_layers;
  unsigned cluster_dim;
  unsigned cluster_n_layers;
  unsigned cluster_hidden_dim;
  unsigned pos_size;
  unsigned pos_dim;
  unsigned embed_dim;

  PostagModelBuilder(po::variables_map & conf);

  PostagModel * build(dynet::ParameterCollection & model);

  void to_json();

  PostagModel * from_json(dynet::ParameterCollection & model);

  ModelType get_model_type(const std::string & model_name);
};

}

#endif // !__TWPIPE_POSTAG_MODEL_BUILDER_H__
