#ifndef __TWPIPE_POSTAG_MODEL_BUILDER_H__
#define __TWPIPE_POSTAG_MODEL_BUILDER_H__

#include <boost/program_options.hpp>
#include "char_rnn_postag_model.h"
#include "char_rnn_crf_postag_model.h"
#include "char_rnn_wcluster_postag_model.h"

namespace po = boost::program_options;

namespace twpipe {

struct PostagModelBuilder {
  enum ModelType {
    kCharacterGRUPostagModel,
    kCharacterLSTMPostagModel,
    kCharacterGRUPostagCRFModel,
    kCharacterLSTMPostagCRFModel,
    kCharacterClusterGRUPostagModel,
    kCharacterClusterLSTMPostagModel
  };

  ModelType model_type;
  std::string model_name;
  const Alphabet & char_map;
  const Alphabet & pos_map;
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

  PostagModelBuilder(po::variables_map & conf, 
                     const Alphabet & char_map,
                     const Alphabet & pos_map);

  PostagModel * build(dynet::ParameterCollection & model);

  void to_json();

  PostagModel * from_json(dynet::ParameterCollection & model);
};

}

#endif // !__TWPIPE_POSTAG_MODEL_BUILDER_H__
