#ifndef __TWPIPE_SEGMENTAL_RNN_TOKENIZE_MODEL_H__
#define __TWPIPE_SEGMENTAL_RNN_TOKENIZE_MODEL_H__

#include "tokenize_model.h"

namespace twpipe {

struct SegmentalRNNTokenizeModel : public TokenizeModel {
  SegmentalRNNTokenizeModel(const po::variables_map & conf);

  dynet::Expression objective(const InputUnits & input_units);
};

}

#endif  //  end for __TWPIPE_SEGMENTAL_RNN_TOKENIZE_MODEL_H__