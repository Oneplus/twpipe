#ifndef __TWPIPE_TOKENIZER_H__
#define __TWPIPE_TOKENIZER_H__

#include <iostream>
#include <vector>
#include "tokenize_model.h"

namespace twpipe {

struct Tokenizer {
  TokenizeModel * engine;

  Tokenizer(TokenizeModel * engine);
  
  void tokenize(const std::string & context, std::vector<std::string> & result);

  void evaluate(const std::vector<std::string> & system,
                const std::vector<std::string> & answer);
};

} //  end for twpipe

#endif  //  end for __TWPIPE_TOKENIZER_H__