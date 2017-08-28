#ifndef __TWPIPE_ALPHABET_COLLECTION_H__
#define __TWPIPE_ALPHABET_COLLECTION_H__

#include "alphabet.h"

namespace twpipe {

struct AlphabetCollection {
protected:
  static AlphabetCollection * instance;
  AlphabetCollection() {}

public:
  Alphabet word_map;
  Alphabet char_map;
  Alphabet pos_map;
  Alphabet deprel_map;

  static AlphabetCollection * get();

  void stat();

  void to_json();

  void from_json();
};

}

#endif  //  end for __TWPIPE_ALPHABET_COLLECTION_H__