#include "alphabet_collection.h"
#include "logging.h"
#include "model.h"

namespace twpipe {

AlphabetCollection * AlphabetCollection::instance = nullptr;

AlphabetCollection * AlphabetCollection::get() {
  if (instance == nullptr) {
    instance = new AlphabetCollection();
  }
  return instance;
}

void AlphabetCollection::stat() {
  _INFO << "AlphabetCollection| # of words = " << word_map.size();
  _INFO << "AlphabetCollection| # of char = " << char_map.size();
  _INFO << "AlphabetCollection| # of pos = " << pos_map.size();
  _INFO << "AlphabetCollection| # of deprel = " << deprel_map.size();
}

void AlphabetCollection::to_json() {
  Model::get()->to_json("char-map", char_map);
  Model::get()->to_json("pos-map", pos_map);
  Model::get()->to_json("deprel-map", deprel_map);
}

void AlphabetCollection::from_json() {
  Model::get()->from_json("char-map", char_map);
  Model::get()->from_json("pos-map", pos_map);
  Model::get()->from_json("deprel-map", deprel_map);
}


}
