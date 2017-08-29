#include "noisify.h"
#include "twpipe/logging.h"
#include "twpipe/alphabet_collection.h"
#include "dynet/dynet.h"

namespace twpipe {

Noisifier::Noisifier(Corpus & c,
                     const std::string & noisify_method_name,
                     float singleton_dropout_prob) :
  corpus(c),
  singleton_dropout_prob(singleton_dropout_prob) {
  if (noisify_method_name == "none") {
    noisify_method = kNone;
  } else if (noisify_method_name == "singleton") {
    noisify_method = kSingletonDroput;
  } else if (noisify_method_name == "word") {
    noisify_method = kWordDropout;
  } else {
    _WARN << "[parse|noisifier] unknown noisify method " << noisify_method_name << ", disable noisify.";
    noisify_method = kNone;
  }
  _INFO << "[parse|noisifier] method = " << noisify_method_name;
  unk = AlphabetCollection::get()->word_map.get(Corpus::UNK);
}

void Noisifier::noisify(InputUnits & units) const {
  if (noisify_method == kNone) {
    return;
  } else if (noisify_method == kSingletonDroput) {
    for (auto& u : units) {
      auto count = corpus.counter.find(u.wid);
      if (count != corpus.counter.end() &&
          count->second == 1 &&
          dynet::rand01() < singleton_dropout_prob) {
        u.wid = unk;
      }
    }
  } else {
    for (auto& u : units) {
      auto count = corpus.counter.find(u.wid);
      float prob = 0.25 / (0.25 + (count == corpus.counter.end() ? 0. : count->second));
      if (dynet::rand01() < prob) {
        u.wid = unk;
      }
    }
  }
}

void Noisifier::denoisify(InputUnits & units) const {
  for (auto& u : units) { u.wid = u.aux_wid; }
}

}