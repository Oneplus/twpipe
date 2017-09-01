#include "system.h"
#include "twpipe/alphabet_collection.h"

namespace twpipe {

unsigned TransitionSystem::num_deprels() {
  return AlphabetCollection::get()->deprel_map.size();
}

}
