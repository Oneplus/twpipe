#ifndef STATE_H
#define STATE_H

#include <vector>

struct State {
  static const unsigned MAX_N_WORDS = 1024;

  std::vector<unsigned> stack;
  std::vector<unsigned> buffer;
  std::vector<unsigned> heads;
  std::vector<unsigned> deprels;
  // std::vector<unsigned> aux;

  State(unsigned n);

  //! Computing the loss on the current state and reference.
  float loss(const std::vector<unsigned>& gold_heads,
    const std::vector<unsigned>& gold_deprels);

  bool terminated() const;
};


#endif  //  end for STATE_H