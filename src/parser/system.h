#ifndef ABSTRACT_SYSTEM_H
#define ABSTRACT_SYSTEM_H

#include <vector>
#include "state.h"
#include "corpus.h"

struct TransitionSystem {
  const Alphabet& deprel_map;

  TransitionSystem(const Alphabet& map) : deprel_map(map) {}
  
  virtual std::string name(unsigned id) const = 0;

  virtual bool allow_nonprojective() const = 0;

  virtual unsigned num_actions() const = 0;

  virtual unsigned num_deprels() const = 0;

  virtual void get_transition_costs(const State& state,
                                    const std::vector<unsigned>& actions,
                                    const std::vector<unsigned>& ref_heads,
                                    const std::vector<unsigned>& ref_deprels,
                                    std::vector<float>& rewards) = 0;

  virtual void perform_action(State& state, const unsigned& action) = 0;

  virtual bool is_valid_action(const State& state, const unsigned& act) const = 0; 

  virtual void get_valid_actions(const State& state, std::vector<unsigned>& valid_actions) = 0;
  
  virtual void get_oracle_actions(const std::vector<unsigned>& heads,
                                  const std::vector<unsigned>& deprels,
                                  std::vector<unsigned>& actions) = 0;

  virtual unsigned get_structure_action(const unsigned & action) = 0;
};

#endif  //  end for SYSTEM_H
