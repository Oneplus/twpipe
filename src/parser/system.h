#ifndef __TWPIPE_PARSER_SYSTEM_H__
#define __TWPIPE_PARSER_SYSTEM_H__

#include <vector>
#include "state.h"
#include "twpipe/alphabet.h"

namespace twpipe {

struct TransitionSystem {
  TransitionSystem() {}

  /// Get the name of transition system.
  virtual std::string name() const = 0;

  /// Get the name of action.
  virtual std::string name(unsigned id) const = 0;

  virtual bool allow_nonprojective() const = 0;

  virtual unsigned num_actions() const = 0;

  unsigned num_deprels();

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

}

#endif  //  end for SYSTEM_H
