#ifndef ARCSTD_H
#define ARCSTD_H

#include "system.h"

struct ArcStandard : public TransitionSystem {
  unsigned n_actions;
  std::vector<std::string> action_names;

  ArcStandard(const Alphabet& deprel_map);

  std::string name(unsigned id) const override;

  bool allow_nonprojective() const override;

  unsigned num_actions() const override;

  unsigned num_deprels() const override;

  void get_transition_costs(const State & state,
                            const std::vector<unsigned>& actions,
                            const std::vector<unsigned>& ref_heads,
                            const std::vector<unsigned>& ref_deprels,
                            std::vector<float>& costs) override;

  unsigned get_structure_action(const unsigned & action) override;

  unsigned cost(const State& state,
                const std::vector<unsigned>& ref_heads,
                const std::vector<unsigned>& ref_deprels) const;

  void perform_action(State & state, const unsigned& action) override;

  bool is_valid_action(const State& state, const unsigned& act) const override;

  void get_valid_actions(const State& state,
                         std::vector<unsigned>& valid_actions) override;
  
  void get_oracle_actions(const std::vector<unsigned>& heads,
                          const std::vector<unsigned>& deprels,
                          std::vector<unsigned>& actions) override;

  void shift_unsafe(State& state) const;
  void drop_unsafe(State& state) const;
  void left_unsafe(State& state, const unsigned& deprel) const;
  void right_unsafe(State& state, const unsigned& deprel) const;

  static bool is_shift(const unsigned& action);
  static bool is_drop(const unsigned& action);
  static bool is_left(const unsigned& action);
  static bool is_right(const unsigned& action);
  
  unsigned get_shift_id() const;
  unsigned get_drop_id() const;
  unsigned get_left_id(const unsigned& deprel) const;
  unsigned get_right_id(const unsigned& deprel) const;
  unsigned parse_label(const unsigned& action) const;

  void get_oracle_actions_onestep(const std::vector<unsigned>& heads,
                                  const std::vector<unsigned>& deprels,
                                  std::vector<unsigned>& sigma,
                                  unsigned& beta,
                                  std::vector<unsigned>& output,
                                  std::vector<unsigned>& actions);
};

#endif  //  end for ARCSTD_H
