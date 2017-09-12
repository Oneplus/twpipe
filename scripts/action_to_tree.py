#!/usr/bin/env python
from __future__ import print_function
import sys
import argparse
import json


class State(object):
    def __init__(self, n):
        # n counts the pseudo root.
        self.n_ = n
        self.buffer_ = range(n)
        self.stack_ = []
        self.heads_ = [-1 for _ in range(self.n_)]
        self.deprels_ = [None for _ in range(self.n_)]

    def terminate(self):
        return len(self.stack_) == 1 and len(self.buffer_) == 0


class Parser(object):
    def __init__(self, deprel_map):
        self.deprel_map_ = deprel_map

    def perform(self, state, action):
        raise NotImplementedError()


class ArcStandard(Parser):
    def __init__(self, deprel_map):
        super(ArcStandard, self).__init__(deprel_map)

    def perform(self, state, action):
        if action == 0:
            self._shift(state)
        elif action % 2 == 1:
            self._left(state, self.deprel_map_[(action - 1) / 2])
        else:
            self._right(state, self.deprel_map_[(action - 2) / 2])

    @classmethod
    def _shift(cls, state):
        assert len(state.buffer_) > 0
        i = state.buffer_[0]
        state.stack_.append(i)
        state.buffer_ = state.buffer_[1:]

    @classmethod
    def _left(cls, state, deprel):
        assert len(state.stack_) >= 2
        hed, mod = state.stack_[-1], state.stack_[-2]
        state.stack_ = state.stack_[:-1]
        state.stack_[-1] = hed
        state.heads_[mod] = hed
        state.deprels_[mod] = deprel

    @classmethod
    def _right(cls, state, deprel):
        assert len(state.stack_) >= 2
        hed, mod = state.stack_[-2], state.stack_[-1]
        state.stack_ = state.stack_[:-1]
        state.stack_[-1] = hed
        state.heads_[mod] = hed
        state.deprels_[mod] = deprel


class ArcHybrid(Parser):
    def __init__(self, deprel_map):
        super(ArcHybrid, self).__init__(deprel_map)

    def perform(self, state, action):
        if action == 0:
            self._shift(state)
        elif action % 2 == 1:
            self._left(state, self.deprel_map_[(action - 1) / 2])
        else:
            self._right(state, self.deprel_map_[(action - 2) / 2])

    @classmethod
    def _shift(cls, state):
        assert len(state.buffer_) > 0
        i = state.buffer_[0]
        state.stack_.append(i)
        state.buffer_ = state.buffer_[1:]

    @classmethod
    def _left(cls, state, deprel):
        assert len(state.stack_) >= 1 and len(state.buffer_) >= 1
        hed, mod = state.buffer_[0], state.stack_[-1]
        state.stack_ = state.stack_[:-1]
        state.heads_[mod] = hed
        state.deprels_[mod] = deprel

    @classmethod
    def _right(cls, state, deprel):
        assert len(state.stack_) >= 2
        hed, mod = state.stack_[-2], state.stack_[-1]
        state.stack_ = state.stack_[:-1]
        state.heads_[mod] = hed
        state.deprels_[mod] = deprel


def main():
    cmd = argparse.ArgumentParser()
    cmd.add_argument('--actions', help='the path to actions.')
    cmd.add_argument('--model', help='the path to the model.')
    cmd.add_argument('--conll', help='the path to conll.')
    opts = cmd.parse_args()

    model = json.load(open(opts.model, 'r'))
    mapping = {}
    deprel_map = model['general']['deprel-map']
    for name in deprel_map:
        i = deprel_map[name]
        mapping[int(i)] = name.encode('utf-8')

    system = model['parser']['config']['system']
    assert system in ('archybrid', 'arcstd')

    if system == 'archybrid':
        parser = ArcHybrid(mapping)
    elif system == 'arcstd':
        parser = ArcStandard(mapping)
    else:
        raise ValueError()

    dataset = {}
    for n, data in enumerate(open(opts.conll, 'r').read().strip().split('\n\n')):
        dataset[n] = data

    for line in open(opts.actions, 'r'):
        line = line.strip()
        fields = json.loads(line)
        i = fields['id']
        actions = fields['action']
        lines = dataset[i].splitlines()
        body = [line for line in lines if not line.startswith('#')]
        n = len(body)
        state = State(n + 1)
        for action in actions:
            assert not state.terminate()
            parser.perform(state, action)
        for i, line in enumerate(body):
            fields = line.split()
            fields[6] = str(state.heads_[i + 1])
            fields[7] = str(state.deprels_[i + 1])
            print('\t'.join(fields))
        print()

if __name__ == "__main__":
    main()
