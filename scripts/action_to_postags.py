#!/usr/bin/env python
from __future__ import print_function
import argparse
import json


def main():
    cmd = argparse.ArgumentParser()
    cmd.add_argument('--actions', help='the path to actions.')
    cmd.add_argument('--model', help='the path to the model.')
    cmd.add_argument('--conll', help='the path to conll.')
    opts = cmd.parse_args()

    model = json.load(open(opts.model, 'r'))
    mapping = {}
    pos_map = model['general']['pos-map']
    for name in pos_map:
        i = pos_map[name]
        mapping[int(i)] = name.encode('utf-8')

    dataset = {}
    for n, data in enumerate(open(opts.conll, 'r').read().strip().split('\n\n')):
        dataset[n] = data

    for line in open(opts.actions, 'r'):
        line = line.strip()
        fields = json.loads(line)
        i = fields['id']
        postags = fields['category']
        lines = dataset[i].splitlines()
        body = [line for line in lines if not line.startswith('#')]
        for i, line in enumerate(body):
            fields = line.split()
            fields[3] = mapping[postags[i]]
            print('\t'.join(fields))
        print()


if __name__ == "__main__":
    main()
