#!/usr/bin/env python
from __future__ import print_function
import sys
import argparse


def main():
    cmd = argparse.ArgumentParser()
    cmd.add_argument('--system', help='the path to the system output')
    cmd.add_argument('--answer', help='the path to the output')
    cmd.add_argument('--exclude_punct', default=False, action='store_true', help='exclude punctuation.')
    cmd.add_argument('--detail', default=False, action='store_true', help='the path to the model')
    args = cmd.parse_args()

    answers = {}
    for data in open(args.answer, 'r').read().strip().split('\n\n'):
        lines = data.splitlines()
        body = [line for line in lines if not line.startswith('#')]
        words = [line.split()[1] for line in body]
        gold_postags = [line.split()[3] for line in body]
        heads = [line.split()[6] for line in body]
        deprels = [line.split()[7] for line in body]

        key = ''.join(words)
        answers[key] = {"data": data, "gold_postags": gold_postags, "heads": heads, "deprels": deprels}

    n, n_uas, n_las = 0, 0, 0
    for data in open(args.system, 'r').read().strip().split('\n\n'):
        lines = data.splitlines()
        body = [line for line in lines if not line.startswith('#')]
        words = [line.split()[1] for line in body]
        heads = [line.split()[6] for line in body]
        deprels = [line.split()[7] for line in body]

        key = ''.join(words)
        if key not in answers:
            print('key {0} not found in answer'.format(key), file=sys.stderr)
            continue

        answer = answers[key]
        gold_heads = answer["heads"]
        gold_deprels = answer['deprels']
        gold_postags = answers["gold_postags"]
        for i in range(len(words)):
            if args.exclude_punct and gold_postags[i] in ('PUCNT', ".", ",", ":", "''", "``"):
                continue
            if gold_heads[i] == heads[i]:
                n_uas += 1
                if gold_deprels[i] == deprels[i]:
                    n_las += 1
            n += 1
    print('{0}'.format(float(n_uas) / n), file=sys.stderr)
    print('{0}'.format(float(n_las) / n), file=sys.stderr)


if __name__ == "__main__":
    main()
