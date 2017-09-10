#!/usr/bin/python
import argparse
import inspect
import sys
from lib import solve, print_step, fail


def solution_dna(dataset):
    counts = []
    for s in "ACGT":
        counts.append(len(filter(lambda x: x == s, dataset.strip())))
    return ''.join(counts)


def solution_rna(dataset):
    return dataset.strip().replace('T', 'U')


def solution_revc(dataset):
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    return ''.join(map(lambda x: complement[x], reversed(dataset.strip())))


def solution_fib(dataset):
    n, k = map(int, dataset.strip().split(' '))

    fib = [1, 1]
    for i in xrange(2, n):
        fib.append(fib[-1] + k * fib[-2])
    return fib[n - 1]


def solution_fibd(dataset):
    n, m = map(int, dataset.strip().split(' '))

    pop = [1, 1]
    new = [1, 0]
    deaths = [0, 0]
    for i in xrange(2, n):
        dead = (new[-m] if m <= len(pop) else 0)
        born = pop[-2] - deaths[-1]
        pop.append(pop[-1] + born - dead)
        new.append(born)
        deaths.append(dead)
    print pop
    print new
    print deaths
    return pop[n - 1]


def solution_iprb(dataset):
    k, m, n = map(float, dataset.strip().split(' '))
    t = k + m + n
    p = (k / t) + \
        (m / t) * ((k + (m - 1) * 0.75 + n * 0.5) / (t - 1)) + \
        (n / t) * ((k + m * 0.5) / (t - 1))
    return "{:.5f}".format(p)


def solution_iev(dataset):
    from random import choice
    n = map(int, dataset.strip().split(' '))
    t = float(sum(n))

    pn = [1., 1., 1., 0.75, 0.5, 0.]
    p = 0
    for i, v in enumerate(n):
        p += n[i] * pn[i] * 2

    # gen = ['AA-AA', 'AA-Aa', 'AA-aa', 'Aa-Aa', 'Aa-aa', 'aa-aa']
    # p = []
    # for i in xrange(10000):
    #     pop = []
    #     for j, v in enumerate(n):
    #         p1, p2 = gen[j].split("-")
    #         pop += [choice(p1) + choice(p2) for k in xrange(v)]
    #         pop += [choice(p1) + choice(p2) for k in xrange(v)]
    #     p.append(len(filter(lambda x: 'A' in x, pop)))
    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Solve rosalind.info problems.')
    parser.add_argument('problem', nargs=1,
                        help='The problem ID')
    parser.add_argument('--test', '-t', dest='test',
                        help='Test dataset (does not login)')
    parser.add_argument('--silent', '-s', dest='silent', action='store_true',
                        help="Don't show dataset and solution")
    args = parser.parse_args()

    funcs = {name: obj for name, obj in inspect.getmembers(sys.modules[__name__])
             if inspect.isfunction(obj) and name.startswith('solution_')}

    problem = args.problem[0]
    f_name = 'solution_' + problem
    if f_name not in funcs:
        fail("Solution not implemented yet for {}".format(problem))

    if args.test is not None:
        print funcs[f_name](args.test)
    else:
        solve(problem, funcs[f_name], silent=args.silent)
