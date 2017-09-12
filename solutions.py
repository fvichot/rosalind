#!/usr/bin/python
import argparse
import inspect
import sys
from lib import solve, fail


def parse_fastas(lines):
    res = []
    id = None
    fasta = ''
    for line in lines:
        if line[0] == '>':
            if id is not None:
                res += [(id, fasta)]
            id = line[1:]
            fasta = ''
        elif id is not None:
            fasta += line
    if id is not None:
        res += [(id, fasta)]
    return res

CODONS = {"UUU": "F", "CUU": "L", "AUU": "I", "GUU": "V", "UUC": "F", "CUC": "L", "AUC": "I", "GUC": "V", "UUA": "L", "CUA": "L", "AUA": "I", "GUA": "V", "UUG": "L", "CUG": "L", "AUG": "M", "GUG": "V", "UCU": "S", "CCU": "P", "ACU": "T", "GCU": "A", "UCC": "S", "CCC": "P", "ACC": "T", "GCC": "A", "UCA": "S", "CCA": "P", "ACA": "T", "GCA": "A", "UCG": "S", "CCG": "P", "ACG": "T", "GCG": "A", "UAU": "Y", "CAU": "H", "AAU": "N", "GAU": "D", "UAC": "Y", "CAC": "H", "AAC": "N", "GAC": "D", "UAA": "Stop", "CAA": "Q", "AAA": "K", "GAA": "E", "UAG": "Stop", "CAG": "Q", "AAG": "K", "GAG": "E", "UGU": "C", "CGU": "R", "AGU": "S", "GGU": "G", "UGC": "C", "CGC": "R", "AGC": "S", "GGC": "G", "UGA": "Stop", "CGA": "R", "AGA": "R", "GGA": "G", "UGG": "W", "CGG": "R", "AGG": "R", "GGG": "G"}
###############################################################################


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
    return pop[n - 1]


def solution_iprb(dataset):
    k, m, n = map(float, dataset.strip().split(' '))
    t = k + m + n
    p = (k / t) + \
        (m / t) * ((k + (m - 1) * 0.75 + n * 0.5) / (t - 1)) + \
        (n / t) * ((k + m * 0.5) / (t - 1))
    return "{:.5f}".format(p)


def solution_iev(dataset):
    n = map(int, dataset.strip().split(' '))
    pn = [1., 1., 1., 0.75, 0.5, 0.]
    p = 0
    for i, v in enumerate(n):
        p += n[i] * pn[i] * 2
    return p


def solution_lia(dataset):
    from random import choice
    k, N = map(int, dataset.strip().split(' '))

    def child(p1, p2):
        c1 = choice(p1)
        c2 = choice(p2)
        if c1 > c2:  # lowercase letter first ?
            return c2 + c1
        return c1 + c2

    def brute(k, N):
        p = 0
        rounds = 100000
        for j in xrange(rounds):
            pop = ['Aa-Bb']
            for g in xrange(k):
                new_pop = []
                for i in pop:
                    f = i.split('-')
                    new_pop.append(child(f[0], 'Aa') + '-' + child(f[1], 'Bb'))
                    new_pop.append(child(f[0], 'Aa') + '-' + child(f[1], 'Bb'))
                pop = new_pop
            n = len(filter(lambda x: x == 'Aa-Bb', pop))
            if n >= N:
                p += 1
        return p / float(rounds)

    def math(k, N):
        pass

    return brute(k, N)


def solution_gc(dataset):
    fastas = parse_fastas(dataset.strip().split('\n'))
    max_gc = 0
    max_id = ''
    for id, dna in fastas:
        gc = len(filter(lambda x: x in 'GC', dna)) / float(len(dna))
        if gc > max_gc:
            max_gc, max_id = gc, id
    return "{}\n{:.6f}".format(max_id, max_gc * 100)


def solution_prot(dataset):
    rna = dataset.strip()
    prot = ''
    for i in xrange(0, len(rna), 3):
        codon = rna[i:i + 3]
        c = CODONS[codon]
        if c == 'Stop':
            break
        prot += c
    return prot


def solution_subs(dataset):
    haystack, needle = dataset.strip().split('\n')
    locations = []
    index = -1
    while True:
        index = haystack.find(needle, index + 1)
        if index == -1:
            break
        locations.append(str(index + 1))
    return ' '.join(locations)


def solution_mprt(dataset):
    import re
    import requests
    ids = dataset.strip().split('\n')
    result = []
    glyco_re = re.compile(r'N[^P][ST][^P]')
    for i in ids:
        r = requests.get('http://www.uniprot.org/uniprot/{}.fasta'.format(i))
        if r.status_code != 200:
            fail("Could not GET {} !".format(r.url))
        fasta = parse_fastas(r.text)[1]
        locations = []
        index = -1
        while True:
            m = glyco_re.search(fasta, index + 1)
            if m is None:
                break
            index = m.start()
            locations.append(str(index + 1))
        if locations != []:
            result.append(i + '\n' + ' '.join(locations))
    return '\n'.join(result)


def solution_hamm(dataset):
    return len(filter(lambda x: x[0] != x[1], zip(*dataset.strip().split('\n'))))


def solution_cons(dataset):
    lines = dataset.strip().split('\n')
    fastas = parse_fastas(lines)
    mat = {k: [0] * len(fastas[0][1]) for k in 'ACGT'}
    for f in fastas:
        for i, c in enumerate(f[1]):
            mat[c][i] += 1
    cons = ''
    for i in xrange(len(fastas[0][1])):
        max_v = 0
        cons += ' '
        for c in 'ACGT':
            if mat[c][i] > max_v:
                max_v = mat[c][i]
                cons = cons[:i] + c
    res = [cons]
    for c in 'ACGT':
        res += ['{}: {}'.format(c, ' '.join(map(str, mat[c])))]
    return '\n'.join(res)


def solution_grph(dataset):
    fastas = parse_fastas(dataset.strip().split('\n'))
    k = 3
    adj = []
    for f1 in fastas:
        for f2 in fastas:
            if f1[0] == f2[0]:
                continue
            if f1[1][-k:] == f2[1][0:k]:
                adj.append(f1[0] + " " + f2[0])
    return '\n'.join(adj)


def solution_prtm(dataset):
    s = dataset.strip()
    weights = {"A": 71.03711, "C": 103.00919, "D": 115.02694, "E": 129.04259, "F": 147.06841, "G": 57.02146, "H": 137.05891, "I": 113.08406, "K": 128.09496, "L": 113.08406, "M": 131.04049, "N": 114.04293, "P": 97.05276, "Q": 128.05858, "R": 156.10111, "S": 87.03203, "T": 101.04768, "V": 99.06841, "W": 186.07931, "Y": 163.06333}
    return sum([weights[x] for x in s])


def solution_mrna(dataset):
    p = list(dataset.strip()) + ["Stop"]
    p2len = {v: len(filter(lambda x: x == v, CODONS.values())) for v in set(CODONS.values())}
    return reduce(lambda a, b: (a * b) % 1000000, map(lambda x: p2len[x], p))


def solution_orf(dataset):
    s = parse_fastas(dataset.strip().split('\n'))[0][1]
    l = len(s)
    c = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    ss = [s[i:(l - i) - (l - i) % 3 + i] for i in xrange(3)]
    ss += [''.join([c[x] for x in y][::-1]) for y in ss]
    prots = []
    for s in ss:
        prot = [CODONS[s[i:i + 3].replace('T', 'U')] for i in xrange(0, len(s), 3)]
        print prot
        try:
            start = prot.index('M')
            stop = -1
            while stop < start:
                # print stop
                stop = prot[stop + 1:].index('Stop') + stop + 1
            prot = prot[start:stop]
        except ValueError:
            continue
        prots.append(prot)
    return '\n'.join(map(lambda x: ''.join(x), prots))

    
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
