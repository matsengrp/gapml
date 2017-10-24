#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Given an outputfile from PHYLIP MIX, produces an ETE tree
"""
from __future__ import print_function
from ete3 import Tree
import re, random
from collections import defaultdict

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Data.IUPACData import ambiguous_dna_values


# iterate over recognized sections in the phylip output file.
def sections(fh):
    patterns = {
        ('sequences', 'mix'): "\s*\(\s*.\s*means\s+same",
        "seqboot_dataset": "Data\s*set"
    }
    patterns = {k: re.compile(v, re.IGNORECASE) for (k, v) in patterns.items()}
    for line in fh:
        for k, pat in patterns.items():
            if pat.match(line):
                yield k
                break


# iterate over entries in the sequences section
def parse_seqdict(fh, mode='mix'):
    #  152        sssssssssG AGGTGCAGCT GTTGGAGTCT GGGGGAGGCT TGGTACAGCC TGGGGGGTCC
    seqs = defaultdict(str)
    edges = []
    pattern0 = re.compile(
        "^\s*(?P<from>[a-zA-Z0-9>_.-]+)\s+(?P<id>[a-zA-Z0-9>_.-]+)\s+(yes\s+|no\s+|maybe\s+)?(?P<seq>[01?. \-]+)"
    )
    pattern_cont = re.compile("^\s*(?P<seq>[01?. \-]+)")
    fh.next()
    last_group_id = None
    for line in fh:
        m = pattern0.match(line)
        m_cont = pattern_cont.match(line)
        if m:
            last_blank = False
            last_group_id = m.group("id")
            edges.append((m.group("from"), m.group("id")))
            seqs[last_group_id] = m.group("seq").replace(" ", "").upper()
        elif m_cont:
            last_blank = False
            seqs[last_group_id] += m_cont.group("seq").replace(" ", "").upper()
        elif line.rstrip() == '':
            if last_blank:
                break
            else:
                last_blank = True
                continue
        else:
            break
    return seqs, edges


# parse the dnaml output file and return data structures containing a
# list biopython.SeqRecords and a dict containing adjacency
# relationships and distances between nodes.
def parse_outfile(outfile):
    '''parse phylip outfile'''
    trees = []
    # Ugg... for compilation need to let python know that these will definely both be defined :-/
    with open(outfile, 'rU') as fh:
        for sect in sections(fh):
            if sect[0] == 'sequences':
                sequences, edges = parse_seqdict(fh, sect[1])
                trees.append(build_tree(sequences, edges))
    return trees


# build a tree from a set of sequences and an adjacency dict.
def build_tree(sequences, edges):
    # build an ete tree
    # first a dictionary of disconnected nodes
    seq_len = 0
    nodes = {}
    for name in sequences:
        node = Tree()
        node.name = name
        node.add_feature('barcode', sequences[node.name])
        seq_len = len(node.barcode)
        nodes[name] = node
    for node_from, node_to in edges:
        if node_from in nodes:
            nodes[node_from].add_child(nodes[node_to])
        else:
            assert (node_from == "root")
            nodes[node_from] = Tree()
            tree = nodes[node_from]
            tree.add_child(nodes[node_to])

    tree.add_feature('barcode', "0" * seq_len)
    return tree


def hamming_distance(seq1, seq2):
    '''Hamming distance between two sequences of equal length'''
    return sum(x != y for x, y in zip(seq1, seq2))
