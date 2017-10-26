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
        'leaves': "Name\s*Characters",
        'edges': "\s*\(\s*.\s*means\s+same",
    }
    patterns = {k: re.compile(v, re.IGNORECASE) for (k, v) in patterns.items()}
    for line in fh:
        for k, pat in patterns.items():
            if pat.match(line):
                yield k
                break


# iterate over entries in the sequences section
def parse_seqdict(fh):
    # key: edge, val: diff between top and bottom node
    edges = {}
    pattern0 = re.compile(
        "^\s*(?P<from>[a-zA-Z0-9>_.-]+)\s+(?P<id>[a-zA-Z0-9>_.-]+)\s+(yes\s+|no\s+|maybe\s+)?(?P<seq>[01?. \-]+)"
    )
    pattern_cont = re.compile("^\s*(?P<seq>[01?. \-]+)")
    fh.readline()
    last_group_id = None
    for line in fh:
        m = pattern0.match(line)
        m_cont = pattern_cont.match(line)
        if m:
            last_blank = False
            last_edge_id = (m.group("from"), m.group("id"))
            edges[last_edge_id] = m.group("seq").replace(" ", "").upper()
        elif m_cont:
            last_blank = False
            edges[last_edge_id] += m_cont.group("seq").replace(" ", "").upper()
        elif line.rstrip() == '':
            if last_blank:
                break
            else:
                last_blank = True
                continue
        else:
            break

    return edges


def parse_leaves(fh):
    fh.readline()
    fh.readline()
    pattern_cont = re.compile("^\s*(?P<seq>[01?. \-]+)")
    pattern0 = re.compile("^(?P<leaf>[a-zA-Z0-9>_.-]+)\s*(?P<seq>[01?. \-]+)")

    leaf_seqs = {}
    last_leaf_id = None
    for line in fh:
        m = pattern0.match(line)
        m_cont = pattern_cont.match(line)
        if m:
            last_blank = False
            last_leaf_id = m.group("leaf")
            leaf_seqs[last_leaf_id] = m.group("seq").replace(" ", "")
        elif m_cont:
            last_blank = False
            leaf_seqs[last_leaf_id] += m_cont.group("seq").replace(" ", "")
        elif line.rstrip() == '':
            if last_blank:
                break
            else:
                last_blank = True
                continue
        else:
            break
    return leaf_seqs


# parse the dnaml output file and return data structures containing a
# list biopython.SeqRecords and a dict containing adjacency
# relationships and distances between nodes.
def parse_outfile(outfile):
    '''parse phylip outfile'''
    trees = []
    # Ugg... for compilation need to let python know that these will definely both be defined :-/
    with open(outfile, 'rU') as fh:
        for sect in sections(fh):
            if sect == 'leaves':
                # This should always be called before the edges section is reached
                leaves = parse_leaves(fh)
            if sect == 'edges':
                edges = parse_seqdict(fh)
                trees.append(build_tree(leaves, edges))
    return trees


# build a tree from a set of edges
def build_tree(leaf_seqs, edges):
    # build an ete tree
    # first a dictionary of disconnected nodes
    seq_len = 0
    nodes = {}
    for (node_from_name, node_to_name), diff_seq in edges.items():
        if node_from_name not in nodes:
            node_from = Tree()
            node_from.name = node_from_name
            nodes[node_from_name] = node_from
        if node_to_name not in nodes:
            node_to = Tree()
            node_to.name = node_to_name
            nodes[node_to_name] = node_to
        else:
            node_to = nodes[node_to_name]
        node_to.add_feature("difference", diff_seq)
        if node_to_name in leaf_seqs:
            node_to.add_feature("binary_barcode", leaf_seqs[node_to_name])
        else:
            node_to.add_feature("binary_barcode", None)

    root_node = nodes["root"]
    root_node.add_feature("binary_barcode", None)

    for node_from_name, node_to_name in edges:
        nodes[node_from_name].add_child(nodes[node_to_name])

    for node in root_node.iter_descendants("postorder"):
        distance = 0
        bcode_arr = []
        for is_diff, bcode_char in zip(node.difference, node.binary_barcode):
            if is_diff == ".":
                bcode_arr.append(bcode_char)
            else:
                bcode_arr.append("0")
                distance += 1
        if node.up.binary_barcode is None:
            node.up.binary_barcode = "".join(bcode_arr)
        node.dist = distance

    return root_node


def hamming_distance(seq1, seq2):
    '''Hamming distance between two sequences of equal length'''
    return sum(x != y for x, y in zip(seq1, seq2))
