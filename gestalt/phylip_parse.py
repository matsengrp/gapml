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


def parse_seqdict(fh):
    """
    @return the list of edges along the tree.
                each edge is associated with a "difference sequence", which is the state at theupper node,
                where "." means it is the same as in the node below the tree
    NOTE: "below" means toward the root in Joe-speak
    """
    # key: edge, val: diff between top and bottom node
    edges = {}
    pattern0 = re.compile(
        "^\s*(?P<from>[a-zA-Z0-9>_-]+)\s+(?P<id>[a-zA-Z0-9>_.-]+)\s+(yes\s+|no\s+|maybe\s+)?(?P<seq>[01?. \-]+)"
    )
    pattern_cont = re.compile("^(?P<seq>[01?. \-]+)")
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
    """
    @return the list of the sequences at the leaves
    """
    fh.readline()
    fh.readline()
    pattern_cont = re.compile("^             (?P<seq>[01?. \-]+)")
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


def parse_outfile(outfile, countfile):
    '''parse phylip mix outfile'''
    if countfile is not None:
        with open(countfile) as f:
            # eat header
            f.readline()
            counts = {l.split()[0]:int(l.split()[1]) for l in f}
    # No count, just make an empty count dictionary:
    else:
        counts = None
    trees = []
    # Ugg... for compilation need to let python know that these will definely both be defined :-/
    with open(outfile, 'rU') as fh:
        for sect in sections(fh):
            if sect == 'leaves':
                # This should always be called before the edges section is reached
                leaves = parse_leaves(fh)
            if sect == 'edges':
                edges = parse_seqdict(fh)
                trees.append(build_tree(leaves, edges, counts))
    return trees


# build a tree from a set of edges
def build_tree(leaf_seqs, edges, counts=None):
    # build an ete tree
    # first a dictionary of disconnected nodes
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
            binary_allele_len = len(leaf_seqs[node_to_name])
            node_to.add_feature("binary_allele", leaf_seqs[node_to_name])
        else:
            node_to.add_feature("binary_allele", None)

    root_node = nodes["root"]
    root_node.add_feature("binary_allele", "".join(['0'] * binary_allele_len))

    for node_from_name, node_to_name in edges:
        nodes[node_from_name].add_child(nodes[node_to_name])

    # generate binary alleles
    for node in root_node.iter_descendants("preorder"):
        allele = ''
        for is_diff, allele_char in zip(node.difference, node.up.binary_allele):
            allele += allele_char if is_diff == "." else is_diff
        if node.is_leaf():
            assert node.binary_allele == allele
        else:
            node.binary_allele = allele

    assert set(root_node.binary_allele) == set('0')

    # # make random choices for ambiguous internal states, respecting tree inheritance
    # sequence_length = len(root_node.binary_allele)
    # for node in root_node.iter_descendants():
    #     for site in range(sequence_length):
    #         symbol = node.binary_allele[site]
    #         if symbol == '?':
    #             new_symbol = '0'#random.choice(('0', '1'))
    #             for node2 in node.traverse(is_leaf_fn=lambda n: False if symbol in [n2.binary_allele[site] for n2 in n.children] else True):
    #                 if node2.binary_allele[site] == symbol:
    #                     node2.binary_allele = node2.binary_allele[:site] + new_symbol + node2.binary_allele[(site+1):]

    # compute branch lengths
    root_node.dist = 0 # no branch above root
    for node in root_node.iter_descendants():
        node.dist = sum(x != y for x, y in zip(node.binary_allele, node.up.binary_allele)) # 0 if node.binary_allele == node.up.binary_allele else 1

    # # # convert the leaves back to contain "?"
    # # # NOTE: don't compute branch lengths from binary_allele differences after this
    # for leaf in root_node:
    #     leaf.binary_allele = leaf_seqs[leaf.name]

    if counts is not None:
        for node in root_node.traverse():
            if node.name in counts:
                node.add_feature('frequency', counts[node.name])
            else:
                node.add_feature('frequency', 0)

    return root_node
