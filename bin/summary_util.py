from typing import List

from cell_lineage_tree import CellLineageTree
"""
Produce summary statistics for a list of cell lineage trees
"""


def editing_profile(trees: List[CellLineageTree], outbase: str):
    '''plot profile of deletion frequency at each position over leaves'''
    for i, tree in enumerate(trees, 1):
        tree.editing_profile('{}.{}.editing_profile.pdf'.format(outbase, i))


def indel_boundary(trees: List[CellLineageTree], outbase: str):
    '''indel start/end plot for each tree'''
    for i, tree in enumerate(trees, 1):
        tree.indel_boundary('{}.{}.indel_boundary.pdf'.format(outbase, i))


def event_joint(trees: List[CellLineageTree], outbase: str):
    '''joint event stats'''
    for i, tree in enumerate(trees, 1):
        tree.event_joint('{}.{}.event_joint.pdf'.format(outbase, i))


def summary_plots(trees: List[CellLineageTree], file_name: str):
    n_cells = []
    n_genotypes = []
    n_indels = []
    indel_lens = []
    for tree in trees:
        # counter for the unique leaf genotypes
        genotypes = Counter(
            [''.join(leaf.barcode.barcode) for leaf in tree.tree])
        n_genotypes.append(len(genotypes))
        n_cells.append(sum(genotypes.values()))
        n_indels.append([
            len(re.findall('[-acgt]+', genotype))
            for genotype in genotypes.elements()
        ])
        indel_lens.append([
            len(run.group(0))
            for genotype in genotypes.elements()
            for run in re.finditer('-+', genotype)
        ])

    plt.figure(figsize=(12, 3))
    plt.subplot(1, 4, 1)
    plt.hist(n_cells, stacked=True)
    plt.xlabel('number of cells')
    plt.xlim([0, None])
    plt.subplot(1, 4, 2)
    plt.hist(n_genotypes, stacked=True)
    plt.xlabel('number of genotypes')
    plt.xlim([0, None])
    plt.subplot(1, 4, 3)
    plt.hist(n_indels, stacked=True)
    # for x in n_deletions:
    #     sns.distplot(x, hist=False)
    plt.xlabel('number of indels')
    plt.xlim([0, None])
    plt.subplot(1, 4, 4)
    plt.hist(indel_lens, stacked=True)
    # for x in deletion_lens:
    #     sns.distplot(x, hist=False)
    plt.xlabel('indel lengths')
    plt.xlim([0, None])
    sns.despine()
    plt.tight_layout()
    plt.savefig(file_name)


def write_sequences(trees: List[CellLineageTree], outbase: str):
    for i, tree in enumerate(trees, 1):
        tree.write_sequences('{}.{}.fastq'.format(outbase, i))


def render(trees: List[CellLineageTree], outbase: str):
    for i, tree in enumerate(trees, 1):
        tree.render('{}.{}.pdf'.format(outbase, i))
