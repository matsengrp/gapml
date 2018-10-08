import sys
import six
import argparse
import os.path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import seaborn as sns

from common import parse_comma_str
from plot_analyze_gestalt_meta import get_allele_to_cell_states, load_fish
from cell_state import CellTypeTree
from cell_lineage_tree import CellLineageTree
from plot_mrca_matrices import plot_tree
import collapsed_tree

# STUFF FOR RENDERING
ORGAN_COLORS = {
    "7B_Brain": "DarkGreen",
    "7B_Eye1": "MediumSeaGreen", # left eye
    "7B_Eye2": "LightGreen",
    "7B_Gills": "Gold",
    "7B_Intestine": "MediumBlue", # intestinal bulb
    "7B_Upper_GI": "DarkBlue", # post. intestine
    "7B_Blood": "Red",
    "7B_Heart_chunk": "Maroon",
    "7B_Heart_diss": "FireBrick", # DHC
    "7B_Heart_GFP-": "LightCoral", # NC
    "7B_Heart_GFP+": "Pink", # cardiomyocytes
}
ORGAN_TRANSLATION = {
    "7B_Brain": "Brain",
    "7B_Eye1": "Left eye",
    "7B_Eye2": "Right eye",
    "7B_Gills": "Gills",
    "7B_Intestine": "Intestinal bulb",
    "7B_Upper_GI": "Post intestine",
    "7B_Blood": "Blood",
    "7B_Heart_chunk": "Heart",
    "7B_Heart_diss": "DHC",
    "7B_Heart_GFP-": "NC",
    "7B_Heart_GFP+": "Cardiomyocytes",
}
COLLAPSE_DIST = 0.001

def parse_args(args):
    parser = argparse.ArgumentParser(
            description="""
            plotting the entire gestalt tree.
            this only works if you have the entire ete package
            """)
    args = parser.parse_args(args)
    parser.add_argument(
        '--fish',
        type=str,
        default="ADR1")
    parser.add_argument(
        '--out-plot',
        type=str,
        default="_output/out.png")
    return args

def _expand_leaved_tree(fitted_bifurc_tree, allele_to_cell_state, cell_state_dict, default_dist_scale = 0, min_abund_thres = 0):
    """
    @param default_dist_scale: how much to assign the leaf branch to the different cell types vs. preserve as internal branch length
    @param min_abund_thres: minimum abundance for us to include that cell type leaf in the tree (we will always include
                        the cell type with the highest abundance, regardless of absolute abundance)
    """
    leaved_tree = fitted_bifurc_tree.copy()
    for l in leaved_tree:
        allele_str = l.allele_events_list_str
        if l.cell_state is None:
            old_dist = l.dist
            l.dist = old_dist * (1 - default_dist_scale)
            sorted_cell_states = sorted(
                    [(c_state_str, abund) for c_state_str, abund in allele_to_cell_state[allele_str].items()],
                    key = lambda c: c[1],
                    reverse=True)
            for c_state_str, abund in sorted_cell_states[:1]:
                new_child = CellLineageTree(
                    l.allele_list,
                    l.allele_events_list,
                    cell_state_dict[c_state_str],
                    dist = old_dist * default_dist_scale,
                    abundance = abund,
                    resolved_multifurcation = True)
                l.add_child(new_child)
            for c_state_str, abund in sorted_cell_states[1:]:
                if abund > min_abund_thres:
                    new_child = CellLineageTree(
                        l.allele_list,
                        l.allele_events_list,
                        cell_state_dict[c_state_str],
                        dist = old_dist * default_dist_scale,
                        abundance = abund,
                        resolved_multifurcation = True)
                    l.add_child(new_child)
    print("num leaves", len(leaved_tree))
    return leaved_tree

def plot_gestalt_tree(
        fitted_bifurc_tree,
        bcode_meta,
        organ_dict,
        allele_to_cell_state,
        cell_state_dict,
        out_plot_file):
    from ete3 import NodeStyle, SeqMotifFace
    fitted_bifurc_tree = _expand_leaved_tree(fitted_bifurc_tree, allele_to_cell_state, cell_state_dict)

    for l in fitted_bifurc_tree:
        nstyle = NodeStyle()
        nstyle["fgcolor"] = ORGAN_COLORS[organ_dict[str(l.cell_state)]]
        nstyle["size"] = 10
        l.set_style(nstyle)

    for leaf in fitted_bifurc_tree:
        # get the motif list for indels in the format that SeqMotifFace expects
        motifs = []
        for event in leaf.allele_events_list[0].events:
            motifs.append([
                event.start_pos,
                event.start_pos + len(event.insert_str),
                '[]',
                len(event.insert_str),
                10,
                'black',
                'blue',
                None
            ])
        for event in leaf.allele_events_list[0].events:
            motifs.append([
                event.start_pos,
                event.del_end,
                '[]',
                event.del_len,
                10,
                'black',
                'red',
                None
            ])
        seq = ''.join(bcode_meta.unedited_barcode)
        seqFace = SeqMotifFace(
            seq=seq.upper(),
            motifs=motifs,
            seqtype='nt',
            seq_format='[]',
            height=10,
            gapcolor='red',
            gap_format='[]',
            fgcolor='black',
            bgcolor='lightgrey')
        leaf.add_face(seqFace, 0, position="aligned")

    # Collapse distances for plot readability
    for node in fitted_bifurc_tree.get_descendants():
        if node.dist < COLLAPSE_DIST:
            node.dist = 0
    col_tree = collapsed_tree.collapse_zero_lens(fitted_bifurc_tree)

    legend_colors = {}
    for organ_key, color in ORGAN_COLORS.items():
        text = ORGAN_TRANSLATION[organ_key.replace("7B_", "")]
        label_dict = {
            "text": text,
            "color": "gray",
            "fontsize": 8}
        legend_colors[color] = label_dict

    print("at plotting phase....")
    plot_tree(
            col_tree,
            out_plot_file,
            width=400,
            show_leaf_name=False,
            legend_colors=legend_colors)

def main(args=sys.argv[1:]):
    args = parse_args(args)
    # TODO: this doesnt work right now. need to add in prefix of tmp_mount
    tree, obs_dict = load_fish(args.fish, do_chronos=False)
    allele_to_cell_state, cell_state_dict = get_allele_to_cell_states(obs_dict)
    organ_dict = obs_dict["organ_dict"]
    bcode_meta = obs_dict["bcode_meta"]
    plot_gestalt_tree(
        tree,
        bcode_meta,
        organ_dict,
        allele_to_cell_state,
        cell_state_dict,
        args.out_plot)


if __name__ == "__main__":
    main()
