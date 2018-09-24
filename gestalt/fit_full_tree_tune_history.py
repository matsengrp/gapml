"""
This goes thru the entire tuning history and refits the entire tree.
This is to check if that the hanging chad tuner is improving the entire tree as the number of
iterations increases.
"""
import six
import sys
import numpy as np

from tune_topology import main
from common import assign_rand_tree_lengths


true_file = 'simulation_topol_consist/_output/model_seed%d/%d/small/true_model.pkl'
obs_file = 'simulation_topol_consist/_output/model_seed%d/%d/small/num_barcodes1/obs_data.pkl'
out_model = 'simulation_topol_consist/_output/model_seed%d/%d/small/num_barcodes1/tune_fitted.pkl'
topo_template = 'simulation_topol_consist/_output/model_seed%d/%d/small/num_barcodes1/tune_ll_step%d.pkl'
init_template = 'simulation_topol_consist/_output/model_seed%d/%d/small/num_barcodes1/tune_ll_step_init%d.pkl'
out_template = 'simulation_topol_consist/_output/model_seed%d/%d/small/num_barcodes1/tune_ll_step%d_out.pkl'
log_template = 'simulation_topol_consist/_output/model_seed%d/%d/small/num_barcodes1/tune_ll_step%d.txt'
scratch_template = '_output/scratch%d'

model_seed = 1
seed = 0
tot_height = 1

new_log_file = "simulation_topol_consist/_output/model_seed%d/%d/small/num_barcodes1/tune_ll_steps.txt" % (model_seed, seed)
#new_log_file = "_output/test.txt"
out_model = out_model % (model_seed, seed)
obs_file = obs_file % (model_seed, seed)
true_file = true_file % (model_seed, seed)

with open(out_model, 'rb') as f:
    tune_hist = six.moves.cPickle.load(f)#['tuning_history']
    print("number of iters", len(tune_hist))

for idx, tune_step in enumerate(tune_hist):
    if idx <= int(sys.argv[1]):
        continue

    new_topo_file = topo_template % (model_seed, seed, idx)
    new_init_file = init_template % (model_seed, seed, idx)
    print(new_topo_file)
    new_out_file = out_template % (model_seed, seed, idx)
    new_scratch = scratch_template % idx

    tree, best_fit_params, best_fit_res = tune_step['chad_tune_result'].get_best_result()
    dist_to_root = best_fit_res.train_history[-1]['dist_to_roots']
    branch_len_inners = best_fit_res.train_history[-1]['var_dict']['branch_len_inners']
    branch_len_offsets = best_fit_res.train_history[-1]['var_dict']['branch_len_offsets_proportion']

    node_to_nochad_id = {}
    nochad_to_node_id = {}
    for node in best_fit_res.orig_tree.traverse():
        if node.nochad_id is not None:
            node_to_nochad_id[node.node_id] = node.nochad_id
            nochad_to_node_id[node.nochad_id] = node.node_id

    for node in tree.traverse():
        node.add_feature('old_id', node.node_id)
    num_nodes = tree.label_node_ids()
    print("num nodes", num_nodes)

    print(best_fit_res.orig_tree.get_ascii(attributes=['nochad_id']))
    branch_len_inners_new = np.zeros(num_nodes)
    branch_len_offsets_new = np.ones(num_nodes) * 0.4 + np.random.rand(num_nodes) * 0.1
    for node in tree.traverse('preorder'):
        if node.old_id is not None and node.old_id in nochad_to_node_id:
            branch_len_inners_new[node.node_id] = branch_len_inners[nochad_to_node_id[node.old_id]]
            branch_len_offsets_new[node.node_id] = branch_len_offsets[nochad_to_node_id[node.old_id]]

    for node in tree.traverse('preorder'):
        if node.old_id is None or node.old_id not in nochad_to_node_id:
            if node.is_leaf():
                branch_len_inners_new[node.node_id] = 1e-10
            else:
                remain_height = tot_height - dist_to_root[nochad_to_node_id[node.up.old_id]]
                assign_rand_tree_lengths(node, remain_height * 0.9)
                branch_len_inners_new[node.node_id] = remain_height * 0.1
                for child_node in node.get_descendants():
                    branch_len_inners_new[child_node.node_id] = child_node.dist
                    print("assigning", child_node.dist)
            break
    print(branch_len_inners_new)
    assert np.all(branch_len_inners_new[1:] > 0)
    print(branch_len_inners_new.shape)
    print(branch_len_offsets_new.shape)

    with open(new_topo_file, 'wb') as f:
        tree_step = {
                'tree': tree,
                'multifurc': True}
        six.moves.cPickle.dump(tree_step, f)

    with open(new_init_file, 'wb') as f:
        best_fit_params['branch_len_inners'] = branch_len_inners_new
        best_fit_params['branch_len_offsets_proportion'] = branch_len_offsets_new
        six.moves.cPickle.dump(best_fit_params, f)

    args = [
        '--obs-file',
        obs_file,
        '--topology-file',
        new_topo_file,
        '--init-model',
        new_init_file,
        '--true-model-file',
        true_file,
        '--out-model-file',
        new_out_file,
        '--log-file',
        new_log_file,
        '--log-barr',
        0.000000001,
        '--dist-to-half-pen-params',
        6400,
        '--num-penalty-tune-iters',
        1,
        '--num-penalty-tune-splits',
        5,
        '--num-chad-tune-iters',
        1,
        '--max-chad-tune-search',
        0,
        '--max-sum-states',
        1000,
        '--max-extra-steps',
        2,
        '--max-iters',
        10000,
        '--num-inits',
        1,
        '--tot-time-known',
        '--num-init-random-rearrange',
        0,
        '--seed',
        3309999,
        '--scratch',
        new_scratch]

    main([str(a) for a in args])
