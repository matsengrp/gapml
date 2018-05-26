import six
import scipy.stats
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from tree_distance import MRCADistanceMeasurer

def plot_mrca_matrices(file_name):
    with open(file_name, "rb") as f:
        pickled_res = six.moves.cPickle.load(f)
    mrca_meas = MRCADistanceMeasurer(pickled_res["true_tree"], None)
    fitted_results = pickled_res["res_workers"]
    num_matrices = len(fitted_results) + 1

    plt.figure(figsize=(4,14))
    plt.subplot(num_matrices * 100 + 11)
    plt.imshow(mrca_meas.ref_tree_mrca_matrix)
    plt.title('true mrca')

    for i, fitted_res in enumerate(fitted_results):
        fitted_mrca_mat = mrca_meas._get_mrca_matrix(fitted_res[0].fitted_bifurc_tree)
        ktau = scipy.stats.kendalltau(fitted_mrca_mat.flatten(), mrca_meas.ref_tree_mrca_matrix.flatten())
        plt.subplot(num_matrices * 100 + 12 + i)
        plt.imshow(fitted_mrca_mat)
        plt.title("%s: corr %.4f" % (fitted_res[1], ktau[0]))

    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.tight_layout()
    plt.savefig(file_name.replace(".pkl", "_mrca.png"))

plot_mrca_matrices("simulation_multifurc/_output/maxleaves80/minleaves40/time45/model300/data300/var0.000900/estimators_multifurc.pkl")
#plot_mrca_matrices("simulation_multifurc/_output/maxleaves80/minleaves40/time45/model207/data207/var0.000900/estimators_multifurc.pkl")
