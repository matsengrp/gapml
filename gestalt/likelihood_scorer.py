from parallel_worker import ParallelWorker

from simulate_common import fit_pen_likelihood

class LikelihoodScorer(ParallelWorker):
    def __init__(self, seed, args, tree, bcode_meta, cell_type_tree, approximator, init_model_vars):
        self.seed = seed
        self.args = args
        self.tree = tree
        self.bcode_meta = bcode_meta
        self.cell_type_tree = cell_type_tree
        self.approximator = approximator
        self.init_model_vars = init_model_vars

    def run_worker(self, shared_obj):
        sess = tf.Session()
        with sess.as_default():
            tf.global_variables_initializer().run()
            pen_ll, res_model = fit_pen_likelihood(
                self.tree,
                self.args,
                self.bcode_meta,
                self.cell_type_tree,
                self.approximator,
                sess,
                warm_start=self.curr_model_vars)
        return pen_ll, res_model
