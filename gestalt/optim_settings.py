class KnownModelParams:
    """
    Specify which model parameters are known
    """
    def __init__(self,
            target_lams: bool = False,
            target_lams_intercept: bool = False,
            double_cut_weight: bool = False,
            branch_lens: bool = False,
            cell_lambdas: bool = False,
            tot_time: bool = False):
        self.target_lams = target_lams
        self.target_lams_intercept = target_lams_intercept or target_lams
        self.double_cut_weight = double_cut_weight or target_lams
        self.branch_lens = branch_lens
        self.cell_lambdas = cell_lambdas
        self.tot_time = tot_time
