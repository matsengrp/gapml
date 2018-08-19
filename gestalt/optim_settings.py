class KnownModelParams:
    """
    Specify which model parameters are known
    """
    def __init__(self,
            target_lams: bool = False,
            double_cut_weight: bool = False,
            trim_long_factor: bool = False,
            branch_lens: bool = False,
            cell_lambdas: bool = False,
            tot_time: bool = False,
            indel_params: bool = False):
        self.target_lams = target_lams
        self.double_cut_weight = double_cut_weight or target_lams
        self.trim_long_factor = trim_long_factor or target_lams
        self.branch_lens = branch_lens
        self.cell_lambdas = cell_lambdas
        self.tot_time = tot_time
        self.indel_params = indel_params
