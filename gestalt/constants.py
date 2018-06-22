BARCODE_V7 = ('CG', 'GATACGATACGCGCACGCTATGG', 'AGTC',
              'GACACGACTCGCGCATACGATGG', 'AGTC', 'GATAGTATGCGTATACGCTATGG',
              'AGTC', 'GATATGCATAGCGCATGCTATGG', 'AGTC',
              'GAGTCGAGACGCTGACGATATGG', 'AGTC', 'GCTACGATACACTCTGACTATGG',
              'AGTC', 'GCGACTGTACGCACACGCGATGG', 'AGTC',
              'GATACGTAGCACGCAGACTATGG', 'AGTC', 'GACACAGTACTCTCACTCTATGG',
              'AGTC', 'GATATGAGACTCGCATGTGATGG', 'GA')

NUM_BARCODE_V7_TARGETS = int(len(BARCODE_V7) / 2)
BARCODE_V7_LEN = len("".join(BARCODE_V7))

NO_EVENT_STRS = ["NONE", "UNKNOWN"]

CONTROL_ORGANS = ["7B_1_to_500_blood", "7B_1_to_20_blood", "7B_1_to_100_blood"]

COLORS = ["cyan", "green", "orange"]

MIX_CFG_FILE = "mix.cfg"
MIX_PATH = "mix" #"~/phylip-3.697/exe/mix"

RSPR_PATH = "/Users/jeanfeng/rspr_1_3_0/rspr" # "/home/jfeng2/rspr_1_3_0/rspr"

UNLIKELY = "unlikely"
PERTURB_ZERO = 1e-10

NO_EVT_STR = "no_evts"
