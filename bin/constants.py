BARCODE_V7 = (  'CG', 'GATACGATACGCGCACGCTATGG',
      'AGTC', 'GACACGACTCGCGCATACGATGG',
      'AGTC', 'GATAGTATGCGTATACGCTATGG',
      'AGTC', 'GATATGCATAGCGCATGCTATGG',
      'AGTC', 'GAGTCGAGACGCTGACGATATGG',
      'AGTC', 'GCTACGATACACTCTGACTATGG',
      'AGTC', 'GCGACTGTACGCACACGCGATGG',
      'AGTC', 'GATACGTAGCACGCAGACTATGG',
      'AGTC', 'GACACAGTACTCTCACTCTATGG',
      'AGTC', 'GATATGAGACTCGCATGTGATGG',
      'GA')

NUM_BARCODE_V7_TARGETS = int(len(BARCODE_V7)/2)
BARCODE_V7_LEN = len("".join(BARCODE_V7))

NO_EVENT_STRS = ["NONE", "UNKNOWN"]

CONTROL_ORGANS = ["7B_1_to_500_blood", "7B_1_to_20_blood", "7B_1_to_100_blood"]
