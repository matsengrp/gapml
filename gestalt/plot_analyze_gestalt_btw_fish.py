import six
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

adr1_fitted_tree_file = "_output/gestalt_aws/ADR1_fitted.pkl"
adr2_fitted_tree_file = "_output/gestalt_aws/ADR2_fitted.pkl"

with open(adr1_fitted_tree_file, "rb") as f:
    res1 = six.moves.cPickle.load(f)[0]["best_res"]
with open(adr2_fitted_tree_file, "rb") as f:
    res2 = six.moves.cPickle.load(f)[0]["best_res"]

res1_targ = res1.model_params_dict["target_lams"]
res2_targ = res2.model_params_dict["target_lams"]

# Preliminary fit values from aws (on the old univarite machine)
#res1_targ = [1.64818856,0.20806834,0.08729824,0.5076326,0.18780561,0.27653949,0.24517234,0.10275936,1.17758429,0.17271956]
#res2_targ = [1.22273463,0.15447688,0.06615114,0.46774872,0.14034198,0.15865912,0.18763698,0.06139234,0.83340076,0.18396533]

plt.plot(res1_targ, label="ADR1")
plt.plot(res2_targ, label="ADR2")
plt.xlabel("Target index")
plt.ylabel("Estimated cut rate")
plt.legend()
plt.savefig("../../gestaltamania-tex/manuscript/images/adr1_2_target_lams.png")
