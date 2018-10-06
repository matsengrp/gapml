import six
import numpy as np
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

label1 = "dome1"
label2 = "dome3"
fitted1_tree_file = "analyze_gestalt/_output/dome1_abund1/sum_states_10/extra_steps_0/tune_pen.pkl"
fitted2_tree_file = "analyze_gestalt/_output/dome3_abund1/sum_states_10/extra_steps_0/tune_pen.pkl"

with open(fitted1_tree_file, "rb") as f:
    #res1 = six.moves.cPickle.load(f)[0]["best_res"]
    res1 = six.moves.cPickle.load(f)["final_fit"]
with open(fitted2_tree_file, "rb") as f:
    #res2 = six.moves.cPickle.load(f)[0]["best_res"]
    res2 = six.moves.cPickle.load(f)["final_fit"]

res1_targ = res1.model_params_dict["target_lams"]
res2_targ = res2.model_params_dict["target_lams"]

# Preliminary fit values from aws (on the old univarite machine)
#res1_targ = [1.64818856,0.20806834,0.08729824,0.5076326,0.18780561,0.27653949,0.24517234,0.10275936,1.17758429,0.17271956]
#res2_targ = [1.22273463,0.15447688,0.06615114,0.46774872,0.14034198,0.15865912,0.18763698,0.06139234,0.83340076,0.18396533]

plt.plot(res1_targ, label=label1)
plt.plot(res2_targ, label=label2)
plt.xlabel("Target index")
plt.ylabel("Estimated cut rate")
plt.legend()
#plt.savefig("../../gestaltamania-tex/manuscript/images/%s_%s_target_lams.png" % (
plt.savefig("_output/%s_%s_target_lams.png" % (
    label1, label2))
