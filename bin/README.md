# GESTALT Code structure

CLT = cell lineage tree

`class Barcode`
* Stores the barcode state
* `self.barcode = list[str]` where each element in the barcode list is a target or a 3-bp separator (concatenating this list produces the full barcode sequence)
* `self.needs_repair = list[int]` -- list of targets that need repair. If list is empty, the barcode has no cuts.
* `def indel(target1, target2, left_delete=0, right_delete=0, insertion='')` -- modify the barcode with insertion/deletion

`class CellState`
* Meta data about the cell state (we may have categorical labels, or continuous measurements, or both!)
* `self.categorical_state` -- cell state variable that is categorical, e.g. ["Brain", "Eye", ..]
* `self.cts_state` -- cell state variable that is continuous, e.g. [0.1, 0.3, ...]

`class CLTNode`
* Node for cell lineage tree
* `self.node = TreeNode()` -- using ETE
* `self.barcode = Barcode()`
* `self.cell_state = CellState()`

`class CellTypeNode`
* Node for cell-type tree
* `self.node = TreeNode()` -- using ETE
* `self.cell_state = CellState()` -- only contains categorical state, no continuous state

`class ObservedAlignedSeq`
* Stores information from an observed sequence, after the sequence has been aligned with the unmodified barcode
* `self.barcode = Barcode()`
* `self.cell_state = CellState()`
* `self.abundance` -- the proportion of times this barcode was observed

`class CellLineageTree`
* History from embryo cell to observed cells. Each node represents a cell divison/death. Class can be used for storing information about the true cell lineage tree and can be used for storing the estimate of the cell lineage tree.
* `self.tree = CLTNode()`
* `def add_child(node)` -- for constructing the tree
* `def render()` -- renders the tree. We can also add other functions for visualizing the cell lineage tree.

`class CellTypeTree`
* Stores a cell-type tree with parameters for generating cell lineage trees.
* `self.tree = CLTNode()` -- root node.

`class CLTSimulator`
* Parent class for simulating cell lineage trees. We can subclass this to play around with different generative models.
* `def simulate(time)`
* `class CLTSimulatorCellType(CLTSimulator)`
  * Subclass where we generate CLT based on cell division/death/cell-type-differentiation. Barcode is independently modified along branches.
  * `self.cell_type_tree = CellTypeTree()`
  * `self.birth_rate`
  * `self.death_rate`
* ... and more subclasses in the future

`class BarcodeSimulator`
* Simulates modifications to the barcode. Can be used by CLTSimulator to modify the barcode, though not necessarily. Using this class would assume that the barcode modifications are independent of cell lineage tree generation.
* `def simulate(time)`

`class CLTObserver`
* Class that will simulate data observations from the CLT. Samples leaf nodes from CLT. Subclass if you want to sample leaf nodes in some fancy way (e.g. you can have biased sampling)
* `def observe_leaves(cell_lineage_tree)` -- samples leaves and returns observations as well as the cell lineage tree post-sampling (so internal nodes are now most recent ancestors)
  * `@return list[ObservedSeq], CellLineageTree`

`class FastqWriter`
* Writes observations to files for other methods to play with data
* `def write_fastq(list[ObservedSeq], file_name)`

`class CLTEstimator`
* Class that estimates (predicts?) the CLT given observed data. We can subclass this if we plan on comparing different estimators. (This can even include phylip)
* `def estimate(list[ObservedSeq])`
  * `@return CellLineageTree`
  * `class CLTMaxPenLikEstimator` -- subclass that does maximum penalized likelihood estimation
    * `self.penalty_fcn` -- some penalty function

