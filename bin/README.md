# GESTALT Code structure

CLT = cell lineage tree

`class NodeDetails`
* Features for nodes in the cell lineage tree (features can be set to None if data is not available)
* `self.barcode` <!-- Can we hear about the internal representation of barcodes? -->
* `self.cell_type` (or `gene_expr` in the future)

<!-- This is an "observed and aligned sequence", namely a description of what barcodes have been cut, right? -->
`class ObservedSeq`
* Stores information from an observed sequence
* `self.barcode`
* `self.cell_type` (or maybe `gene_expr` in the future)
* `self.abundance`
* any other things?

`class CellLineageTree`
* History from embryo cell to observed cells. Each node represents a cell divison/death. Class can be used for storing information about the true cell lineage tree and can be used for storing the estimate of the cell lineage tree.
* `self.tree = TreeNode()` -- root node with feature `tree.details = NodeDetails(...)` <!--- is TreeNode an ETE thing or something? Are you rolling your own? --->
* `def add_child(node)` -- for constructing the tree
* `def render()` -- renders the tree. We can also add other functions for visualizing the cell lineage tree.

`class CellTypeTree`
* Stores a cell-type tree with parameters for generating cell lineage trees.
* `self.tree = TreeNode()` -- root node.
<!-- What differentiates this from the previous class in terms of code? (not in terms of concept). For species trees, the extra info is population sizes so one can generate coalescent trees. -->

`class CLTSimulator`
* Parent class for simulating cell lineage trees. We can subclass this to play around with different generative models.
* `def simulate(time)`
* `class CLTSimulatorCellType(CLTSimulator)`
  * Subclass where we generate CLT based on cell division/death/cell-type-differentiation. Barcode is independently modified along branches.
  <!-- Is it worth factoring out the barcode simulation from the tree simulation? I suppose that's limiting us in future development, but given our fundamental independence assumption it seems tidier. -->
  * `self.cell_type_tree = CellTypeTree()`
  * `self.birth_rate`
  * `self.death_rate`
* ... and more subclasses in the future

`class CLTObserver`
* Class that will simulate data observations from the CLT. Samples leaf nodes from CLT. Subclass if you want to sample leaf nodes in some fancy way (e.g. you can have biased sampling)
* `def observe_leaves(cell_lineage_tree)` -- samples leaves and returns observations as well as the cell lineage tree post-sampling (so internal nodes are now most recent ancestors)
  * `@return list[ObservedSeq], CellLineageTree`

`class FastqWriter`
* Writes observations to files for other methods to play with data
* `def write_fastq(list[ObservedSeq], file_name)`

`class CLTEstimator`
<!-- perhaps too early, but is this version 3 without the penalty? -->
* Class that estimates (predicts?) the CLT given a list of the observed sequences
* `def estimate(list[ObservedSeq])`
  * `@return CellLineageTree`
