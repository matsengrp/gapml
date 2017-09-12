# gestaltamania

`simulation.py` <-- python3 script to simulate barcode editing as a Poisson process. `-h` flag for usage.

# Using PHYLIP MIX
Use options: O subgroup 1; 4; 5; P

# Pipeline
python simulation.py
python fasta_to_phylip.py
mix
python compare_trees.py
