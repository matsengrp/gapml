library(ape)

arg <- commandArgs(TRUE)
in.file <- arg[1]
out.file <- arg[2]
tree.height <- as.numeric(arg[3])
lambda <- as.numeric(arg[4])

tr <- read.tree(in.file)
calibration <- makeChronosCalib(tr, node="root", age.max=tree.height)
chr <- chronos(tr, lambda = lambda, calibration = calibration, quiet=T)

write.tree(chr, file = out.file)
