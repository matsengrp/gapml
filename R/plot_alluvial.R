library(ggplot2)
library(ggalluvial)

arg <- commandArgs(TRUE)
in.file <- arg[1]
out.file <- arg[2]

flow_df <- as.data.frame(read.table(in.file, sep=",", header=TRUE))
head(flow_df)
ggplot(flow_df,
       aes(y = Freq, x=time, alluvium=leaf_id, stratum = progenitor, fill = progenitor)) +
  geom_lode() + geom_flow() +
  geom_stratum() + geom_text(stat = "stratum", label.strata = TRUE)
ggsave(out.file, width = 25, height = 10)
