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
  geom_stratum() + geom_text(stat = "stratum", label.strata = TRUE) +
  facet_grid(fish ~ cell_type)
num_cell_types = length(unique(flow_df$cell_type))
num_fish = length(unique(flow_df$fish))
ggsave(out.file, width = num_cell_types * 15, height = num_fish * 6, limitsize=F)
