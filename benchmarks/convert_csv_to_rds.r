#!/usr/bin/env Rscript
library(optparse)

option_list = list(
    make_option(c("-i", "--input"), type="character"),
    make_option(c("-o", "--output"), type="character"),
    make_option(c("-f", "--force"), default=FALSE, action="store_true")
)

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

if (file.exists(opt$output) && !opt$force) {
    cat(sprintf("File `%s` exists. Doing nothing.", opt$output))
    quit()
}

cat(sprintf("Generating `%s`...", opt$output))
data = read.csv(opt$input, sep=" ", header=F)
saveRDS(data, opt$output)
