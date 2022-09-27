#!/usr/bin/env Rscript
library(optparse)
library(Rtsne)


sessionInfo()

option_list = list(
    make_option(c("--fname"), type="character"),
    make_option(c("--repetitions"), type="integer", default=1),
    make_option(c("--n-samples"), type="integer", default=FALSE),
    make_option(c("--n-threads"), type="integer", default=1)
)
opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

data = readRDS(opt$fname)

for (i in 1:opt$repetitions) {
    set.seed(i)
    print("--------------------------------------------------------------------------------")
    print("Rtsne")
    print(packageVersion("Rtsne"))
    cat("Random state", i, "\n")
    print("--------------------------------------------------------------------------------")

    if (opt$`n-samples`) {
        sample_idx = sample(nrow(data), opt$`n-samples`, replace=FALSE)
        sample_data = data[sample_idx,]
    } else {
        sample_data = data
    }
    print(dim(sample_data))

    start_time = Sys.time()
    embedding = Rtsne(sample_data, perplexity=30, theta=0.5, pca=FALSE, verbose=TRUE,
                      eta=200, num_threads=opt$`n-threads`)
    cat("Rtsne benchmark time:", difftime(Sys.time(), start_time, units="secs"), "\n")
}
