using TSne, Statistics
using CSV, DataFrames
using ArgParse
using Random
using StatsBase
using Pkg


s = ArgParseSettings()
@add_arg_table s begin
    "--repetitions"
        arg_type = Int
        default = 1
    "--n-samples"
        arg_type = Int
        default = 0
end

args = parse_args(s)


dataset = CSV.File("data/10x_mouse_zheng.csv", delim=' ', header=false) |> DataFrame
data = Matrix{Float64}(dataset)


for i in 1:args["repetitions"]
    println("--------------------------------------------------------------------------------")
    println("TSne.jl: ", Pkg.installed()["TSne"])
    println("random state: $i")
    println("--------------------------------------------------------------------------------")

    Random.seed!(i)

    if args["n-samples"] != 0
        sample_idx = sample(axes(data, 1), args["n-samples"]; replace = false)
        sample_data = data[sample_idx, :]
    else
        sample_data = data
    end

    @time tsne(sample_data, 2, 50, 1000, 30.0, verbose=true, progress=false)
end
