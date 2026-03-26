module TestTask

using Plots

export ConvolutionalInterleaver, 
    compute_output, 
    recalculate_internal_params!, 
    plot_with_constant_interpolation!

include("utils.jl")
include("convolutional_interleaver.jl")

end # module TestTask
