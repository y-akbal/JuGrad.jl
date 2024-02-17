using JuGrad
using JuGrad.nn

abstract type AbstractOptimiser end

## We borrow some stuff from Flux 

mutable struct descent <: AbstractOptimiser
    η::Float32
end


function step!(opt::descent, network::JuGrad.nn.AbstractNeuralNetwork)
    return nothing
end

