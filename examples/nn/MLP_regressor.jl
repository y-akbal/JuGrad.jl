using JuGrad
using JuGrad.nn:Linear
using Random
using IterTools
using BenchmarkTools

@kwdef mutable struct sequential <: JuGrad.nn.AbstractNeuralNetwork
    lay1 = Linear(10, 20; σ = JuGrad.tanh_)
    lay2 = Linear(20, 20; σ = JuGrad.tanh_)
    lay3 = Linear(20, 1)
end

function (seq::sequential)(x)
    return seq.lay3(seq.lay2(seq.lay1(x)))
end

function loss(x,y)
    L, B = x |> size
    return sum((x-y).^2)/B
end


begin 
    ## Fake dataset
    Random.seed!(0)
    network = sequential()
    X = randn(10, 200) 
    y = randn(1, 200)
end

loss_ = loss(network(X),y) ## Calculate the loss!!!
loss_.grad = 1 ## This is kinda must, we can eliminate; though the code would be pretty ugly and hard to read!!!
backward!(loss_) ## We accumulated the gradients in a backwards manner now!!!


abstract type AbstractOptimiser end

## We borrow some stuff from Flux 

mutable struct Descent <: AbstractOptimiser
    η::Float32
end


opt = Descent(0.1)

function step!(opt::AbstractOptimiser, layer::JuGrad.nn.AbstractNeuralNetwork)
    ## This is the main functions should not be overridden!!!!
    for lay in propertynames(layer)
        field = getfield(layer, lay)
        if isa(field, JuGrad.nn.AbstractLayer)
            step!(opt, field)
        end
    end
end


function step!(opt::Descent, layer::JuGrad.nn.AbstractLayer)
    for lay in propertynames(layer)
        field = getfield(layer, lay)
        if isa(field, AbstractVecOrMat)
            ## We need to take step here!!!!
            println(field .|> x->x.w )
        end
    end
end

step!(opt, network)