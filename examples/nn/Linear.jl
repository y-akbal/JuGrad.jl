using JuGrad



@kwdef mutable struct sequential <: AbstractNeuralNetwork
    lay1 = Linear(10, 20; σ = JuGrad.sigmoid_)
    lay2 = Linear(20, 1)
end

function (seq::sequential)(x)
    return seq.lay2(seq.lay1(x))
end

function loss(x,y)
    return sum((x-y).^2)
end



