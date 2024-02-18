using JuGrad
using JuGrad.nn:Linear
using Random
using StatsBase


@kwdef mutable struct sequential_binary <: JuGrad.nn.AbstractNeuralNetwork
    lay1 = Linear(10, 20; σ = JuGrad.relu_)
    lay2 = Linear(20, 1) ## 
end



function (seq::sequential_binary)(x)
    return seq.lay2(seq.lay1(x))
end

function loss_from_logits(y_pred, y_true)
    return  -mean(@. (1-y_true)*JuGrad.log_(JuGrad.sigmoid_(1- y_pred)) + (y_true)*JuGrad.log_(JuGrad.sigmoid_(y_pred)))
end

## Put a real dataset here!!!

optimizer = JuGrad.nn.Descent(0.0000001)


for i in 1:1000
    loss_ = loss(network(X),y) ## Calculate the loss!!!
    @info "loss is $(loss_)"
    loss_.grad = 1 ## This is kinda must, we can eliminate; though the code would be pretty ugly and hard to read!!!
    backward!(loss_) ## We accumulated the gradients in a backwards manner now!!!
    JuGrad.nn.step!(optimizer, network)
    zero_grad!(loss_)
end



