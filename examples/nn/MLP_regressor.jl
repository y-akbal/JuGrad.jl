using JuGrad
using JuGrad.nn:Linear
using Random
using PyCall
@pyimport sklearn.datasets as dt



@kwdef mutable struct sequential <: JuGrad.nn.AbstractNeuralNetwork
    lay1 = Linear(10, 20; σ = JuGrad.relu_)
    lay2 = Linear(20, 20; σ = JuGrad.relu_)
    lay3 = Linear(20, 1)
end



function (seq::sequential)(x)
    return seq.lay3(seq.lay2(seq.lay1(x)))
end

function loss(x,y)
    L, B = x |> size
    return sum((x.-y).^2)/(L*B)
end


begin 
    ## Fake dataset
    Random.seed!(0)
    network = sequential()
    X = randn(10, 20) 
    y = randn(1, 20)
end
sum((network(X) - y).^2)


optimizer = JuGrad.nn.Descent(0.000000001)


for i in 1:1000
    loss_ = loss(network(X),y) ## Calculate the loss!!!
    @info "loss is $(loss_)"
    loss_.grad = 1 ## This is kinda must, we can eliminate; though the code would be pretty ugly and hard to read!!!
    backward!(loss_) ## We accumulated the gradients in a backwards manner now!!!
    JuGrad.nn.step!(optimizer, network)
    zero_grad!(loss_)
end

"""
a = randn(10) .|> x->JuGrad.t_number(x)
x = prod(a)
x.grad = 1
backward!(x)
"""

