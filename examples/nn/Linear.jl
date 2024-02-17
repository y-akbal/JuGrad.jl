using JuGrad
using JuGrad.nn:Linear
using Random

@kwdef mutable struct sequential <: AbstractNeuralNetwork
    lay1 = Linear(10, 20; σ = JuGrad.sigmoid_)
    lay2 = Linear(20, 1)
end

function (seq::sequential)(x)
    return seq.lay2(seq.lay1(x))
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
network.lay1.w .|> x->x.grad  ## As soon as I can I will write get grads function!!!!