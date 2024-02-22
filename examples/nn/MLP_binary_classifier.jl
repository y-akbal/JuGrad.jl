using JuGrad
using JuGrad.nn:Linear
using Random
using StatsBase
using PyCall
@pyimport sklearn.datasets as dataset
@pyimport numpy as np



function return_dataset(n_samples = 100; seed = 0)

    X_train, y_train = dataset.make_circles(n_samples=n_samples, random_state = seed)
    X_test, y_test = dataset.make_circles(n_samples=n_samples, random_state = seed+1)
    return map(x->np.transpose(x), [X_train, y_train, X_test, y_test])
end


@kwdef mutable struct sequential_binary <: JuGrad.nn.AbstractNeuralNetwork
    lay1 = Linear(2, 10; σ = JuGrad.leakyrelu_)
    lay2 = Linear(10, 10; σ = JuGrad.leakyrelu_)
    lay3 = Linear(10, 1) ## 
end

function (seq::sequential_binary)(x)
    return seq.lay3(seq.lay2(seq.lay1(x)))
end

function loss_from_logits(y_pred, y_true)
    return  -mean(@. (1-y_true)*JuGrad.log_(JuGrad.sigmoid_(1- y_pred)) + (y_true)*JuGrad.log_(JuGrad.sigmoid_(y_pred)))
end
"""
X_train, y_train, X_test, y_test = return_dataset()
network = sequential_binary()
network(X_test)
"""
function main()
    optimizer = JuGrad.nn.Descent(0.000000000000000000001)
    X_train, y_train, X_test, y_test = return_dataset()
    @info "Data Collected!!!"
    network = sequential_binary()
    @info "Network Created, the training is about to start!!!"

    for i in 1:1000
        loss_ = loss_from_logits(network(X_train),y_train) ## Calculate the loss!!!
        @info "loss is $(loss_)"
        loss_.grad = 1 ## This is kinda must, we can eliminate; though the code would be pretty ugly and hard to read!!!
        backward!(loss_) ## We accumulated the gradients in a backwards manner now!!!
        JuGrad.nn.step!(optimizer, network)
        zero_grad!(loss_)
    end
end




main()
## Put a real dataset here!!!





