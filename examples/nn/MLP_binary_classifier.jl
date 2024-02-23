using JuGrad
using JuGrad.nn:Linear
using Random
using StatsBase
using PyCall

@pyimport sklearn.datasets as dataset
@pyimport numpy as np



function return_dataset(n_samples = 100; seed = 0)
    X_train, y_train = dataset.make_moons(n_samples=n_samples, random_state = seed)
    X_test, y_test = dataset.make_moons(n_samples=n_samples, random_state = seed+1)
    X_train, X_test = map(x->x+0.1*randn(x|>size), [X_train, X_test])
    return map(x->np.transpose(x), [X_train, y_train, X_test, y_test])
end

"""X_train, y_train, X_test, y_test = return_dataset()

using Plots
X_train = X_train |> transpose
scatter(X_train[y_train .== 1, 1], X_train[y_train .== 1, 2], label = "Label-1")
scatter!(X_train[y_train .== 0, 1], X_train[y_train .== 0, 2], label = "Label-0")
using IterTools
p = product(1:10,1:10)
f.(p)
network = sequential_binary()
product(1:10, 1:10)
for i in -1:0.1:1
    for j in -1:0.1:1
        scatter!([i j])

    end
end
"""
@kwdef mutable struct sequential_binary <: JuGrad.nn.AbstractNeuralNetwork
    lay1 = Linear(2, 10; σ = JuGrad.tanh_)
    lay2 = Linear(10, 1)
end

function (seq::sequential_binary)(x)
    return seq.lay2(seq.lay1(x))
end

seq_prediction(X) = map(x->Float64(x.w), network(X |> transpose)) |> transpose

function loss_from_logits(y_pred, y_true)
    y_true = y_true |> transpose ## Remember that Julia is column-major!!!<ssss
    return -mean( @. (1-y_true)*JuGrad.log_(1-JuGrad.sigmoid_(y_pred)) + (y_true)*JuGrad.log_(JuGrad.sigmoid_(y_pred)))
end


function main()
    Random.seed!(0)

    optimizer = JuGrad.nn.Descent(0.01)
    X_train, y_train, X_test, y_test = return_dataset()

    @info "Data Collected!!!"
    network = sequential_binary()
    @info "Network Created, the training is about to start!!!"
    acc::Float64 = 0.0

    for i in 1:10000
        loss_ = loss_from_logits(network(X_train),y_train) ## Calculate the loss!!!
        if i%100 == 0
            @info "loss is $(loss_) and accuracy $(acc)"
        end

        loss_.grad = 1 ## This is kinda must, we can eliminate; though the code would be pretty ugly and hard to read!!!
        backward!(loss_) ## We accumulated the gradients in a backwards manner now!!!
        JuGrad.nn.step!(optimizer, network)

        acc = (1*(map(x->Float64(x.w), network(X_test)) .> 0) .== y_test |> transpose .|> Int64) |> mean
        zero_grad!(loss_)

    end

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end





