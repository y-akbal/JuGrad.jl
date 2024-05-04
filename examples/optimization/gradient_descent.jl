using JuGrad
using Zygote
using LinearAlgebra
using Random

const  lr::Float32 = 0.0f1
f(x) = (1.5 - x[1]+x[1]*x[2])^2 + (2.25 - x[1] +x[1]*x[2]^2)^2 + (2.625 - x[1] +x[1]*x[2]^3)^2

function optimize(f::Function,  ## Approximately takes 8285 steps to hit the local minima
    x_init::Vector{Float32}; 
    lr::Float32 = lr, 
    stop_criterion::Float32 = 1e-2 |> Float32,
    max_iter::Int = 1000)
    val = zero(f(x_init))
    for iter in 1:max_iter
        val, grad_ = grad(f, x_init)
        if norm(grad_[1]) < stop_criterion
            @info "Iteration stopped in $(iter) iterations !!! The norm of the gradient is $(norm(grad_))"
            return x_init, val
        else
            x_init -= lr*grad_
        end
    end          
    return x_init
end

begin
    Random.seed!(0)
    x_init = rand(Float32, 2) 
    optimize(x->f(x), x_init; lr = 0.001f0, max_iter = 10000)
end
