using JuGrad
using Test
using Zygote
using Random
using BenchmarkTools


@testset "JuGrad.jl" begin

    function test_1(q::Int = 100000)::Bool
        f_1(x) = x[1]^2 + x[2]^2
        f_2(x) = x[1]^2*exp(x[2])^2/x[3]
        f_3(x) = sum(x)
        f_4(x) = (1.5 - x[1]+x[1]*x[2])^2 + (2.25 - x[1] +x[1]*x[2]^2)^2 + (2.625 - x[1] +x[1]*x[2]^3)^2
        f_5(x) = JuGrad.log_(prod(x)+1000)

        L::Array{Function} = [f_1, f_2, f_3, f_4, f_5]
        Shapes::Array{Tuple} = [(2,1), (3,1), (10,10), (2,1), (7,7)]
        m::Int = 0

        for (f, shape) in zip(L, Shapes)
            for _ in 1:q
                x = randn(shape)  
                m += 1*isapprox(Zygote.withgradient(f, x).grad[1], grad(f, x)[2])
            end
        end
        return m == q*length(L)
    end    

    function test_2(;size = 100)
        activation_funcs = [JuGrad.tanh_, JuGrad.relu_, JuGrad.sigmoid_, JuGrad.leakyrelu_]
        q::Int = 0
        target::Int = length(activation_funcs)*size
        for func in activation_funcs
            for t in 10*randn(size)
                j_grad = JuGrad.grad(x->func(x), t)
                z_grad = Zygote.gradient(x->func.f_(x),t)
                q += isapprox(j_grad[2], z_grad[1]; atol = 1e-4)
            end
        end
    return q == target
    end

    #TODO: Add here something on Neural Networks!!!
    #TODO: Forward functions and gradients of some loss functions
    @test test_1()
    @test test_2()
end

