using JuGrad
using Test
using Zygote
using Random
using BenchmarkTools



@testset "JuGrad.jl" begin
    f_1(x) = x[1]^2 + x[2]^2
    f_2(x) = x[1]^2*exp(x[2])^2/x[3]
    f_3(x) = sum(x)
    f_4(x) = (1.5 - x[1]+x[1]*x[2])^2 + (2.25 - x[1] +x[1]*x[2]^2)^2 + (2.625 - x[1] +x[1]*x[2]^3)^2
    f_5(x) = JuGrad.log_(prod(x)+1000)
    L::Array{Function} = [f_1, f_2, f_3, f_4, f_5]
    Shapes::Array{Tuple} = [(2,1), (3,1), (10,10), (2,1), (7,7)]

    function test_1(L::Array{Function}, Shapes::Array{Tuple} ; q::Int = 100000)::Bool
        m::Int = 0
        for (f, shape) in zip(L, Shapes)
            for _ in 1:q
                x = randn(shape)  
                m += 1*isapprox(Zygote.withgradient(f, x).grad[1], grad(f, x)[2])
            end
        end
        return m == q*length(L)
    end    



    @test test_1(L, Shapes; q = 1000)

end

