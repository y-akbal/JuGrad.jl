using JuGrad

mutable struct Linear{T} <: AbstractLayer where T
    w::AbstractVecOrMat{T}
    b::AbstractVecOrMat{T}
    f::JuGrad.diff_f
end

function Linear(d_in::Int, d_out::Int; σ = JuGrad.ID)
    ## At the moment Float64 is fixed, later we shall change this!!!
    return Linear(randn(d_out, d_in)/(d_out) .|> JuGrad.t_number, randn(d_out,1)/(d_out) .|> JuGrad.t_number, σ)
end

Base.show(io::IO, t::AbstractLayer) = print(io, "Linear layer with $(get_num_params(t)) parameters, and $((t.f).def) activation!!!")
Base.show(io::IO, t::AbstractNeuralNetwork) = print(io, "Network with $(get_num_params(t)) parameters!!!")

function (lin::Linear)(x)
    ## This dude is the forward pass function!!!!
    return map(lin.f, lin.w*x .+ lin.b)
end



"""
X = randn(10, 10)
y = randn(1, 10)
series = sequential()

loss_ = loss(series(X), y)
loss_.grad = 1
loss_.w

backward!(loss_)

get_grads(series)[:lay1][:w]


"""


"""
lin = Linear(10,1;σ = JuGrad.tanh_)
z = lin(randn(10,20)) |> JuGrad.sum_
z.grad = 1.0
JuGrad.backward!(z)
lin.w 
2.0*lin.w
"""





