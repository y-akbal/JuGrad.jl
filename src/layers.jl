using JuGrad

abstract type AbstractLayer end
abstract type AbstractNeuralNetwork end

function get_num_params(layer::AbstractLayer)
    counter::Int64 = 0
    for lay in propertynames(layer)
        field = getfield(layer, lay)
        if isa(field, AbstractVecOrMat)
            counter += field |> size |> prod
        end
    end
    return counter
end


function get_num_params(network::AbstractNeuralNetwork)
    counter::Int64 = 0
    for layer in propertynames(network)
        field = getfield(network, layer)
        if isa(field, AbstractLayer)
            counter += get_num_params(field)
        end
    end
    return counter
end


mutable struct Linear{T} <: AbstractLayer where T
    w::AbstractVecOrMat{T}
    b::AbstractVecOrMat{T}
    f::JuGrad.diff_f
end

function Linear(d_in::Int, d_out::Int; σ = JuGrad.ID)
    return Linear(randn(d_out, d_in) .|> JuGrad.t_number, randn(d_out,1) .|> JuGrad.t_number, σ)
end

Base.show(io::IO, t::AbstractLayer) = print(io, "Layer with $(get_num_params(t)) parameters, and activation $(t.f)")
Base.show(io::IO, t::AbstractNeuralNetwork) = print(io, "Network with $(get_num_params(t)) parameters!!!")




function (lin::Linear)(x)
    ## This dude is the forward pass function!!!!
    return map(lin.f, lin.w*x .+ lin.b)
end

function get_weights(layer::AbstractLayer)
    states = Dict{Symbol, AbstractVecOrMat}()
    for lay in propertynames(layer)
        field = getfield(layer, lay)
        if isa(field, AbstractVecOrMat)
            states[lay] = field
        end
    end
    return states
end

function retrieve_grads(layer::AbstractLayer)
    states = Dict{Symbol, AbstractVecOrMat}()
    for lay in propertynames(layer)
        field = getfield(layer, lay)
        if isa(field, AbstractVecOrMat)
            states[lay] = field .|> x->x.grad
        end
    end
    return states
end

### Examples Now
function retrieve_grads(network:: AbstractNeuralNetwork)
    states = Dict{Symbol, Any}()
    for lay in propertynames(network)
        field = getfield(network, lay)
        if isa(field, AbstractLayer)
            states[lay] = get_grads(field)
        end
    end
    return states
end



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


X = randn(10, 10)
y = randn(1, 10)
series = sequential()

loss_ = loss(series(X), y)
loss_.grad = 1
loss_.w

backward!(loss_)

get_grads(series)[:lay1][:w]



"""
series(randn(10,10))

"""
lin = Linear(10,1;σ = JuGrad.tanh_)
z = lin(randn(10,20)) |> JuGrad.sum_
z.grad = 1.0
JuGrad.backward!(z)
lin.w 
2.0*lin.w
"""





