abstract type AbstractLayer end
abstract type AbstractNeuralNetwork end


include("layers.jl")
include("optimizers.jl")


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


function get_num_params(network::T) where T <: AbstractNeuralNetwork
    counter::Int64 = 0
    for layer in propertynames(network)
        field = getfield(network, layer)
        if isa(field, AbstractLayer)
            counter += get_num_params(field)
        end
    end
    return counter
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

function get_grads(a::AbstractVecOrMat)
    return map(x->x.grad, a)
end

function get_grads(layer::AbstractLayer)
    states = Dict{Symbol, AbstractVecOrMat}()
    for lay in propertynames(layer)
        field = getfield(layer, lay)
        if isa(field, AbstractVecOrMat)
            states[lay] = get_grads(field)
        end
    end
    return states
end

### Examples Now
function get_grads(network)
    states = Dict{Symbol, Any}()
    for lay in propertynames(network)
        field = getfield(network, lay)
        if isa(field, AbstractLayer)
            states[lay] = get_grads(field)
        end
    end
    return states
end


