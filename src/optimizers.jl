using JuGrad
using JuGrad.nn

abstract type AbstractOptimiser end

## We borrow some stuff from Flux 

mutable struct Descent <: AbstractOptimiser
    η::Float32
end


function step!(opt::AbstractOptimiser, layer::JuGrad.nn.AbstractNeuralNetwork)
    for lay in propertynames(layer)
        field = getfield(layer, lay)
        if isa(field, JuGrad.nn.AbstractLayer)
            ## Do you magic here!!!
            step!(opt, field)
        end
    end
end



@inline function step!(opt::Descent, layer::JuGrad.nn.AbstractLayer)
    for lay in propertynames(layer)
        field = getfield(layer, lay)
        if isa(field, AbstractVecOrMat)
            ## We need to take step here!!!!
            step!(opt, field, field .|>  x->x.grad)
        end
    end
end

@inline function step!(opt::Descent, a::AbstractVecOrMat{T}, ∇::AbstractVecOrMat) where T <: tracked_number
    for i in eachindex(a, ∇)
            @inbounds a[i].w += -opt.η*∇[i]
    end
end


