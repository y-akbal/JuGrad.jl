using JuGrad
using JuGrad.nn

abstract type AbstractOptimiser end

## We borrow some stuff from Flux 

mutable struct Descent <: AbstractOptimiser
    η::Float32
end


opt = Descent(0.1)


function step!(opt::AbstractOptimiser, layer::JuGrad.nn.AbstractNeuralNetwork)
    for lay in propertynames(layer)
        field = getfield(layer, lay)
        if isa(field, JuGrad.nn.AbstractLayer)
            ## Do you magic here!!!
            step!(opt, field)
        end
    end
end



function step!(opt::Descent, layer::JuGrad.nn.AbstractLayer)
    for lay in propertynames(layer)
        field = getfield(layer, lay)
        if isa(field, AbstractVecOrMat)
            ## We need to take step here!!!!
            println(field .|> x->x.w )
        end
    end
end

@inline function step!(a::AbstractVecOrMat{T}, ∇::AbstractVecOrMat{T}) where T <: tracked_number
    for (i, (a_x, g_x)) in enumerate(zip(a,∇))
            a[i].w += g_x
    end
end



#step!(opt, network)



