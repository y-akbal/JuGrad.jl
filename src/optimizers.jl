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
            step!(opt, field)
        end
    end
end


function step!(opt::Descent, layer::JuGrad.nn.AbstractLayer)
    for lay in propertynames(layer)
        field = getfield(layer, lay)
        if isa(field, AbstractVecOrMat)
            println(field)
        end
    end
end



