## To be used in lecture just for illustration purposes ##
## Instead of arrays, we will do the stuff with floating numbers ##
## -- Pytorch style --
## Beware of the fact that grad function is not fully compatible with native functions in Julia ##
## Therefore should be used with caution!!!

module JuGrad

__precompile__(true)

using Base:show, +, *, -, /, zero, one

function __init__()
    @info "Further debugging needed! Use with caution!!!"
end

export t_number, grad, backward!, zero_grad!, tracked_number

abstract type tracked_number end
mutable struct t_number{T} <: tracked_number where T <: Real
    w::T
    grad::T
    parents_grads::Dict{tracked_number, T}
end


Base.:zero(x::t_number) = zero(x.w)
Base.:one(x::t_number) = one(x.w)

##Constructor dude...
function t_number(w::T) where T <: Real
    return t_number(w, zero(w), Dict{tracked_number, T}())
end

include("diff_functions.jl")

Base.show(io::IO, t::T) where T<: tracked_number = print(io, t.w)
## Basic arithmetic operations are defined here

function Base.:+(t::t_number, l::t_number)::t_number
    result = t_number(t.w+l.w)
    result.parents_grads[t] = one(t.w)
    result.parents_grads[l] = one(l.w)
    return result
end

function Base.:-(t::t_number, l::t_number)::t_number
    result = t_number(t.w-l.w)
    result.parents_grads[t] = one(t.w)
    result.parents_grads[l] = -one(l.w)
    return result
end

function Base.:+(t::t_number, l::T)::t_number where T <: Real
    result = t_number(t.w+l)
    result.parents_grads[t] = one(t.w)
    return result
end

function Base.:+(l::T, t::t_number)::t_number where T <: Real
    return t+l
end


function Base.:-(t::t_number, l::T)::t_number where T <: Real
    result = t_number(t.w-l)
    result.parents_grads[t] = one(t.w)
    return result
end

function Base.:-(l::T, t::t_number)::t_number where T <: Real
    result = t_number(l-t.w)
    result.parents_grads[t] = -one(t.w)
    return result
end



function Base.:*(t::t_number, l::t_number)::t_number
    result = t_number(t.w*l.w)
    result.parents_grads[t] = l.w
    result.parents_grads[l] = t.w
    return result
end

function Base.:/(t::t_number, l::t_number)::t_number
    result = t_number(t.w/l.w)
    result.parents_grads[t] = 1/l.w
    result.parents_grads[l] = -t.w/(l.w)^2
    return result
end

function Base.:/(t::t_number, l::T)::t_number where T <: Real
    result = t_number(t.w/l)
    result.parents_grads[t] = 1/l
    return result
end


function Base.:/(l::T, t::t_number)::t_number where T <: Real
    result = t_number(l/t.w)
    result.parents_grads[t] = -1/(t.w)^2
    return result
end


function Base.:*(t::t_number, c::T)::t_number where T<:Real
    result = t_number(c*t.w)
    result.parents_grads[t] = c*one(t.w)
    return result
end

function Base.:*(c::T, t::t_number)::t_number where T<:Real
    result = t_number(c*t.w)
    result.parents_grads[t] = c*one(t.w)
    return result
end

function Base.:-(t::t_number)::t_number 
    result = t_number(-t.w)
    result.parents_grads[t] = -one(t.w)
    return result
end

function Base.:exp(t::t_number)::t_number
    result = t_number(exp(t.w))
    result.parents_grads[t] = exp(t.w)
    return result
end


function Base.:sin(t::t_number)::t_number
    result = t_number(sin(t.w))
    result.parents_grads[t] = cos(t.w)
    return result
end


function Base.:cos(t::t_number)::t_number
    result = t_number(cos(t.w))
    result.parents_grads[t] = -sin(t.w)
    return result
end

function Base.:^(t::t_number, i::T)::t_number where T <: Real
    if i != 0
        result = t_number((t.w)^i)
        result.parents_grads[t] = i*(t.w)^(i-1)
    else
        result = t_number(zero(t.w))
        result.parents_grads[t] = zero(t.w)
    end
    return result
end

function zero_grad!(t::t_number)
    t.grad = zero(t.grad)
    for parents in t.parents_grads |> keys
        zero_grad!(parents)
    end
end

function backward!(l::t_number)
    for parents in l.parents_grads |> keys
        parents.grad += l.parents_grads[parents]*l.grad
        ## Recursive call 
        backward!(parents)
    end 
end

function grad(f::Function, x::AbstractVecOrMat{T}) where T <: Real
    x = x .|> t_number
    z = f(x) 
    z.grad = 1
    backward!(z)
    return z.w, map(x->x.grad, x)
end

function grad(f::Function, x::T) where T <: Real
    x = x |> t_number{Float32}
    z = f(x) 
    z.grad = 1
    backward!(z)
    return z.w, x.grad
end


module nn

include("nn.jl")

__precompile__(true)

export Linear

end


end


