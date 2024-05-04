## To be used in lecture just for illustration purposes ##
## Instead of arrays, we will do the stuff with floating numbers ##
## -- Pytorch style --
## Beware of the fact that grad function is not fully compatible with native functions in Julia ##
## Therefore should be used with caution!!!

module JuGrad

__precompile__(true)

using Base:show, +, *, -, /, zero, one, abs

function __init__()
    @warn "Further debugging needed!!! Please use with caution!!!"
end

export t_number, grad, backward!, zero_grad!, tracked_number

abstract type tracked_number end
mutable struct t_number{T} <: tracked_number where T <: Real
    w::T
    grad::T
    parents_grads::Dict{tracked_number, T}
    terminal::Bool
end



## Basic arithmetic operations are defined here
function to_nonterminal!(t::t_number)
    ## We use this dude to convert the output to non_terminal state
    t.terminal = true
end

@inline function Base.:+(t::t_number, l::t_number)::t_number
    result = t_number(t.w+l.w)
    result.parents_grads[t] = one(t.w)
    result.parents_grads[l] = one(l.w)
    return result
end

@inline function Base.:-(t::t_number, l::t_number)::t_number
    result = t_number(t.w-l.w)
    result.parents_grads[t] = one(t.w)
    result.parents_grads[l] = -one(l.w)
    return result
end

@inline function Base.:+(t::t_number, l::T)::t_number where T <: Real
    result = t_number(t.w+l)
    result.parents_grads[t] = one(t.w)
    return result
end

@inline function Base.:+(l::T, t::t_number)::t_number where T <: Real
    ## We use here t_number + l 
    return t+l
end


@inline function Base.:-(t::t_number, l::T)::t_number where T <: Real
    result = t_number(t.w-l)
    result.parents_grads[t] = one(t.w)
    return result
end

@inline function Base.:-(l::T, t::t_number)::t_number where T <: Real
    result = t_number(l-t.w)
    result.parents_grads[t] = -one(t.w)
    return result
end



@inline function Base.:*(t::t_number, l::t_number)::t_number
    result = t_number(t.w*l.w)
    result.parents_grads[t] = l.w
    result.parents_grads[l] = t.w
    return result
end

@inline function Base.:/(t::t_number, l::t_number)::t_number
    result = t_number(t.w*inv(l.w))
    result.parents_grads[t] = inv(l.w)
    result.parents_grads[l] = -t.w*inv(l.w)^2
    return result
end

@inline function Base.:/(t::t_number, l::T)::t_number where T <: Real
    result = t_number(t.w*inv(l))
    result.parents_grads[t] = inv(l)
    return result
end


@inline function Base.:/(l::T, t::t_number)::t_number where T <: Real
    result = t_number(l*inv(t.w))
    result.parents_grads[t] = -inv(t.w)^2
    return result
end


@inline function Base.:*(t::t_number, c::T)::t_number where T<:Real
    result = t_number(c*t.w)
    result.parents_grads[t] = c*one(t.w)
    return result
end

@inline function Base.:*(c::T, t::t_number)::t_number where T<:Real
    result = t_number(c*t.w)
    result.parents_grads[t] = c*one(t.w)
    return result
end

@inline function Base.:-(t::t_number)::t_number 
    result = t_number(-t.w)
    result.parents_grads[t] = -one(t.w)
    return result
end

@inline function Base.:exp(t::t_number)::t_number
    result = t_number(exp(t.w))
    result.parents_grads[t] = exp(t.w)
    return result
end

@inline function Base.:abs(t::t_number)::t_number
    result = t_number(abs(t.w))
    result.parents_grads[t] = ifelse(t.w >= 0, 1, -1)
    return result
end


@inline function Base.:sin(t::t_number)::t_number
    result = t_number(sin(t.w))
    result.parents_grads[t] = cos(t.w)
    return result
end


@inline function Base.:cos(t::t_number)::t_number
    result = t_number(cos(t.w))
    result.parents_grads[t] = -sin(t.w)
    return result
end

@inline function Base.:^(t::t_number, i::T)::t_number where T <: Real
    if i != 0
        result = t_number((t.w)^i)
        result.parents_grads[t] = i*(t.w)^(i-1)
    else
        result = t_number(zero(t.w))
        result.parents_grads[t] = zero(t.w)
    end
    return result
end



Base.:zero(x::t_number) = zero(x.w)
Base.:one(x::t_number) = one(x.w)



##Constructor dude...
function t_number(w::T) where T <: Real
    return t_number(w, zero(w), Dict{tracked_number, T}(), false)
end

Base.show(io::IO, t::T) where T<: tracked_number = print(io, t.w)

@inline function zero_grad!(t::t_number)
    t.grad = zero(t.grad)
    ## Something that has been alread zeroes may be zeroed one more time!!!
    for parents in t.parents_grads |> keys
        zero_grad!(parents)
    end
end

@inline function backward!(l::t_number)
    for parents in l.parents_grads |> keys
        @fastmath parents.grad += l.parents_grads[parents]*l.grad
        ## Recursive call 
        backward!(parents)
    end 
end

function grad(f::Function, x::AbstractVecOrMat{T}) where T <: Real
    x_ = x .|> t_number
    z = f(x_) 
    z.grad = 1
    backward!(z)
    return z.w, map(x->x.grad, x_)
end

function grad(f::Function, x::T) where T <: Real
    x_ = x |> t_number
    z = f(x_) 
    z.grad = 1
    backward!(z)
    return z.w, x_.grad
end

include("diff_functions.jl")

module nn

__precompile__(true)

include("nn.jl")

export AbstractLayer, AbstractNeuralNetwork

end
end


