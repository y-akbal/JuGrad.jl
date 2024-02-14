
mutable struct diff_f <: Function
    f_::Function
    grad::Function
end

function (f::diff_f)(x::T) where T 
    ## This dude is the forward pass function!!!
    return f.f_(x)
end

function grad(f::diff_f, x::T)  where T <: Real
    return f.grad(x)
end

function grad(f::diff_f, x::T)  where T <: tracked_number
    return f.grad(x.w)
end

function (f::diff_f)(x::T) where T <: tracked_number
    ## This is for backwards pass
    res = t_number(f.f_(x.w))
    res.parents_grads[x] = grad(f, x)
    return res
end

## activation functions, and their derivatives!!!!
## Use here  @inline and @fastmath kind a stuff
relu_ = diff_f(x->max(x,0), x-> ifelse(x>0, one(x), zero(x)))
leakyrelu_ = diff_f(x->max(x,-0.1*x), x-> ifelse(x>0, one(x), -0.1*one(x)))
sigmoid_ = diff_f(x->1/(1+exp(-x)), x -> exp(x)/(1+exp(x))^2)
tanh_ = diff_f(x->(exp(x) - exp(-x))/(exp(x) + exp(-x)), x -> one(x)-((exp(x) - exp(-x))/(exp(x) + exp(-x)))^2)
log_ = diff_f(x->log(x), x->1/x)
sum_ = diff_f(x->sum(x), x->zero(x).+1)
ID = diff_f(x->x, x->one(x))
##
## Simple functions will be here like sum norm mean loss vs vs....
## 
