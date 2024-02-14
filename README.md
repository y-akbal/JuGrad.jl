# JuGrad.jl

This is a experimental reverse mode autograd stuff designed for educational purposses. Eventhough it works well, no promise to be fast and furious! It is in Pytorch's style: uses backward! and zero_grad! functions.
It does take scalar based gradients, therefore can be used to implement simple optimization algorithms. 

````julia
grad(x->x^2, 1) ## Returns val and grads
## or 
grad(x->x[1]^2+x[2]^2, randn(2))
````
A little lower level version works as follows:

````julia
x = t_number(3.f0)
y = t_number(5.f0)
z = sin(x^2 + y^2)
z.grad = 1
backward!(z)
x.grad, y.grad
zero_grad!(z)
````

Here t_number is a special type that whispers the gradient values (or whatever needed) to do backwardpass to its child t_number. At the moment there is NO ""stop gradient"" kinda option. zero_grad! function zeros the grads of all the dudes in the computational graph. At the moment there is no plan to implement things like Hessian so forth. 


<p align="center">

<img src="memes.png" width="512" class="center"/>

</p>

# JuGrad.jl
