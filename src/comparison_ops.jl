import Base:<, <=,>, >=

for c in [:<, :<=, :>, :>=]
    @eval function ($c)(x::t_number, y::Real) 
        return ($c)(x.w, y)    
    end

    @eval function ($c)(x::Real, y::t_number)
        return ($c)(x, y.w)    
    end
end



