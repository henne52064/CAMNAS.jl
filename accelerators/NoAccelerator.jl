export NoAccelerator, CPU_LUdecomp
export discover_accelerator, check_accelerator, mna_decomp, mna_solve

struct NoAccelerator <: AbstractAccelerator 
    name::String
    properties::AcceleratorProperties
    

    function NoAccelerator(name::String; properties=AcceleratorProperties(true, 1, 1.0, 1.0))
        new(name, properties)
    end

    function NoAccelerator(name::String)
        new(name, AcceleratorProperties(false, 1, 1.0, 1.0))
    end
    
    function NoAccelerator()
        new("cpu", AcceleratorProperties(true, 1, 1.0, 1.0))
    end

end

struct CPU_LUdecomp <: AbstractLUdecomp 
    lu_decomp::LinearAlgebra.TransposeFactorization
end


# function discover_accelerator(accelerator::AbstractAccelerator) end
# function check_accelerator(accelerator::AbstractAccelerator) end
# function estimate_flops(accelerator::AbstractAccelerator) end
# function mna_decomp(sparse_mat, accelerator::AbstractAccelerator) end
# function mna_solve(my_system_matrix, rhs, accelerator::AbstractAccelerator) end
