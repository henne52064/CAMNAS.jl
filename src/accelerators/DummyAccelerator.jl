export DummyAccelerator, DummyLUdecomp
export discover_accelerator, check_accelerator, mna_decomp, mna_solve

struct DummyAccelerator <: AbstractAccelerator
    name::String
    properties::AcceleratorProperties

    function DummyAccelerator(name::String, properties=AcceleratorProperties(true, 1, 1.0, 1.0))
        new(name, properties)
    end



    function DummyAccelerator()
        new("", AcceleratorProperties(true, 1, 1.0, 1.0))
    end

end

struct DummyLUdecomp <: AbstractLUdecomp
end

# function discover_accelerator()
#     return DummyAccelerator()
# end

# function discover_accelerator(accelerator::AbstractAccelerator) end
# function check_accelerator(accelerator::AbstractAccelerator) end
# function estimate_flops(accelerator::AbstractAccelerator) end

# function mna_solve(my_system_matrix, rhs, accelerator::AbstractAccelerator) end



# function mna_decomp(sparse_mat::DummyLUdecomp, accelerator::DummyAccelerator)
#     lu_decomp = SparseArrays.lu(sparse_mat) |> CPU_LUdecomp
#     @debug "Dummy"
#     return lu_decomp
# end