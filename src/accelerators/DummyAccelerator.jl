export DummyAccelerator, DummyAccelerator_LUdecomp
export discover_accelerator, check_accelerator, mna_decomp, mna_solve

struct DummyAccelerator <: AbstractAccelerator
    name::String
    properties::AcceleratorProperties

    function DummyAccelerator(name::String = "dummy_accelerator", properties=AcceleratorProperties(true, 1, 1.0, 1.0))
        new(name, properties)
    end


end

struct DummyAccelerator_LUdecomp <: AbstractLUdecomp
    lu_decomp::LinearAlgebra.TransposeFactorization
end

function has_driver(accelerator::DummyAccelerator) 
    return true
end

function discover_accelerator(accelerators::Vector{AbstractAccelerator}, accelerator::DummyAccelerator) 
    
    try
        has_driver(accelerator)
    catch e
        @error "Dummy driver not found: $e"
        return
    end
    
    if !isempty(filter(x -> x.name == "dummy_accelerator", accelerators)) # check if cpu is already in accelerators_vector
        return
    end

    dummy_accelerator_flops = estimate_flops(DummyAccelerator())
    dummy_accelerator = DummyAccelerator("dummy_accelerator", AcceleratorProperties(true, 1, dummy_accelerator_flops, 95.0))
    push!(accelerators, dummy_accelerator)
end

# same implementation as NoAccelerator
function estimate_flops(accelerator::DummyAccelerator) # returns flops in GFLOPs
    
    return 400.0    #   choose an arbitrary FLOPs value for dummyaccelerator

end

function get_tdp(accelerator::DummyAccelerator) # returns flops in GFLOPs
    
    return 400.0    #   choose an arbitrary powerconsumption value for dummyaccelerator

end



# function mna_solve(my_system_matrix, rhs, accelerator::AbstractAccelerator) end



# function mna_decomp(sparse_mat::DummyAccelerator_LUdecomp, accelerator::DummyAccelerator)
#     lu_decomp = SparseArrays.lu(sparse_mat) |> NoAccelerator_LUdecomp
#     @debug "Dummy"
#     return lu_decomp
# end