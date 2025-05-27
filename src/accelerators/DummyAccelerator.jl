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

function discover_accelerator() end

function discover_accelerator(accelerators::Vector{AbstractAccelerator}, accelerator::DummyAccelerator) 
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



# function mna_decomp(sparse_mat::DummyLUdecomp, accelerator::DummyAccelerator)
#     lu_decomp = SparseArrays.lu(sparse_mat) |> CPU_LUdecomp
#     @debug "Dummy"
#     return lu_decomp
# end