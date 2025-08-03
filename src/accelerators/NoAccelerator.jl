export NoAccelerator, NoAccelerator_LUdecomp
export discover_accelerator, check_accelerator, mna_decomp, mna_solve

struct NoAccelerator <: AbstractAccelerator 
    name::String
    properties::AcceleratorProperties
    

    function NoAccelerator(name::String, properties=AcceleratorProperties(true, 1, 1.0, floatmax()))
        new(name, properties)
    end

    NoAccelerator() = new()

end

struct NoAccelerator_LUdecomp <: AbstractLUdecomp 
    lu_decomp::LinearAlgebra.TransposeFactorization
end

function has_driver(accelerator::NoAccelerator)
    return true
end

function discover_accelerator(accelerators::Vector{AbstractAccelerator}, accelerator::NoAccelerator)
    
    try
        has_driver(NoAccelerator())
    catch e
        @error "NoAccelerator driver not found: $e"
        return
    end
    
    if !isempty(filter(x -> x.name == "cpu", accelerators)) # check if cpu is already in accelerators_vector
        return
    end

    if Sys.islinux() 
        cpu_flops = estimate_flops(NoAccelerator(), inT=Float64)
    elseif Sys.isapple()  
        cpu_flops = estimate_flops(NoAccelerator(), inT=Float32)    # Apple Silicon uses Float32 for Metal.jl
    end
    
    cpu_power = get_tdp(NoAccelerator())
    cpu = NoAccelerator("cpu", AcceleratorProperties(true, 1, cpu_flops, cpu_power))
    push!(accelerators, cpu)

end



function get_tdp(accelerator::NoAccelerator)
    return 95.0
end
