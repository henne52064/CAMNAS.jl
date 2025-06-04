export NoAccelerator, NoAccelerator_LUdecomp
export discover_accelerator, check_accelerator, mna_decomp, mna_solve

struct NoAccelerator <: AbstractAccelerator 
    name::String
    properties::AcceleratorProperties
    

    function NoAccelerator(name::String, properties=AcceleratorProperties(true, 1, 1.0, 1.0))
        new(name, properties)
    end

    
    function NoAccelerator()
        new("cpu", AcceleratorProperties(true, 1, 1.0, 1.0))
    end

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

    cpu_flops = estimate_flops(NoAccelerator())
    cpu_power = get_tdp(NoAccelerator())
    cpu = NoAccelerator("cpu", AcceleratorProperties(true, 1, cpu_flops, cpu_power))
    push!(accelerators, cpu)

end

function estimate_flops(accelerator::NoAccelerator) # returns flops in GFLOPs
    float_bits::Int = 64
    # run lscpu and collect lines
    if Sys.islinux()
    output = read(`lscpu`, String)
    lines = split(output, '\n')

    function get_field(key)
        for line in lines
            if startswith(line, key)
                return strip(split(line, ':')[2])
            end
        end
        return ""
    end

    # get required fields
    cores_per_socket = parse(Int, get_field("Core(s) per socket"))
    sockets = parse(Int, get_field("Socket(s)"))
    max_mhz = try
        parse(Float64, get_field("CPU max MHz"))
    catch
        # if max not available
        parse(Float64, get_field("CPU MHz"))
    end
    flags = split(get_field("Flags"))

    # Determine SIMD width in bits
    simd_bits = if "avx512f" in flags
        512
    elseif "avx2" in flags || ("avx" in flags && "fma" in flags)
        256
    elseif "sse2" in flags || "sse" in flags
        128
    else
        64  # fallback guess
    end
    elseif Sys.isapple()
        # Apple Silicon

        return 1.0  # Apple Silicon does not support FLOPs estimation via lscpu

    else
        error("Unsupported OS for FLOPs estimation")
    end


    # estimate FLOPs per cycle per core
    floats_per_vector = simd_bits / float_bits
    flops_per_cycle_per_core = floats_per_vector * 2  # 1 FMA = 2 FLOPs

    total_cores = cores_per_socket * sockets
    clock_hz = max_mhz * 1e6


    flops = total_cores * clock_hz * flops_per_cycle_per_core

    #println("Estimated FP64 peak: $(round(flops / 1e9, digits=2)) GFLOPs")

    return round(flops / 1e9, digits=2)

    

end

function get_tdp(accelerator::NoAccelerator)
    return 95.0
end



# function mna_decomp(sparse_mat, accelerator::AbstractAccelerator) end
# function mna_solve(my_system_matrix, rhs, accelerator::AbstractAccelerator) end
