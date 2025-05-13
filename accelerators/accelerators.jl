module Accelerators

export AbstractAccelerator, AcceleratorProperties, AbstractLUdecomp
export discover_accelerator, check_accelerator, mna_decomp, mna_solve, estimate_flops

using SparseArrays
using LinearAlgebra
# Hardwareawareness
abstract type AbstractAccelerator end

abstract type AbstractLUdecomp end

struct AcceleratorProperties
    availability::Bool
    priority::Int64
    flops::Float64      # in GFLOPs
    power_watts::Float64            # max Power usage
    energy_efficiency::Float64      # flops/W

    function AcceleratorProperties(availability::Bool, priority::Int64, flops::Float64, power_watts::Float64) 
        new(availability, priority, flops, power_watts, round(flops/power_watts, digits=4))
    end

    function AcceleratorProperties()
        new(true, 1, 1.0, 1.0, 1.0)
    end

end

include("CUDAccelerator.jl")
include("NoAccelerator.jl")
include("DummyAccelerator.jl")


function check_accelerator(accelerator::AbstractAccelerator) end

function discover_accelerator(accelerators::Vector{AbstractAccelerator}) 
    # call every discover function 

    discover_accelerator(accelerators, NoAccelerator())
    discover_accelerator(accelerators, CUDAccelerator())
    discover_accelerator(accelerators, DummyAccelerator())

    

end

function discover_accelerator(accelerators::Vector{AbstractAccelerator}, accelerator::AbstractAccelerator)

    if !isempty(filter(x -> x.name == "cpu", accelerators)) # check if cpu is already in accelerators_vector
        return
    end

    cpu_flops = estimate_flops(NoAccelerator())
    cpu = NoAccelerator("cpu", properties = AcceleratorProperties(true, 1, cpu_flops, 95.0))
    push!(accelerators, cpu)


end


function mna_decomp(sparse_mat, accelerator::AbstractAccelerator)
    lu_decomp = SparseArrays.lu(sparse_mat) |> CPU_LUdecomp
    @debug "CPU $lu_decomp"
    return lu_decomp
end


function mna_solve(system_matrix::AbstractLUdecomp, rhs, accelerator::AbstractAccelerator)
    return system_matrix.lu_decomp \ rhs
end

function estimate_flops(accelerator::AbstractAccelerator)     # returns flops in GFLOPs
    float_bits::Int = 64
    # run lscpu and collect lines
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

    # estimate FLOPs per cycle per core
    floats_per_vector = simd_bits / float_bits
    flops_per_cycle_per_core = floats_per_vector * 2  # 1 FMA = 2 FLOPs

    total_cores = cores_per_socket * sockets
    clock_hz = max_mhz * 1e6


    flops = total_cores * clock_hz * flops_per_cycle_per_core

    #println("Estimated FP64 peak: $(round(flops / 1e9, digits=2)) GFLOPs")

    return round(flops / 1e9, digits=2)
end


end