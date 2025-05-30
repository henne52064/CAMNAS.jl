module Accelerators
import ..CAMNAS

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


# not in function on purpose, otherwise scope issue with include statements

# include all accelerator files
global accelerator_files
accelerator_files = Vector()
read(`pwd`, String)
for file in readdir(dirname(@__FILE__), join=true)
    if endswith(file, ".jl") && basename(file) != "Accelerators.jl" && basename(file) ∉ accelerator_files
        @debug "file found $file"
        push!(accelerator_files, basename(file))
        include(basename(file))
    end
end
@debug accelerator_files





function load_all_accelerators(accelerators::Vector{AbstractAccelerator})   # Accelerator structs are called like the .jl file
    global accelerator_files
    if isempty(accelerators)
        for file in accelerator_files
            structname = split(file, ".")[1]
            symbol =  Symbol(structname)

            if !isdefined(Accelerators, symbol)
                @warn "No struct named '$structname' found in module Accelerators."
                continue
            end

            accelerator_type = getfield(Accelerators, symbol)

            if !isdefined(@__MODULE__, :discover_accelerator)
                @warn "No function `discover_accelerator` defined for file '$file'."
                continue
            end

            try
                instance = accelerator_type()
                has_driver(instance)
                discover_accelerator(accelerators, instance)
            catch e
                @error "Failed to create instance of $structname , call discover_accelerator or driver not found: $e"
            end

        end
    end
end

function has_driver(accelerator::AbstractAccelerator)
    @error "driver not found"
end

function discover_accelerator(accelerators::Vector{AbstractAccelerator}, accelerator::AbstractAccelerator)

    if !isempty(filter(x -> x.name == "cpu", accelerators)) # check if cpu is already in accelerators_vector
        return
    end

    cpu_flops = estimate_flops(NoAccelerator())
    cpu = NoAccelerator("cpu", AcceleratorProperties(true, 1, cpu_flops, 95.0))
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
    return 1    # in case of missing implementation return 1 Flop
end

function get_tdp(accelerator::AbstractAccelerator)
    return floatmax(Float64)
end


end