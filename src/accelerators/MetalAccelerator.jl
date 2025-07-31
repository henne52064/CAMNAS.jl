export MetalAccelerator, MetalAccelerator_LUdecomp
export discover_accelerator, check_accelerator, mna_decomp, mna_solve

using Metal

struct MetalAccelerator <: AbstractAccelerator 
    name::String
    properties::AcceleratorProperties


    function MetalAccelerator(name::String, properties::AcceleratorProperties)
        new(name, properties)
    end

    function MetalAccelerator(name::String)
        new(name, AcceleratorProperties(true, 1, 1.0, 1.0))
    end

    function MetalAccelerator()
        new("metal", AcceleratorProperties(true, 1, 1.0, 1.0))
    end

end

# FIXME: DenseMatrix since Sparse Matrix support for Metal.jl is not implemented yet.

struct MetalAccelerator_LUdecomp <: AbstractLUdecomp 
    lu_decomp::LU{Float32, MtlMatrix{Float32, Metal.PrivateStorage}, MtlVector{UInt32, Metal.PrivateStorage}}
    inverse::LU{Float32, MtlMatrix{Float32, Metal.PrivateStorage}, MtlVector{UInt32, Metal.PrivateStorage}}
end


function discover_accelerator(accelerators::Vector{AbstractAccelerator}, accelerator::MetalAccelerator)
    @debug "discovering MetalAccelerator"

    if !isempty(filter(x -> x.name == "metal", accelerators)) # check if metal is already in accelerators_vector
        return
    end

    #metal_flops = estimate_flops(MetalAccelerator())
    #metal_power = get_tdp(MetalAccelerator())
    #metal = MetalAccelerator("metal", AcceleratorProperties(true, 1, metal_flops, metal_power))
    metal = MetalAccelerator("metal", AcceleratorProperties(true, 1, 1.0, 1.0))
    push!(accelerators, metal)
    @debug "MetalAccelerator discovered and added to accelerators vector"
end

function has_driver(accelerator::MetalAccelerator)
    try
        !isempty(Metal.devices())
    catch e
        @error "Metal driver not found: $e"
        return false
    end
    return true
end

function check_accelerator(accelerators::Vector{AbstractAccelerator}, accelerator::MetalAccelerator)

    if !has_driver(accelerator)
        return false
    end

    return true
end

function estimate_flops(accelerator::MetalAccelerator) # returns flops in GFLOPs
    # TODO: Implement a proper estimation for Metal

    n = 4096
    trials = 5

    A = Metal.mtl(ones(Float32, n, n))
    B = Metal.mtl(ones(Float32, n, n))
    C = Metal.mtl(zeros(Float32, n, n))

    times = zeros(Float64, trials)

    # warmup
    mul!(C, A, B)

    
    for i in 1:trials
        GC.gc()  # does this make sense? to avoid GC in during benchmark
        times[i] = @elapsed begin
            Metal.@sync mul!(C, A, B)
        end
    end

    min_time = minimum(times)
    flops = 2 * n^3
    gflops = flops / (min_time * 1e9)

    println("Best time: $(round(min_time, digits=6)) s")
    println("GFLOPs: $(round(gflops, digits=2))")

    return gflops

    #return round(flops / 1e9, digits=2)
    
end

function get_tdp(accelerator::MetalAccelerator)
    # TODO: Implement a proper TDP retrieval for Metal
end


# Metal does not support ldiv or \, this is why we calculate with the inverse 
# either we calculate 2 lu decompositions or we move the calculated lu decomposition back to cpu, 
# to calculate the inverse only to then move the inverse back to the gpu

# this seems ineffective

function mna_decomp(sparse_mat, accelerator::MetalAccelerator)

    if rank(sparse_mat) < size(sparse_mat, 1)
        # matrix is not of full rank -> can convert, but not invertible
        @error "Matrix is not of full rank, cannot invert."
    end
    inverse = inv(sparse_mat)

    sparse_mat_mtl = Metal.mtl(sparse_mat)
    sparse_mat_mtl_inverse = Metal.mtl(inverse)

    decomp = lu(sparse_mat_mtl)
    inv_decomp = lu(sparse_mat_mtl_inverse) 
    
    return MetalAccelerator_LUdecomp(decomp, inv_decomp)
end

# Metal.jl does not include a solve function for LUdecomp yet and wrapper would be in objective-c
# so we use the inverse to solve the system
function mna_solve(system_matrix::MetalAccelerator_LUdecomp, rhs, accelerator::MetalAccelerator)
# FIXME:
# are dpsim matrices always nonsingular?

    # GPU solve step: apparently not numerically stable, not as performant as \ or ldiv
    
    rhs_mtl = Metal.mtl(rhs)    
    lhs_mtl = Metal.mtl(system_matrix.inverse.P) * (system_matrix.inverse.L * (system_matrix.inverse.U * rhs_mtl))

    lhs = Array(lhs_mtl) # convert back to CPU array
    return Float64.(lhs) # convert back to Float64 

    # CPU solve step
    # lu_decomp = Matrix(system_matrix.lu_decomp)
    # return system_matrix.lu_decomp \ rhs

end
