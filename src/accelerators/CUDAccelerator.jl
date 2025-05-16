export CUDAccelerator, CUDA_LUdecomp
export discover_accelerator, check_accelerator, mna_decomp, mna_solve

using CUDA
using CUDA.CUSPARSE
using CUSOLVERRF

struct CUDAccelerator <: AbstractAccelerator 
    name::String
    properties::AcceleratorProperties


    function CUDAccelerator(name::String, properties=AcceleratorProperties(true, 1, 1.0, 1.0))
        new(name, properties)
    end


    function CUDAccelerator()
        new("cuda", AcceleratorProperties(true, 1, 1.0, 1.0))
    end

end

struct CUDA_LUdecomp <: AbstractLUdecomp 
    lu_decomp::CUSOLVERRF.RFLU
end

# function discover_accelerator(accelerator::CUDAccelerator) end
# function check_accelerator(accelerator::CUDAccelerator) end
# function estimate_flops(accelerator::CUDAccelerator) end

# function discover_accelerator()
#     return CUDAccelerator()
# end

function discover_accelerator(accelerators::Vector{AbstractAccelerator}, accelerator::CUDAccelerator) 

    devices = collect(CUDA.devices())   # Vector of CUDA devices 
    power_limits = readlines(`nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits`)
    power_limits = parse.(Float64, power_limits)  

    i = 1
    for dev in devices 
        cuda_acc = CUDAccelerator(CUDA.name(dev))
        cuda_flops = estimate_flops(dev)
        cuda_acc = CUDAccelerator(CUDA.name(dev), AcceleratorProperties(true, 1, cuda_flops, power_limits[i]))
        push!(accelerators, cuda_acc)
        i += 1
    end
    
end

function mna_decomp(sparse_mat, accelerator::CUDAccelerator)
    matrix = CuSparseMatrixCSR(CuArray(sparse_mat)) # Sparse GPU implementation
    lu_decomp = CUSOLVERRF.RFLU(matrix; symbolic=:RF) |> CUDA_LUdecomp


    return lu_decomp
end

function mna_solve(system_matrix::CUDA_LUdecomp, rhs, accelerator::CUDAccelerator)
    rhs_d = CuVector(rhs)
    ldiv!(system_matrix.lu_decomp, rhs_d)
    return Array(rhs_d)
end

function estimate_flops(dev::CUDA.CuDevice)   # returns flops in GFLOPs

    # devices = collect(CUDA.devices())
    # dev::CUDA.CuDevice = findfirst(device -> CUDA.name(device) == accelerator.name, devices)


    n_sms = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    clock_hz = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_CLOCK_RATE) * 1000 # in kHz

    cc = CUDA.capability(dev)
    cores_per_sm = get_cores_per_sm(cc)

    # compute theoretical FLOPs
    total_cores = n_sms * cores_per_sm
    flops = 2.0 * total_cores * clock_hz

    # println("Device: ", CUDA.name(dev))
    # println("Compute Capability: ", cc)
    # println("SMs: $n_sms, Cores/SM: $cores_per_sm, Total Cores: $total_cores")
    # println("Clock: $(clock_hz / 1e6) MHz")
    # println("Estimated FP64 peak: $(round(flops / 1e9, digits=2)) GFLOPs")

    return round(flops / 1e9, digits=2)
end

function get_cores_per_sm(cc::VersionNumber)
    # Add lookup cores per sm, check with 'CUDA.capability( 'your_CUDA_device'  )' for capability
    # and then in doc for num for 64FP cores
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities and
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions
    if cc == v"6.1" return 4  # Tesla P40
    elseif cc == v"7.5" return 32   # Tesla T4
    elseif cc == v"8.6" return 32 # NVIDIA A2
    else
        @warn "Unknown compute capability $cc; assuming 2 cores/SM"
        return 2
    end
end