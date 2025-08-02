export CUDAccelerator, CUDAccelerator_LUdecomp
export discover_accelerator, check_accelerator, mna_decomp, mna_solve

using CUDA
using CUDA.CUSPARSE
using CUSOLVERRF

using Printf

struct CUDAccelerator <: AbstractAccelerator 
    name::String
    properties::AcceleratorProperties
    device::CuDevice


    function CUDAccelerator(name::String, dev::CuDevice, properties=AcceleratorProperties(true, 1, 1.0, floatmax()))
        new(name, properties, dev)
    end

    CUDAccelerator() = new()

end

struct CUDAccelerator_LUdecomp <: AbstractLUdecomp 
    lu_decomp::CUSOLVERRF.RFLU
end

function has_driver(accelerator::CUDAccelerator)
    try
        CUDA.has_cuda()
    catch e
        @error "CUDA driver not found: $e"
        return false
    end
    return true
end

function discover_accelerator(accelerators::Vector{AbstractAccelerator}, accelerator::CUDAccelerator) 

    try
        has_driver(CUDAccelerator())
    catch e
        @error "CUDA driver not found: $e"
        return
    end

    devices = collect(CUDA.devices())   # Vector of CUDA devices 


    for dev in devices 
        cuda_acc = CUDAccelerator(CUDA.name(dev), dev)
        power_limit = get_tdp(cuda_acc)
        cuda_flops = estimate_flops(dev)
        cuda_acc = CUDAccelerator(CUDA.name(dev), dev, AcceleratorProperties(true, 1, cuda_flops, power_limit))
        push!(accelerators, cuda_acc)
    end
    
end

function mna_decomp(sparse_mat, accelerator::CUDAccelerator)
    @debug "Calculate Decomposition on $(CUDA.device()) on Thread $(Threads.threadid())"
    @debug "Calculating on $(accelerator.name)"
    matrix = CuSparseMatrixCSR(CuArray(sparse_mat)) # Sparse GPU implementation
    lu_decomp = CUSOLVERRF.RFLU(matrix; symbolic=:RF) |> CUDAccelerator_LUdecomp


    return lu_decomp
end

function mna_solve(system_matrix::CUDAccelerator_LUdecomp, rhs, accelerator::CUDAccelerator)
    @debug "Calculate Solve step on $(CUDA.device())"
    rhs_d = CuVector(rhs)
    ldiv!(system_matrix.lu_decomp, rhs_d)
    return Array(rhs_d)
end

function estimate_flops(dev::CUDA.CuDevice)   # returns flops in GFLOPs

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

function get_tdp(accelerator::CUDAccelerator)

    mapping = map_CuDevice_to_nvidiasmi()
    cuda_device_id = accelerator.device.handle

    cmd = `nvidia-smi -i $(mapping[cuda_device_id]) --query-gpu=power.limit --format=csv,noheader,nounits`
    power_limit = readlines(cmd)
    power_limit = parse(Float64, power_limit[1]) 
    return power_limit
end

# function set_accelerator!(acc::CUDAccelerator)
#     @debug "THIS CUDA.device!() is called on $(Threads.threadid()) before: $(CUDA.device())"
#     CUDA.device!(acc.device)
#     @debug "Current CUDA device is $(CUDA.device())"
#     CAMNAS.accelerator = acc

# end

function set_acceleratordevice!(acc::CUDAccelerator)
    # This function is used to set the CUDA device for the current thread
    # It is called by the CAMNAS.jl module to ensure that the correct device is used

    if acc.device == CUDA.device()
        @debug "CUDA device $(acc.device) is already set on Thread $(Threads.threadid())"
        return
    end


    @debug "Setting CUDA device to $(acc.device) on Thread $(Threads.threadid())"
    CUDA.device!(acc.device)
    @debug "Current CUDA device is now $(CUDA.device())"
end

function map_CuDevice_to_nvidiasmi()
    # collect CuDevice PCI bus IDs
    cuda_devices = Dict{Int, String}()
    for i in 0:length(CUDA.devices()) - 1
        dev = CuDevice(i)
        pci = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_PCI_BUS_ID)
        pci_hex = @sprintf("%02x" , pci) |> uppercase
        cuda_devices[i] = pci_hex
    end

    # collect nvidia-smi device list with PCI and IDs
    smi_output = readlines(`nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader`)
    smi_devices = Dict{String, Int}()
    for line in smi_output
        #isempty(strip(line)) && continue
        idx, pci_full = split(strip(line), ',')
        idx = parse(Int, strip(idx))
        pci_bus_id = strip(pci_full)[10:11]  # extrace "XX" from "00000000:XX:00.0"
        smi_devices[pci_bus_id] = idx
    end

    mapping = Dict{Int, Int}()
    for (i, pci) in cuda_devices
        if haskey(smi_devices, pci)
            mapping[i] = smi_devices[pci]
        else
            @warn "No matching nvidia-smi device found for CuDevice($i) (PCI $pci)"
        end
    end

    return mapping

end