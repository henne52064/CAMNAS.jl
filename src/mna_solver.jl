using SparseArrays
using CUDA.CUSPARSE
using CUSOLVERRF
using LinearAlgebra
using FileWatching
using TOML

include("config.jl")


# Hardwareawareness
abstract type AbstractAccelerator end

struct AcceleratorProperties
    availability::Bool
    priority::Int64
    flops::Float64      # in GFLOPs
    #memory_gb::Float64
    #memory_bandwith_gbps::Float64
    #stability_rating::Float64       # 0.0-1.0
    power_watts::Int64            # max Power usage
    energy_efficiency::Float64      # flops/W

    function AcceleratorProperties(availability::Bool, priority::Int64, flops::Float64, power_watts::Int64) 
        new(availability, priority, flops, power_watts, round(flops/power_watts, digits=4))
    end

    function AcceleratorProperties()
        new(true, 1, 1.0, 1, 1.0)
    end

end

struct NoAccelerator <: AbstractAccelerator 
    name::String
    properties::AcceleratorProperties
    

    function NoAccelerator(name::String; properties=AcceleratorProperties(true, 1, 1.0, 1))
        new(name, properties)
    end

    
    function NoAccelerator()
        new("cpu", AcceleratorProperties(true, 1, 1.0, 1))
    end

end
struct CUDAccelerator <: AbstractAccelerator 
    name::String
    properties::AcceleratorProperties

    function CUDAccelerator(name::String; properties=AcceleratorProperties(true, 1, 1.0, 1))
        new(name, properties)
    end

    function CUDAccelerator()
        new("cuda", AcceleratorProperties(true, 1, 1.0, 1))
    end

end
struct DummyAccelerator <: AbstractAccelerator
    name::String
    properties::AcceleratorProperties

    function DummyAccelerator(name::String; properties=AcceleratorProperties(true, 1, 1.0, 1))
        new(name, properties)
    end

    function DummyAccelerator()
        new("", AcceleratorProperties(true, 1, 1.0, 1))
    end

end


# Accelerator Selection
abstract type AbstractSelectionStrategy end

struct DefaultStrategy <: AbstractSelectionStrategy end #choose first accelerator from a defined order
struct LowestPowerStrategy <: AbstractSelectionStrategy end
struct HighestFlopsStrategey <: AbstractSelectionStrategy end

acceleratorPropertiesDict = Dict()

# LU Struct
abstract type AbstractLUdecomp end

struct CUDA_LUdecomp <: AbstractLUdecomp 
    lu_decomp::CUSOLVERRF.RFLU
end
struct CPU_LUdecomp <: AbstractLUdecomp 
    lu_decomp::LinearAlgebra.TransposeFactorization
end
struct DummyLUdecomp <: AbstractLUdecomp
end

# Vector of available accelerators
global accelerators = Vector{AbstractAccelerator}()
system_environment = Channel(1)
accelerator = NoAccelerator()
#system_matrix = nothing
#system_matrix = Vector{Any}(undef,2)
system_matrix = Vector{AbstractLUdecomp}(undef,2)   

function load_accelerator_properties()
    @debug "Reading accelerators.toml"
    content = TOML.parsefile("test/accelerators.toml")

    for (name, data) in content
        global acceleratorPropertiesDict["$name"] = AcceleratorProperties(
            data["available"],
            data["priority"],
            data["flops"],
            #data["memory_gb"],
            #data["memory_bandwidth_gbps"],
            #data["stability_rating"],
            data["power_watts"]
        )
    end 
    @debug "Stored properties for all accelerators:\n$(join(["$name => $(acceleratorPropertiesDict[name])" for name in keys(acceleratorPropertiesDict)], "\n"))"
end

function select_accelerator(strategy::AbstractSelectionStrategy, accelerators::Vector{AbstractAccelerator})
    global accelerators
    @debug "Strategy not implemented, falling back to DefaultStrategy"
    select_accelerator(DefaultStrategy(), accelerators)
end

function select_accelerator(strategy::DefaultStrategy, accelerators::Vector{AbstractAccelerator})
    # sort vector of accelerators to a specific order and then choose the first available

    
end

function select_accelerator(strategy::LowestPowerStrategy, accelerators::Vector{AbstractAccelerator})
    global accelerators
    available = filter(x -> x.properties.availability, accelerators)
    value, index = findmin(x -> x.properties.power_watts, available)
    accelerator = available[index]


end

function select_accelerator(strategy::HighestFlopsStrategey, accelerators::Vector{AbstractAccelerator})
    global accelerators
    available = filter(x -> x.properties.availability, accelerators)
    value, index = findmax(x -> x.properties.flops, available)
    accelerator = available[index]

end





function estimate_cpu_flops(; float_bits::Int = 64)     # returns flops in GFLOPs
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

function estimate_cuda_fp64_flops(dev::CUDA.CuDevice = CUDA.device())   # returns flops in GFLOPs

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

function setup_accelerators()
    global accelerators
    cpu_flops = estimate_cpu_flops()
    cpu = NoAccelerator("cpu", properties = AcceleratorProperties(true, 1, cpu_flops, 95)) # not a direct way from Julia to get CPU TDP
    push!(accelerators, cpu)

    
    gpu_flops = estimate_cuda_fp64_flops(CuDevice(0))
    gpu_p40 = CUDAccelerator("P40", properties = AcceleratorProperties(true, 1, gpu_flops, 250))
    push!(accelerators, gpu_p40)

    gpu_flops = estimate_cuda_fp64_flops(CuDevice(1))
    gpu_t4 = CUDAccelerator("T4", properties = AcceleratorProperties(true, 1, gpu_flops, 70))
    push!(accelerators, gpu_t4)

    gpu_flops = estimate_cuda_fp64_flops(CuDevice(2))
    gpu_a2 = CUDAccelerator("A2", properties = AcceleratorProperties(true, 1, gpu_flops, 60))
    push!(accelerators, gpu_a2)

    gpu_flops = estimate_cuda_fp64_flops(CuDevice(3))
    gpu_p40_2 = CUDAccelerator("P40_2", properties = AcceleratorProperties(true, 1, gpu_flops, 250))
    push!(accelerators, gpu_p40_2)


    # Since powerconsumption not readable from system info, might add later with NVML
    # for dev in CUDA.devices()
    #     gpu_flops = estimate_cuda_fp64_flops(dev)
    #     gpu = CUDAccelerator(properties = AcceleratorProperties(true, 1, gpu_flops, 70))
    #     push!(gpu)


end

function find_accelerator()
    global accelerators
    @debug "Present accelerators: $([a.name for a in accelerators])"
    if varDict["allow_gpu"] && has_cuda()
        @debug "CUDA available! Try using CUDA accelerator..."
        try
            CuArray(ones(1))
            accelerator = CUDAccelerator()
            @info "[CAMNAS] CUDA driver available and CuArrays package loaded. Using CUDA accelerator..."
        catch e
            @warn "CUDA driver available but could not load CuArrays package."
        end
    elseif !@isdefined accelerator
        @info "[CAMNAS] No accelerator found."
        accelerator = NoAccelerator()
    end
    @debug "Accelerator type is $(typeof(accelerators))"
    accelerator = select_accelerator(HighestFlopsStrategey(), accelerators)
    @debug "Lowest power consumption with $accelerator as accelerator"
    return accelerator
end


function systemcheck()
    if varDict["hwAwarenessDisabled"]
        @info "[CAMNAS] Hardware awareness disabled... Using Fallback implementation"
        return NoAccelerator()
    else
        return find_accelerator()
    end
end


function file_watcher()
    file_system_env = (@__DIR__)*"/system.env"
    @debug "Watching sytem environment at : $file_system_env"
    global run
    while run
        # @debug "Waiting for file change..."
        fw = watch_file(file_system_env, 3)
        if fw.changed
            @debug "Filewatcher triggered!"
            content = read(file_system_env, String)
            if isready(system_environment)
                take!(system_environment)
            end

            put!(system_environment, content)
            @debug "System environment updated!"
        end
    end
    @debug "File watcher stopped!"
end

function determine_accelerator()
    global accelerator
    while true
        val = take!(system_environment)
        @debug "Received new system environment!: $val"

        for line in split(val, '\n')[2:end]
            if length(line) == 0
                continue
            end
            key, value = split(line)
            if key == "allow_cpu"
                allow_cpu = parse(Bool, value)
                varDict["allow_cpu"] = allow_cpu
            elseif key == "allow_gpu"
                allow_gpu = parse(Bool, value)
                varDict["allow_gpu"] = allow_gpu
            else
                varDict[key] = parse(Bool, value)
            end
        end

        @debug "Allow CPU is: $(varDict["allow_cpu"])"
        @debug "Allow GPU is: $(varDict["allow_gpu"])"
        @debug "$varDict"

        # Stop accelerator determination if nothing-value is received
        val === nothing ? break : nothing

        # Currently, we implement a GPU-favoring approach
        if varDict["force_cpu"] && varDict["force_gpu"]
            @debug "Conflict: Both 'force_cpu' and 'force_gpu' are set. Only one can be forced."
            typeof(accelerator) == NoAccelerator || set_accelerator!(NoAccelerator())
        elseif varDict["allow_cpu"] && varDict["force_cpu"]
            typeof(accelerator) == DummyAccelerator || set_accelerator!(DummyAccelerator())
        elseif varDict["allow_gpu"] && has_cuda() 
            typeof(accelerator) == CUDAccelerator || set_accelerator!(CUDAccelerator())
        elseif varDict["allow_cpu"]
            typeof(accelerator) == DummyAccelerator || set_accelerator!(DummyAccelerator())
        else
            typeof(accelerator) == NoAccelerator || set_accelerator!(NoAccelerator())
        end

        @info "[CAMNAS] Currently used accelerator: $(typeof(accelerator))"
        if typeof(accelerator) == CUDAccelerator 
            index = CUDA.device().handle
            @debug "Current CUDA accelerator: $(CUDA.name(CuDevice(index)))"
        end
    end
    @debug "Accelerator determination stopped!"
end

function set_accelerator!(acc)
    @debug "Setting accelerator to: $(typeof(acc))"
    global accelerator = acc
end

# Housekeeping
function mna_init(sparse_mat)
    global varDict = parse_env_vars()
    create_env_file()
    setup_accelerators()
    #load_accelerator_properties()

    global accelerator = systemcheck()
    #global accelerators = [k for (k, v) in acceleratorPropertiesDict if v.availability]
    @debug accelerators
    global run = true
    global csr_mat = sparse_mat

    global fw = Threads.@spawn file_watcher()
    global da = Threads.@spawn determine_accelerator()
end

function mna_cleanup()
    global run = false

    wait(fw)
    put!(system_environment, nothing) # Signal to stop
    wait(da)
    close(system_environment)
    @debug "Cleanup done!"
end

# Solving Logic
function set_csr_mat(csr_matrix)
    global csr_mat = csr_matrix
end

function mna_decomp(sparse_mat, accelerator::AbstractAccelerator)
    lu_decomp = SparseArrays.lu(sparse_mat) |> CPU_LUdecomp
    @debug "CPU $lu_decomp"
    return lu_decomp
end

function mna_decomp(sparse_mat, accelerator::DummyAccelerator)
    lu_decomp = SparseArrays.lu(sparse_mat) |> CPU_LUdecomp
    @debug "Dummy"
    return lu_decomp
end

function mna_decomp(sparse_mat, accelerator::CUDAccelerator)
    matrix = CuSparseMatrixCSR(CuArray(sparse_mat)) # Sparse GPU implementation
    lu_decomp = CUSOLVERRF.RFLU(matrix; symbolic=:RF) |> CUDA_LUdecomp
    # @debug "Befor Transfer"
    # lu_decomp_cpu = mna_transfer(lu_decomp)
    # @debug "Transfer worked: $lu_decomp_cpu"

    return lu_decomp
end

function mna_decomp(sparse_mat)
    set_csr_mat(sparse_mat)
    if varDict["runtime_switch"]
        return [mna_decomp(sparse_mat, NoAccelerator()), mna_decomp(sparse_mat, CUDAccelerator())]
    elseif accelerator == CUDAccelerator()
        return [DummyLUdecomp(), mna_decomp(sparse_mat, accelerator)]
        #sys_mat = Vector{AbstractLUdecomp}(DummyLUdecomp(), mna_decomp(sparse_mat, accelerator))
        #return sys_mat
        #return [missing, mna_decomp(sparse_mat, accelerator)]
    else
        return [mna_decomp(sparse_mat, accelerator), missing]
    end
end


function mna_solve(system_matrix, rhs, accelerator::AbstractAccelerator)
    return system_matrix.lu_decomp \ rhs
end

function mna_solve(system_matrix, rhs, accelerator::CUDAccelerator)
    rhs_d = CuVector(rhs)
    ldiv!(system_matrix.lu_decomp, rhs_d)
    return Array(rhs_d)
end

function mna_solve(my_system_matrix, rhs)

    # Allow printing accelerator without debug statements
    (haskey(ENV, "JL_MNA_PRINT_ACCELERATOR") && ENV["JL_MNA_PRINT_ACCELERATOR"] == "true" ?
        println(typeof(accelerator))
        : nothing)
    (typeof(accelerator) == CUDAccelerator) ? sys_mat = my_system_matrix[2] : sys_mat = my_system_matrix[1]

    return mna_solve(sys_mat, rhs, accelerator)
end
mna_solve(system_matrix, rhs, accelerator::DummyAccelerator) = mna_solve(system_matrix, rhs, NoAccelerator())

function transfer_LU_CUDA2CPU(cuda_lu::CUDA_LUdecomp) #transfer LU factorization from CUSOLVERRF.RFLU to SparseArrays.UMFPACK.UMFPACKLU type
    # Access combined LU matrix (GPU, CSR format)
    M_gpu = cuda_lu.lu_decomp.M    # M = L + U
    
    rowPtr = collect(M_gpu.rowPtr)
    colVal = collect(M_gpu.colVal)
    nzVal = collect(M_gpu.nzVal)

    nrow = size(M_gpu, 1)
    ncol = size(M_gpu, 2)

    # Construct CPU-side sparse matrix in CSR format
    M_cpu = SparseMatrixCSR{1}(nrow, ncol, rowPtr, colVal, nzVal)   # 1 indicates index base
    cpu_lu_decomp = SparseArrays.lu(M_cpu) |> CPU_LUdecomp
    #cpu_lu_decomp = CPU_LUdecomp(M_cpu)
    @debug "Type of lu_decomp is $(typeof(lu_decomp))"
    return cpu_lu_decomp
end

# function transfer_LU_CUDA2CPU(rf::CUSOLVERRF.RFLU) #transfer LU factorization from CUSOLVERRF.RFLU to SparseArrays.UMFPACK.UMFPACKLU type
#     # Access combined LU matrix (GPU, CSR format)
#     M_gpu = rf.M    # M = L + U
    
#     # Transfer to CPU
#     @debug "typeof collect(M_gpu): $(typeof(collect(M_gpu)))"
#     M_cpu = SparseMatrixCSR(collect(M_gpu))  # -> added SparseMatricesCSR.jl, otherwise in CSC Format
#     @debug "Type of M_gpu is $(typeof(M_gpu)), typeof M_cpu is $(typeof(M_cpu))"
#     lu_decomp = SparseArrays.lu(M_cpu)
#     @debug "Type of lu_decomp is $(typeof(lu_decomp))"
#     return lu_decomp
# end
