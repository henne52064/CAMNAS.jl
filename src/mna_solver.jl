using SparseArrays
using CUDA.CUSPARSE
using CUSOLVERRF
using LinearAlgebra
using FileWatching
using TOML

include("accelerators/Accelerators.jl")
include("config.jl")


using .Accelerators






# Accelerator Selection
abstract type AbstractSelectionStrategy end

struct DefaultStrategy <: AbstractSelectionStrategy end #choose first accelerator from a defined order
struct LowestPowerStrategy <: AbstractSelectionStrategy end
struct HighestFlopsStrategy <: AbstractSelectionStrategy end

acceleratorPropertiesDict = Dict()

# LU Struct






# Vector of available accelerators
global accelerators_vector = Vector{AbstractAccelerator}()
system_environment = Channel(1)
accelerator = NoAccelerator()
#system_matrix = nothing
#system_matrix = Vector{Any}(undef,2)
system_matrix = Vector{AbstractLUdecomp}(undef,2)   



function select_strategy(strategy::AbstractSelectionStrategy, accelerators_vector::Vector{AbstractAccelerator})
    global accelerators_vector
    @debug "Strategy not implemented, falling back to DefaultStrategy"
    select_accelerator(DefaultStrategy(), accelerators_vector)
end

function select_strategy(strategy::DefaultStrategy, accelerators_vector::Vector{AbstractAccelerator})
    # sort vector of accelerators to a specific order and then choose the first available

    
end

function select_strategy(strategy::LowestPowerStrategy, accelerators_vector::Vector{AbstractAccelerator})
    global accelerators_vector
    available = filter(x -> x.properties.availability, accelerators_vector)
    value, index = findmin(x -> x.properties.power_watts, available)
    accelerator = available[index]


end

function select_strategy(strategy::HighestFlopsStrategy, accelerators_vector::Vector{AbstractAccelerator})
    global accelerators_vector
    available = filter(x -> x.properties.availability, accelerators_vector)
    value, index = findmax(x -> x.properties.flops, available)
    accelerator = available[index]

end









function setup_accelerators()
    global accelerators_vector
    @debug "type of accelerators_vector $(typeof(accelerators_vector))"
    cpu_flops = Accelerators.estimate_flops()
    cpu = NoAccelerator("cpu", properties = AcceleratorProperties(true, 1, cpu_flops, 95)) # not a direct way from Julia to get CPU TDP
    push!(accelerators_vector, cpu)

    
    
    gpu_flops = Accelerators.estimate_flops(CuDevice(0))
    gpu_p40 = CUDAccelerator("P40", properties = AcceleratorProperties(true, 1, gpu_flops, 250))
    push!(accelerators_vector, gpu_p40)
    @debug "2nd time: type of accelerators_vector $(typeof(accelerators_vector))"


    gpu_flops = Accelerators.estimate_flops(CuDevice(1))
    gpu_t4 = CUDAccelerator("T4", properties = AcceleratorProperties(true, 1, gpu_flops, 70))
    push!(accelerators_vector, gpu_t4)

    gpu_flops = Accelerators.estimate_flops(CuDevice(2))
    gpu_a2 = CUDAccelerator("A2", properties = AcceleratorProperties(true, 1, gpu_flops, 60))
    push!(accelerators_vector, gpu_a2)

    gpu_flops = Accelerators.estimate_flops(CuDevice(3))
    gpu_p40_2 = CUDAccelerator("P40_2", properties = AcceleratorProperties(true, 1, gpu_flops, 250))
    push!(accelerators_vector, gpu_p40_2)


    # Since powerconsumption not readable from system info, might add later with NVML
    # for dev in CUDA.devices()
    #     gpu_flops = estimate_cuda_fp64_flops(dev)
    #     gpu = CUDAccelerator(properties = AcceleratorProperties(true, 1, gpu_flops, 70))
    #     push!(gpu)


end

function find_accelerator()
    global accelerators_vector
    @debug "Present accelerators: $([a.name for a in accelerators_vector])"
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
    @debug "Accelerator type is $(typeof(accelerator))"
    accelerator = select_strategy(LowestPowerStrategy(), accelerators_vector)
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
    global accelerators_vector
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
    Accelerators.load_all_accelerators(accelerators_vector)

    global accelerator = systemcheck()
    #global accelerators = [k for (k, v) in acceleratorPropertiesDict if v.availability]
    @debug accelerators_vector
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






function mna_decomp(sparse_mat)
    set_csr_mat(sparse_mat)
    if varDict["runtime_switch"]
        return [Accelerators.mna_decomp(sparse_mat, NoAccelerator()), Accelerators.mna_decomp(sparse_mat, CUDAccelerator())]
    elseif accelerator == CUDAccelerator()
        return [DummyLUdecomp(), Accelerators.mna_decomp(sparse_mat, accelerator)]
        #sys_mat = Vector{AbstractLUdecomp}(DummyLUdecomp(), mna_decomp(sparse_mat, accelerator))
        #return sys_mat
        #return [missing, mna_decomp(sparse_mat, accelerator)]
    else
        return [Accelerators.mna_decomp(sparse_mat, accelerator), missing]
    end
end





function mna_solve(my_system_matrix, rhs)

    # Allow printing accelerator without debug statements
    (haskey(ENV, "JL_MNA_PRINT_ACCELERATOR") && ENV["JL_MNA_PRINT_ACCELERATOR"] == "true" ?
        println(typeof(accelerator))
        : nothing)
    (typeof(accelerator) == CUDAccelerator) ? sys_mat = my_system_matrix[2] : sys_mat = my_system_matrix[1]

    return Accelerators.mna_solve(sys_mat, rhs, accelerator)
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
