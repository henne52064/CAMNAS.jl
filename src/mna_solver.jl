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



# Vector of available accelerators
global accelerators_vector = Vector{AbstractAccelerator}()
system_environment = Channel(1)
accelerator = NoAccelerator()
current_strategy = DefaultStrategy()
system_matrix = Vector{AbstractLUdecomp}(undef,2)   



function select_strategy(strategy::AbstractSelectionStrategy, accelerators_vector::Vector{AbstractAccelerator})
    global accelerators_vector
    @debug "Strategy not implemented, falling back to DefaultStrategy"
    select_accelerator(DefaultStrategy(), accelerators_vector)
end

function select_strategy(strategy::DefaultStrategy, accelerators_vector::Vector{AbstractAccelerator})
    # sort vector of accelerators to a specific order and then choose the first available
    global current_strategy = strategy
    
end

function select_strategy(strategy::LowestPowerStrategy, accelerators_vector::Vector{AbstractAccelerator})
    global accelerators_vector
    global current_strategy = strategy
    available = filter(x -> x.properties.availability, accelerators_vector)
    value, index = findmin(x -> x.properties.power_watts, available)
    set_accelerator!(available[index])
    


end

function select_strategy(strategy::HighestFlopsStrategy, accelerators_vector::Vector{AbstractAccelerator})
    global accelerators_vector
    global current_strategy = strategy
    available = filter(x -> x.properties.availability, accelerators_vector)
    value, index = findmax(x -> x.properties.flops, available)
    set_accelerator!(available[index])
    

end



function find_accelerator()
    global accelerators_vector

    Accelerators.load_all_accelerators(accelerators_vector)

    if !isempty(accelerators_vector) && varDict["allow_gpu"]
        accelerator = findfirst(x -> typeof(x) == CUDAccelerator, accelerators_vector)
    elseif !@isdefined accelerator
        @info "[CAMNAS] No accelerator found."
        accelerator = findfirst(x -> x.name == "cpu", accelerators_vector)
    end

    @debug "Present accelerators: $([a.name for a in accelerators_vector])"

    return accelerator

    # if varDict["allow_gpu"] && has_cuda()
    #     @debug "CUDA available! Try using CUDA accelerator..."
    #     try
    #         CuArray(ones(1))
    #         accelerator = CUDAccelerator()
    #         @info "[CAMNAS] CUDA driver available and CuArrays package loaded. Using CUDA accelerator..."
    #     catch e
    #         @warn "CUDA driver available but could not load CuArrays package."
    #     end
    # elseif !@isdefined accelerator
    #     @info "[CAMNAS] No accelerator found."
    #     accelerator = NoAccelerator()
    # end
    # @debug "Accelerator type is $(typeof(accelerator))"
    # return accelerator
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
        if varDict["allow_strategies"] && varDict["highest_flop_strategy"] && varDict["lowest_power_strategy"] || !(varDict["allow_strategies"])
            @debug "Selected DefaultStrategy"
            select_strategy(DefaultStrategy(), accelerators_vector)    

        elseif varDict["allow_strategies"] && varDict["highest_flop_strategy"]
            @debug "Selected HighestFlopsStrategy"
            select_strategy(HighestFlopsStrategy(), accelerators_vector)
        
        elseif varDict["allow_strategies"] && varDict["lowest_power_strategy"] 
            @debug "Selected LowestPowerStrategy"
            select_strategy(LowestPowerStrategy(), accelerators_vector)
        
        elseif varDict["allow_strategies"] #&& !(varDict["highest_flop_strategy"]) && !(varDict["lowest_power_strategy"])
            @debug "Selected DefaultStrategy"
            select_strategy(DefaultStrategy(), accelerators_vector)
        
        elseif varDict["force_cpu"] && varDict["force_gpu"]
        
            @debug "Conflict: Both 'force_cpu' and 'force_gpu' are set. Only one can be forced."
            idx = findfirst(x -> x.name == "cpu", accelerators_vector)
            typeof(accelerator) == NoAccelerator || set_accelerator!(accelerators_vector[idx])
        
        elseif varDict["allow_gpu"] && varDict["force_gpu"]
        
            idx = findfirst(x -> typeof(x) == CUDAccelerator, accelerators_vector)
            typeof(accelerator) == CUDAccelerator || Accelerators.set_accelerator!(accelerators_vector[idx])
            @debug "did this change? $(CUDA.device())"
        
        elseif varDict["allow_cpu"] && varDict["force_cpu"]
            idx = findfirst(x -> x.name == "dummy_accelerator", accelerators_vector)
            typeof(accelerator) == DummyAccelerator || set_accelerator!(accelerators_vector[idx])
        
        elseif varDict["allow_gpu"] 
        
            idx = findlast(x -> typeof(x) == CUDAccelerator, accelerators_vector)
            typeof(accelerator) == CUDAccelerator || Accelerators.set_accelerator!(accelerators_vector[idx])
            @debug "did this change? $(CUDA.device())"
        
        elseif varDict["allow_cpu"]
        
            idx = findfirst(x -> x.name == "cpu", accelerators_vector)
            typeof(accelerator) == NoAccelerator || set_accelerator!(accelerators_vector[idx])
        
        else
            @debug "Conflict: Nothing is allowed. THIS DOESNT MAKE SENSE!"
        end

        @info "[CAMNAS] Currently used accelerator: $accelerator" 
        if varDict["allow_strategies"]
            @info "[CAMNAS] Currently used strategy: $(typeof(current_strategy))"
        end
    end
    @debug "Accelerator determination stopped!"
end

function set_accelerator!(acc::AbstractAccelerator)
    @debug "Setting accelerator to: $(typeof(acc))"
    global accelerator = acc
end

# Housekeeping
function mna_init(sparse_mat)
    global varDict = parse_env_vars()
    create_env_file()
    #Accelerators.load_all_accelerators(accelerators_vector)

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
    
    @debug "Type of lu_decomp is $(typeof(lu_decomp))"
    return cpu_lu_decomp
end
