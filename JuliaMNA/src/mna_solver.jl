using SparseArrays
using CUDA.CUSPARSE
using CUSOLVERRF
using LinearAlgebra
using FileWatching

include("config.jl")

# Hardwareawareness
struct AbstractAccelerator end
struct CUDAccelerator end
struct DummyAccelerator end

system_environment = Channel(1)
accelerator = AbstractAccelerator()
system_matrix = nothing


function find_accelerator()
    # @info ENV
    # CUDA Accelerator
    if allow_gpu && has_cuda()
        @debug "CUDA available! Try using CUDA accelerator..."
        try
            CuArray(ones(1))
            # push!(accelerators, CUDAccelerator())
            accelerator = CUDAccelerator()
            @info "CUDA driver available and CuArrays package loaded. Using CUDA accelerator..."
        catch e
            @warn "CUDA driver available but could not load CuArrays package."
        end
    elseif !@isdefined accelerator
        @info "No accelerator found."
        accelerator = AbstractAccelerator()
    end
    return accelerator
end

function systemcheck()
    if hwAwarenessDisabled
        @info "Hardware awareness disabled... Using Fallback implementation"
        return AbstractAccelerator()
    else
        return find_accelerator()
    end
end


function file_watcher()
    file_system_env = (@__DIR__)*"/system.env"
    @debug "Watching sytem environment at : $file_system_env"
    global run
    while run
        @debug "Waiting for file change..."
        fw = watch_file(file_system_env, 3)
        if fw.changed
            @debug "Filewatcher triggered!"
            content = read(file_system_env, String)
            if isready(system_environment)
                take!(system_environment)
            end
            put!(system_environment, content)
        end
    end
    @debug "File watcher stopped!"
    # @info "File watcher stopped!"
end

function determine_accelerator()
    global accelerator
    while true
        val = take!(system_environment)
        @debug "Received new system environment!"

        val === nothing ? break : nothing

        if typeof(accelerator) == DummyAccelerator
            set_accelerator!(AbstractAccelerator())
            # accelerator = AbstractAccelerator()
        elseif typeof(accelerator) == AbstractAccelerator
            set_accelerator!(CUDAccelerator())
            # accelerator = DummyAccelerator()
        elseif typeof(accelerator) == CUDAccelerator
            set_accelerator!(AbstractAccelerator())
            # accelerator = DummyAccelerator()
        end
        @debug "Accelerator changed to: $(typeof(accelerator))"
    end
    @debug "Accelerator determination stopped!"
end

function set_accelerator!(acc)
    global system_matrix = mna_decomp(csr_mat, acc)
    global accelerator = acc
end

# Housekeeping
function mna_init(sparse_mat)
    global accelerator = systemcheck()
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
end

# Solving Logic
function set_csr_mat(csr_matrix)
    global csr_mat = csr_matrix
end

function mna_decomp(sparse_mat, accelerator::AbstractAccelerator)
    lu_decomp = SparseArrays.lu(sparse_mat)
    @info "CPU"
    return lu_decomp
end

function mna_decomp(sparse_mat, accelerator::DummyAccelerator)
    lu_decomp = SparseArrays.lu(sparse_mat)
    @info "Dummy"
    return lu_decomp
end

function mna_decomp(sparse_mat, accelerator::CUDAccelerator)
    matrix = CuSparseMatrixCSR(CuArray(sparse_mat)) # Sparse GPU implementation
    lu_decomp = CUSOLVERRF.RFLU(matrix; symbolic=:RF)
    @info "GPU"
    return lu_decomp
end

function mna_decomp(sparse_mat)
    set_csr_mat(sparse_mat)
    return mna_decomp(sparse_mat, accelerator)
end

function mna_solve(system_matrix, rhs, accelerator::AbstractAccelerator)
    return system_matrix \ rhs
end

function mna_solve(system_matrix, rhs, accelerator::CUDAccelerator)
    rhs_d = CuVector(rhs)
    ldiv!(system_matrix, rhs_d)
    return Array(rhs_d)
end
# mna_solve(system_matrix, rhs) = mna_solve(system_matrix, rhs, accelerator)
function mna_solve(system_matrix, rhs)
    println(typeof(accelerator))
    return mna_solve(system_matrix, rhs, accelerator)
end
mna_solve(system_matrix, rhs, accelerator::DummyAccelerator) = mna_solve(system_matrix, rhs, AbstractAccelerator())