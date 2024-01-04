begin
    ENV["JULIA_DEBUG"] = "JuliaMNA"
    push!(LOAD_PATH, "/home/wege/Projects/dpsim-refactor/dpsim/src/SolverPlugins/julia/JuliaMNA")

    using Pkg
    Pkg.activate(".")
    Pkg.status()
    # Pkg.instantiate()

    using JuliaMNA
    using Profile

    struct ArrayPath path::String end
    struct VectorPath path::String end

    function read_input(path::ArrayPath)
        # Read system matrix from file
        system_matrix_strings = readlines(path.path)

        # Sanize strings
        system_matrix_strings = replace.(system_matrix_strings, r"[\[\],]" => "")

        # Convert system to dpsim_csr_matrix
        values = parse.(Float64, split(system_matrix_strings[1]))
        rowIndex = parse.(Cint, split(system_matrix_strings[2]))
        colIndex = parse.(Cint, split(system_matrix_strings[3]))

        system_matrix = dpsim_csr_matrix(
            Base.unsafe_convert(Ptr{Cdouble}, values),
            Base.unsafe_convert(Ptr{Cint}, rowIndex),
            Base.unsafe_convert(Ptr{Cint}, colIndex),
            parse(Int32, system_matrix_strings[4]),
            parse(Int32, system_matrix_strings[5])
        )

        return system_matrix
    end

    function read_input(path::VectorPath)
        # Reard right hand side vector from file
        rhs_vector_strings = readlines(path.path)

        # Sanize rhs strings and parse into Float64 vector
        rhs_vector_strings = replace.(rhs_vector_strings, r"[\[\],]" => "")
        rhs_vector = parse.(Float64, split(rhs_vector_strings[1]))
    end

    GC.enable(false) # We cannot be sure that system_matrix is garbage collected before the pointer is passed... 
    # system_matrix = read_input(ArrayPath("$(@__DIR__)/system_matrix_small.txt"))
    # system_matrix = read_input(ArrayPath("$(@__DIR__)/system_matrix.txt"))
    system_matrix = read_input(ArrayPath("$(@__DIR__)/system_matrix_big.txt"))
    # system_matrix = read_input(Arrpushfirst!(LOAD_PATH, raw"/home/wege/.vscode-server/extensions/julialang.language-julia-1.47.2/scripts/packages");using VSCodeServer;popfirst!(LOAD_PATH);VSCodeServer.serve(raw"/tmp/vsc-jl-repl-1d1d4224-9e3b-455e-95d3-507bb6e33452"; is_dev = "DEBUG_MODE=true" in Base.ARGS, crashreporting_pipename = raw"/tmp/vsc-jl-cr-f0548d33-3806-4c6f-96fa-786930490057");nothing # re-establishing connection with VSCodeayPath("test/system_matrix_small.txt"))
    system_matrix_ptr = pointer_from_objref(system_matrix)

    # @show unsafe_wrap(Array, system_matrix.values, system_matrix.nnz)

    # rhs_vector = read_input(VectorPath("$(@__DIR__)/rhs_small.txt"))
    # rhs_vector = read_input(VectorPath("$(@__DIR__)/rhs.txt"))
    rhs_vector = read_input(VectorPath("$(@__DIR__)/rhs_big.txt"))
    # rhs_vector = read_input(VectorPath("test/rhs_small.txt"))

    lhs_vector = zeros(Float64, length(rhs_vector))
    rhs_reset = ones(Float64, length(rhs_vector))
end
begin
    init(Base.unsafe_convert(Ptr{dpsim_csr_matrix}, system_matrix_ptr))
    GC.enable(true)

    # @time solve(Base.unsafe_convert(Ptr{Cdouble}, rhs_vector), Base.unsafe_convert(Ptr{Cdouble}, lhs_vector))
    # @profile solve(Base.unsafe_convert(Ptr{Cdouble}, rhs_vector), Base.unsafe_convert(Ptr{Cdouble}, lhs_vector))
end
@time solve(Base.unsafe_convert(Ptr{Cdouble}, rhs_reset), Base.unsafe_convert(Ptr{Cdouble}, lhs_vector))
begin
    @time solve(Base.unsafe_convert(Ptr{Cdouble}, rhs_vector), Base.unsafe_convert(Ptr{Cdouble}, lhs_vector))
end
begin
    @time decomp(Base.unsafe_convert(Ptr{dpsim_csr_matrix}, system_matrix_ptr))
end
## Profling
# @time begin
#     @profile begin
#         for i in 1:1000
#             solve(Base.unsafe_convert(Ptr{Cdouble}, rhs_vector), Base.unsafe_convert(Ptr{Cdouble}, lhs_vector))
#         end
#     end
# # end
# @time begin
#     begin
#         for i in 1:1000
#             solve(Base.unsafe_convert(Ptr{Cdouble}, rhs_vector), Base.unsafe_convert(Ptr{Cdouble}, lhs_vector))
#             # sleep(1)
#         end
#     end
# end

cpu = []
begin
    begin
        for i in 1:5000
            append!(cpu, @elapsed solve(Base.unsafe_convert(Ptr{Cdouble}, rhs_vector), Base.unsafe_convert(Ptr{Cdouble}, lhs_vector)))
            # sleep(1)
        end
    end
end

gpu = []
begin
    begin
        for i in 1:5000
            append!(gpu, @elapsed solve(Base.unsafe_convert(Ptr{Cdouble}, rhs_vector), Base.unsafe_convert(Ptr{Cdouble}, lhs_vector)))
            # sleep(1)
        end
    end
end

begin
    begin
        for i in 1:50
            append!(cpu, @elapsed solve(Base.unsafe_convert(Ptr{Cdouble}, rhs_vector), Base.unsafe_convert(Ptr{Cdouble}, lhs_vector)))
            # sleep(1)
        end
    end
end

cleanup()
# using StatProfilerHTML
# statprofilehtml()


# using BenchmarkTools
# @benchmark begin
#     for i in 1:1000
#         solve(Base.unsafe_convert(Ptr{Cdouble}, rhs_vector), Base.unsafe_convert(Ptr{Cdouble}, lhs_vector))
#     end
# end
