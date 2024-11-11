# build.jl


import PackageCompiler

const build_dir = @__DIR__
const target_dir = ARGS[1]

# FIXME: Remove once the issue with PackageCompiler and Julia v1.11. is resolved.
# See: https://github.com/JuliaLang/PackageCompiler.jl/issues/990
delete!(ENV, "JULIA_NUM_THREADS")

println("Creating Julia MNA solver library in $target_dir")
PackageCompiler.create_library("$(build_dir)/..", target_dir;
                                lib_name="juliamna",
                                precompile_execution_file="$(@__DIR__)/precompile_statements.jl",
                                incremental=true,
                                filter_stdlibs=false,
                                header_files = ["$(@__DIR__)/juliamna.h"],
                                force=true
                            )