# build.jl


import PackageCompiler

const build_dir = @__DIR__
const target_dir = ARGS[1]

println("Creating Julia MNA solver library in $target_dir")
PackageCompiler.create_library("$(build_dir)/..", target_dir;
                                lib_name="juliamna",
                                precompile_execution_file="$(@__DIR__)/precompile_statements.jl",
                                incremental=true,
                                filter_stdlibs=true,
                                header_files = ["$(@__DIR__)/juliamna.h"],
                                force=true
                            )