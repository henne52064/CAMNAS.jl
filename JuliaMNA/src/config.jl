export hwAwarenessDisabled
export allow_cpu, allow_gpu

function env(env_var, default)
    if haskey(ENV, env_var)
        @info env_var * " = " * ENV[env_var]
        parse(typeof(default), ENV[env_var])
    else
        @info env_var * " not set. Using default value: " * string(default)
        default
    end
end

varDict = Dict()
# Macro called @var that adds a variable defintion of type x = y to the varDict dictionary
#FIXME: Current implementation calls env() twice, once when registering the variable and once during eval(ex.args[2])
macro var(ex)
    varDict[String(ex.args[1])] = eval(ex.args[2])
    eval(ex)
end

function create_env_file()
    filepath = "$(@__DIR__)/system.env"
    @info filepath
    println("Creating system environment file at $filepath")
    open(filepath, "w") do f
        println(f, "# This file is machine-generated - editing it directly is not advised\n")
        for (k, v) in varDict
            println(f, "$k $v")
        end
    end
end

@var hwAwarenessDisabled = env("JL_MNA_DISABLE_AWARENESS", false)

@var force_gpu = env("JL_MNA_FORCE_GPU", false)
@var force_cpu = env("JL_MNA_FORCE_CPU", false)
(force_cpu && force_gpu) && error("Cannot force CPU and GPU at the same time.")

@var allow_gpu = env("JL_MNA_ALLOW_GPU", true)
@var allow_cpu = env("JL_MNA_ALLOW_CPU", true)
(!allow_cpu && !allow_gpu) && error("Cannot forbid CPU and GPU at the same time.")

create_env_file()
