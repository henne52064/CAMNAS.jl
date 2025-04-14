# CAMNAS.jl

**C**ontext-**A**ware **M**odified **N**odal **A**nalysis **S**olver Plugin for the Dynamic Power System Simulator [DPsim](https://github.com/sogno-platform/dpsim).

CAMNAS is a simple Julia implementation of a Modified Nodal Analysis (MNA) solver designed for integration as a solver plugin in DPsim.
Its key features include:
- Automatic detection of NVidia GPU accelerators
- Dynamic switching of accelerators between time steps
- Basic rule-based automation for accelerator selection during runtime (*under development*)
- Asynchronous system monitoring/decision-making for accelerator selection (*under development*)

## Building
```
git clone https://github.com/RWTH-ACS/CAMNAS.jl.git
cd CAMNAS.jl
julia --project=$(pwd) --eval="using Pkg;Pkg.instantiate()"
make
```

## Dependencies
Build Depedencies:
- `julia>=1.10`
- `gcc`
- `make`
- ...

For an overview of the required Julia package depdendencies, see the `[deps]` section in [Project.toml](CAMNAS/Project.toml#6).

## Usage

```
<dpsim-scenario> -U "Plugin" -P "camnasjl"
```

To allow CAMNAS and DPsim to dynamically link the necessary libraries, ensure that your `LD_LIBRARY_PATH` includes both the CAMNAS base directory and `CAMNASCompiled/lib/`.

For more information on the interaction of DPsim and the Plugin, see [DPsim's MNASolverPlugin.cpp](https://github.com/sogno-platform/dpsim/blob/master/dpsim/src/MNASolverPlugin.cpp).

### Environment Variables
Certain features and behavior patterns of the solver can be managed through environment variables during the initialization of the plugin. For details, refer to the following:

| Variable | Values [default] | [TYPE] Description | 
| :--: | :--: | :-- |
|JL_MNA_DISABLE_AWARENESS|Boolean [false]| [CONTROL] Toggle accelerater detection/awareness. When disabled, calculations will always fallback to CPU.|
|JL_MNA_ALLOW_CPU|Boolean [true]| [CONTROL] Allow accelerator selection to use CPU. |
|JL_MNA_ALLOW_GPU|Boolean [true]|[CONTROL] Allow accelerator selection to use CPU. |
|JL_MNA_FORCE_CPU|Boolean [false]|[CONTROL] *Currently unused...* |
|JL_MNA_FORCE_GPU|Boolean [false]|[CONTROL] *Currently unused...*|
|JL_MNA_PRINT_ACCELERATOR| Boolean [false]| [DEBUG] Print currently used accelerator independent from debug statements.|

> [!Warning]
> To modify the plugin's behavior interactively during runtime (e.g., dynamically switching accelerators between timesteps), users can edit the automatically generated `system.env` file. However, this feature is still experimental and therefore not recommended and requires to set the optional ENV variable `JL_MNA_RUNTIME_SWITCH` before loading CAMNAS.


## Development / Testing
Developing or testing a shared library or plugin can be tideous, especially due to long compilations times. However, the purely Julia script `test/test_interface.jl` comes to your help!

This script allows testing all externally accessible functions directly within Julia.

To ensure compatibility with DPsim, three test case inputs of different sizes are provided. The specific input size can be adjusted using the `inputSize` variable at the beginning of the script. All inputs are designed to satisfy the equation `x = A \ b`, where `A` is a sparse matrix in CSR format and `b` is a vector.


## Author
- Felix Wege [fwege@eonerc.rwth-aachen.de](mailto:fwege@eonerc.rwth-aachen.de)

[Institute for Automation of Complex Power Systems (ACS) EON Energy Research Center (EONERC) RWTH University Aachen, Germany](http://www.acs.eonerc.rwth-aachen.de/)
