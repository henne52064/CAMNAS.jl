# CAMNAS.jl

**C**ontext-**A**ware **M**odified **N**odal **A**nalysis **S**olver Plugin for the Dynamic Phasor Power System Simulator [DPsim](https://github.com/sogno-platform/dpsim).

CAMNAS is a simple Julia implementation of a Modified Nodal Analysis (MNA) solver designed for integration as a solver plugin in DPsim.
Its key features include:
- Automatic detection of NVidia GPU accelerators
- Dynamic switching of accelerators between time steps
- Basic rule-based automation for accelerator selection during runtime (*under development*)
- Asynchronous system monitoring/decision-making for accelerator selection (*under development*)

## Building
```
git clone XXX
cd CAMNAS.jl && make
```

## Deps
Building:
- `gcc`
- `make`
- `julia>=1.10`
- ...

For Julia package depdendencies, see [Project.toml](CAMNAS/Project.toml#6)

## Usage

```
<dpsim-scenario> -U "Plugin" -P "camnasjl"
```

For more information on using MNASolverPlugins in DPsim, see: 

### Environment Variables
Certain features and behavior patterns of the solver can be managed through environment variables. For details, refer to the following:

| Variable | Values (**default**) | Description | 
| :--: | :--: | :--: |
|JL_MNA_DISABLE_AWARENESS|**true**/false| [CONTROL] Toggle accelerater detection/awareness. When disabled, calculations will always fallback to CPU.|
|JL_MNA_ALLOW_CPU|**true**/false| [CONTROL] Allow accelerator selection to use CPU. |
|JL_MNA_FORCE_CPU|true/**false**|[CONTROL] *Currently unused...* |
|JL_MNA_ALLOW_GPU|**true**/false|[CONTROL] Allow accelerator selection to use CPU. |
|JL_MNA_FORCE_GPU|true/**false**|[CONTROL] *Currently unused...*|
|PRINT_ACCELERATOR|true/**false**|[DEBUG] Print used accelerator type during solving steps, without DEBUG statements.|