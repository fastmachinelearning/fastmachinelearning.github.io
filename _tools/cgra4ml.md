---
layout: gallery-item
title: CGRA4ML
summary: >-
  CGRA4ML is an open source Python library for building, training, and
  implementing deep neural networks on FPGAs and custom ASICs.

  It combines a configurable accelerator, SystemVerilog RTL, C firmware, and system integration scripts in one workflow.

  Models reuse the same coarse-grained reconfigurable array (CGRA) across layers, allowing the hardware to support networks that are larger than its on-chip storage.
submitter: Abarajithan Gnaneswaran (https://abapages.com/#contact-me)
domain: Transpiler
image: /images/crga4ml.png
review_status: draft
---
This tutorial follows the [repository’s quick start](https://github.com/KastnerRG/cgra4ml#quick-start):
  - Train a quantized MNIST model
  - Choose an accelerator configuration
  - Check the generated hardware and firmware against the Python model.

## Setup

The package requires Python 3.11.5.
This walkthrough uses XSim from Vivado 2024.1.
Make sure its `xsc`, `xvlog`, `xelab`, and `xsim` commands are on your PATH, by checking `which xsim`.
The README also offers a Dockerfile setup with Verilator installed.

```bash
conda create -n cgra4ml_env python=3.11.5 pip -y
conda activate cgra4ml_env

git clone https://github.com/KastnerRG/cgra4ml
cd cgra4ml
python -m pip install .
python -m pip check
mkdir -p run/work
```

Run the commands below from this checkout, with the environment active.

## 1. Train and run the example

```bash
cd run/work
python ../example.py --sim xsim
```

The script downloads MNIST, trains the model, saves and reloads `mnist.h5`, exports the accelerator configuration and inference data, then runs simulation and prints a performance estimate.
No FPGA board is needed for this step.

The model has six convolution bundles and a final dense classifier.
It exercises batch normalization, pooling, residual additions, flattening, and softmax.
Its quantization is specified with:

```python
sys_bits = SYS_BITS(x=4, k=4, b=16)
```

This selects 4-bit activations and weights, with 16-bit biases.
The number of integer bits are specified in the `XBundle`.
Models are built using `XModel` and `XBundle`. 
A bundle groups a convolution or dense operation with its associated processing.
For example, the first bundle in `run/example.py` is:

```python
self.b1 = XBundle(
    core=XConvBN(
        k_int_bits=0, b_int_bits=0, filters=8,
        kernel_size=7, strides=(2, 1),
        act=XActivation(sys_bits=sys_bits, o_int_bits=0,
                        type='relu', slope=0)),
    pool=XPool(
        type='avg', pool_size=(3, 4), strides=(2, 3), padding='same',
        act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None))
)
```

The Python excerpts here explain code already in the example.
The short training run demonstrates the workflow.

## 2. Choose the hardware

The example’s `Hardware(...)` configuration uses an **8 × 24 array of multiply-add units**, a 250 MHz target clock, 4-bit inputs and weights, a 20-bit accumulator, and a 128-bit AXI interface.
Array dimensions, buffer depths, and supported tensor sizes can all be changed through the Python API.

A larger array offers more parallel computation, but its benefit depends on the model shapes and memory bandwidth.
The target clock is a design setting. 
Timing closure must be checked after implementation.

The example saves the configuration and generates the Vivado entry point:

```python
hw.export_json()
hw = Hardware.from_json('hardware.json')
hw.export()
hw.export_vivado_tcl(board='zcu104')
```

These calls produce `hardware.json`, `config_hw.svh`, `config_hw.tcl`, and `vivado_flow.tcl`.
The current example targets **ZCU104**.
Select the appropriate board before exporting if using different hardware.

## 3. Verify inference and inspect performance

The final part of the example exports model data and checks the RTL together with the C runtime:

```python
export_inference(loaded_model, hw, batch_size=1)
verify_inference(loaded_model, hw, SIM=SIM)

pprint.pprint(predict_model_performance(hw))
```

Verification checks intermediate integer results and packed data for exact agreement.
The final softmax output uses a numerical tolerance, so this example should not be described as bit-exact end to end.
Successful checks print `Bundle ..., Error: .... Passed`.
Generated inference data, including `vectors/wbx.bin`, remain in `run/work` for deployment.

## 4. Build for an FPGA

Use Vivado **2022.2** for the FPGA flow.
From `run/work`, select the installed version explicitly (adjust the installation path on other machines):

```bash
/tools/Xilinx/Vivado/2022.2/bin/vivado -mode batch -source vivado_flow.tcl
```

The flow creates the Zynq system, connects the accelerator’s AXI interfaces, runs synthesis and implementation, and exports a bitstream and hardware platform.
For ZCU104, the intended platform output is `dsf_zcu104/design_1_wrapper.xsa`. 
Timing and utilization reports go under `dsf_zcu104/reports`.

To run on a board, use Vitis 2022.2 and follow the README’s procedure with the additional include path below:

1. Create an application using the generated `.xsa` and the repository’s `deepsocflow/c/xilinx_example.c`.
2. Add the absolute paths of `run/work`, `deepsocflow/c`, and `deepsocflow/firebridge` to the compiler include paths, select `-O3`, and link the math library (`m`).
3. Build the application, connect the matching Zynq board, and launch debugging.
4. Break at `model_setup()`, load `vectors/wbx.bin` at the address printed by the application, and continue.

The current C example calls `hardware_setup()`, `model_run_timed(p_mem, 20)`, and `print_output(p_mem)`, then cleans up.
It reports average inference time across 20 timed runs after an initial run.

## Further reading

The [CGRA4ML repository](https://github.com/KastnerRG/cgra4ml) includes additional models, Ibex SoC integration, and ASIC implementation scripts.

G. Abarajithan, Zhenghua Ma, Ravidu Munasinghe, Francesco Restuccia, and Ryan Kastner, [“CGRA4ML: A Hardware/Software Framework to Implement Neural Networks for Scientific Edge Computing”](https://doi.org/10.1145/3801097), *ACM Transactions on Reconfigurable Technology and Systems*, 19(2), Article 26, 2026.
