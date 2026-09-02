---
layout: gallery-item
title: AIE4ML
summary: Run your model on an AI Engine as dataflow in a few easy steps! We
  convert your model into a spatial array of vector-processor tiles; each layer
  becomes a kernel placed on one tile, or a block of them when parallelised.
  Tensors pass between kernels either directly (between local memory banks) or a
  through memory tile's DMA when the layout must be reshaped, so placement and
  connections matter as much as compute.
submitter: Dimitrios Danopolous
domain: AI engine, transpiler
image: /images/aie4ml_logo_big.png
review_status: draft
---
**\# 🚀 AIE4ML Tutorial**
A single quantized model that stresses several system-level features of the compiler. Along the way we apply *\*tuning\** directives and confirm the result is still bit-exact with QKeras. 
- **\*\*The AI Engine runs your model as dataflow.\*\*** It's a spatial array of vector-processor tiles; each layer becomes a kernel placed on one tile, or a block of them when parallelised. Tensors pass between kernels either directly (between local memory banks) or a through memory tile's DMA when the layout must be reshaped, so placement and connections matter as much as compute.
- **\*\*Arithmetic.\*\*** The vector unit computes in integers, but every tensor carries a fixed-point interpretation — a Q-format scale of 2^-frac — that the compiler tracks end to end. So the arithmetic is exact and deterministic, while you keep reasoning in the quantized real-valued domain your model was trained in. Float-like datatypes are also supported from the recent AIEML generations.
- **\*\*Automatic like hls4ml, but steerable.\*\*** As hls4ml is to FPGAs, aie4ml is to the AI Engine: a quantized graph becomes a Vitis working project. But it isn't a black box — you steer it with per-layer directives (parallelism, layout, placement), and you can register a new kernel variant for a layer.

> We use the **\*\*QKeras / hls4ml\*\*** path here because it is seamless to write and readas cleanly for> a demo. The **\*\*ONNX\*\*** frontend is more expressive and more fully supported in aie4ml — and the> same directives work through it unchanged.

**\## ⚙️ Setup & Imports**
\`\`\`bashconda create -n aie4ml_env python=3.10 -y && conda activate aie4ml_envpip install "tensorflow==2.15.*" "tensorflow-model-optimization==0.7.5" qkeras hls4mlpip install aie4ml pyparsing ipykernel pydot graphviz\`\`\`


\`\`\` pythonimport numpy as npfrom tensorflow.keras.models import Modelfrom tensorflow.keras.layers import Input, Permutefrom qkeras import QDense, QActivation, quantized_bits, quantized_reluimport hls4ml
from aie4ml.model import from_hls4mlfrom aie4ml.report import report\`\`\`

**\## 💻 1. The model**
A shared trunk fans out to two branches. **\*\*Branch 1\*\*** applies a \`Permute\`; **\*\*Branch 2\*\*** is astraight dense chain. Each branch has its own output. The input is quantized to **\*\*16 bits\*\***;weights and activations are int8. 

\`\`\`pythonTOKENS, FEATURES = 16, 64
def dense(name, units):    return QDense(units, name=name,                  kernel_quantizer=quantized_bits(8, 0, alpha=1),                  bias_quantizer=quantized_bits(8, 2, alpha=1))
def build_model():    inp = Input(shape=(TOKENS, FEATURES), name="in")    x = QActivation(quantized_bits(16, 6), name="in_q16")(inp)      # 16-bit input
    x = dense("trunk_fc", FEATURES)(x)                              # shared trunk -> fanout    x = QActivation(quantized_relu(8), name="trunk_relu")(x)
    # Branch 1: has a Permute (a transpose the DMA must physically reorder)    b1 = dense("b1_fc1", FEATURES)(x)    b1 = QActivation(quantized_relu(8), name="b1_relu")(b1)    b1 = Permute((2, 1), name="b1_permute")(b1)    b1 = dense("b1_fc2", FEATURES)(b1)    b1 = QActivation(quantized_bits(8, 2), name="b1_out")(b1)
    # Branch 2: a straight dense chain    b2 = dense("b2_fc1", FEATURES)(x)    b2 = QActivation(quantized_relu(8), name="b2_relu1")(b2)    b2 = dense("b2_fc2", FEATURES)(b2)    b2 = QActivation(quantized_relu(8), name="b2_relu2")(b2)    b2 = dense("b2_fc3", FEATURES)(b2)    b2 = QActivation(quantized_bits(8, 2), name="b2_out")(b2)
    return Model(inp, \[b1, b2], name="branching_model")
model = build_model()model.compile(loss="mse")model.summary()\`\`\`

**\#### 💡 Architecture decisions highlighted by this model**
These are not just layer conversions — each is a system-level choice in lowering, memoryplanning and graph I/O:
- **\*\*16-bit input\*\*** — wider input precision carried through the staging descriptors.- **\*\*Fanout after \`trunk_fc\`\*\*** — one tensor consumed by two branches (broadcast).- **\*\*\`Permute\` in Branch 1\*\*** — usually the compiler reorders it through a memory tile, with explicit layout/view handling in the buffer descriptors.- **\*\*Two graph outputs\*\*** — output-port mapping and per-tensor simulator decoding.- **\*\*ND Dense\*\*** — \`Dense\` on rank-3 tensors stresses the descriptor shape/traversal contracts.


**\## 📍 2. Tuning directives: parallelism and placement**
Directives are attached per layer in the hls4ml config.
**\*\*Parallelism\*\*** — \`cas_num\` independent cascade chains, each \`cas_length\` deep:- \`cas_length\` splits the **\*\*reduction\*\*** (K, the input features).- \`cas_num\` splits the **\*\*output features\*\*** (N) by default (\`'contract': 'inner'\`), or the **\*\*rows\*\*** (M) with  \`'contract': 'outer'\`.- More parallelism = more AIE tiles = higher throughput. Keep both **\*\*≤ 8\*\***. Here, the token dimension is what parallelism will split (\`'outer'\`).
**\*\*Placement\*\*** — \`{'row': r, 'col': c}\` pins a kernel to a tile. Branch 2 is a straight chain with compatible layouts/partinioning, so its kernels can connect **\*\*directly\*\*** (avoiding the memory tiles).In this case specifically, each can read the memory bank its left neighbour wrote so we can pin them to **\*\*adjacent columns\*\***, otherwise we have to keep an empty tile in between. The compiler's automatic placer can usually handle the placement for us.

\`\`\`pythoncfg = hls4ml.utils.config_from_keras_model(model, granularity="name")
for layer in ("trunk_fc", "b1_fc1", "b1_fc2", "b2_fc1", "b2_fc2", "b2_fc3"):    cfg\["LayerName"]\[layer]\["parallelism"] = {"cas_num": 2, "cas_length": 1, "contract": "outer"}
for column, layer in zip((7, 8, 9), ("b2_fc1", "b2_fc2", "b2_fc3")):    cfg\["LayerName"]\[layer]\["placement"] = {"row": 1, "col": column}\`\`\`

**\## 3. Convert and compile**
Use the hls4ml frontend to convert to the AIE backend, then compile for the x86 simulator.

\`\`\`pythonaie_model = hls4ml.converters.convert_from_keras_model(    model, hls_config=cfg,    output_dir="proj_tutorial3", project_name="proj_tutorial3",    backend="aie", batch_size=1, iterations=10,)aie_model.compile()\`\`\`

**\## 4. Bit-exact comparison with QKeras**


\`\`\`pythonx = np.random.random((1, TOKENS, FEATURES)).astype(np.float32)
y_qkeras = model.predict(x, verbose=0)y_aie = aie_model.predict(x, simulator="x86")
for name, out, ref in zip(("branch1", "branch2"), y_aie.values(), y_qkeras):    print(f"{name}: max abs diff = {float(np.max(np.abs(np.asarray(out) - ref)))}")\`\`\`

**\## 🤖 5. Build for the AI Engine and profile**
The x86 simulator checks the output numbers but not the timing. To get **\*\*cycles and latency\*\***, wecompile the graph for the AI Engine and run the cycle-accurate simulator with profiling. It isslower than x86, but it is what fills in the performance numbers in the report below -- and weconfirm it stays **\*\*bit-exact\*\*** here too.

\`\`\`pythonaie_model.build()                                   # aiecompiler: compile for the AI Enginey_aie_sim = aie_model.predict(x, simulator="aie")   # cycle-accurate simulation + profiling
for name, out, ref in zip(("branch1", "branch2"), y_aie_sim.values(), y_qkeras):    print(f"{name}: max abs diff = {float(np.max(np.abs(np.asarray(out) - ref)))}")\`\`\`

**\## 📝 6. The project report**
\`report\` summarises the compiled project — how many kernels, how many AIE tiles and memory-tilebuffers, and (after running inference in '\`aie\`' mode) cycles and latency. 
In this example, the design achieves 2169 GOP/s with an end-to-end latency of approximately 3 μs. While increasing the number of AIE tiles can further improve performance, doing so eventually yields diminishing returns. As more tiles are added, data movement and synchronization overhead begin to dominate execution time, reducing overall hardware utilization. For the highest efficiency, each AIE tile should process a workload that fits entirely within its local memory. In practice, this typically means keeping the working set—especially the model weights—within the tile's local memory banks (commonly around 16 KB for weights). 

\`\`\`pythonreport(aie_model)\`\`\`

**\## 💡 Takeaways**
- One model exercised **\*\*fanout, a transpose, two outputs, ND dense, and a 16-bit input\*\*** - **\*\*Parallelism\*\*** (\`cas_num\` / \`cas_length\` / \`contract\`) trades AIE tiles for throughput.- A **\*\*transpose\*\*** (\`Permute\`) is realised through a memory tile automatically;- Tuning don't change the numerics — the result stays **\*\*bit-exact\*\*** with QKeras.
The same directives apply through the ONNX frontend, which is the more expressive and more fully supported path in aie4ml.
**\## 📖 Further Reading**For a deeper understanding of the AIE4ML compilation flow and the optimization techniques presented in this tutorial, we recommend the following publications:
\* Danopoulos Dimitrios et al., "AIE4ML: An End-to-End Framework for Compiling Neural Networks for the Next Generation of AMD AI Engines," FCCM 2026. Also available on arXiv: https://arxiv.org/abs/2512.15946
\* Danopoulos Dimitrios et al., "Taming the Exponential: A Fast Softmax Surrogate for Integer-Native Edge Inference," arXiv:2604.02292 (2026). Available on arXiv: https://arxiv.org/abs/2604.02292
⭐ If you build something interesting with AIE4ML, we'd love to hear about it! Feel free to open an issue or submit a pull request to the project repository.
