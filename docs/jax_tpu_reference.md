# JAX & TPU Reference Guide

*A practical reference for understanding JAX, hardware accelerators, and when to use what.*

---

## Table of Contents
1. [Hardware: CPU vs GPU vs TPU](#hardware-cpu-vs-gpu-vs-tpu)
2. [Why NVIDIA Dominates](#why-nvidia-dominates)
3. [JAX Fundamentals](#jax-fundamentals)
4. [When to Use What](#when-to-use-what)
5. [Practical JAX Examples](#practical-jax-examples)
6. [Spellbrush Context](#spellbrush-context)

---

## Hardware: CPU vs GPU vs TPU

### CPU (Central Processing Unit)

```
Architecture:   Few powerful cores (4-16 typical)
Strengths:      Complex logic, branching, sequential tasks
Weaknesses:     Parallelism limited to core count

Analogy:        A brilliant professor solving problems one by one
                Fast per problem, but only has 2 hands

Use for:        General computing, web servers, preprocessing
Cores:          4-16 (M2 has 8)
```

### GPU (Graphics Processing Unit)

```
Architecture:   Thousands of tiny cores (10,000+)
Strengths:      Massive parallelism, same operation on many elements
Weaknesses:     Complex branching, sequential logic

Analogy:        An army of calculators working simultaneously
                Each slower than CPU, but 10,000 of them in parallel

Use for:        Matrix operations, deep learning, graphics
Cores:          ~10,000-16,000 (RTX 4090 has 16,384)

Why ML loves GPUs:
  Neural network training = matrix multiplications
  Matrix multiply = apply same operation to every element
  Perfect for parallel execution
```

### TPU (Tensor Processing Unit)

```
Architecture:   Systolic array optimized for matrix math
Strengths:      Matrix operations (and ONLY matrix operations)
Weaknesses:     Cannot run general-purpose code

Analogy:        A factory assembly line that ONLY makes one product
                Can't do anything else, but makes that one thing
                faster and cheaper than anyone

Use for:        Large-scale ML training at Google
Cores:          Not comparable - different architecture entirely

Why Google made TPUs:
  - Paying NVIDIA billions per year
  - GPUs are general-purpose (paying for features not needed)
  - Build chips that ONLY do ML math = cheaper at scale
```

### Visual Comparison

```
TASK: Multiply two 1000x1000 matrices

CPU (8 cores):
┌─────────────────────────────┐
│ ∙ → ∙ → ∙ → ∙ → ∙ → ∙ → ... │  Sequential processing
└─────────────────────────────┘
Time: ~100ms

GPU (16,384 cores):
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ ∙ │ ∙ │ ∙ │ ∙ │ ∙ │ ∙ │ ∙ │ ∙ │  Parallel processing
├───┼───┼───┼───┼───┼───┼───┼───┤
│ ∙ │ ∙ │ ∙ │ ∙ │ ∙ │ ∙ │ ∙ │ ∙ │  (10,000+ at once)
└───┴───┴───┴───┴───┴───┴───┴───┘
Time: ~1ms

TPU (systolic array):
╔═══════════════════════════════╗
║  Data flows through array     ║
║  Matrix multiply = 1 op       ║
╚═══════════════════════════════╝
Time: ~0.1ms
```

---

## Why NVIDIA Dominates

### The CUDA Moat

```
2007: NVIDIA releases CUDA
      - Programming language for GPUs
      - Free tools, tutorials, ecosystem
      - Makes GPU programming accessible

2012: AlexNet wins ImageNet using NVIDIA GPUs
      - "Graphics cards can do AI!"
      - Every researcher buys NVIDIA

2024: NVIDIA is a $2T company
      - All ML frameworks built on CUDA
      - PyTorch, TensorFlow = CUDA under the hood
      - Switching cost is enormous
```

### The Stack

```
Your Python Code
       │
       ▼
┌─────────────────┐
│    PyTorch      │  ← High-level API
└─────────────────┘
       │
       ▼
┌─────────────────┐
│     CUDA        │  ← NVIDIA's proprietary layer
└─────────────────┘
       │
       ▼
┌─────────────────┐
│   NVIDIA GPU    │  ← Hardware
└─────────────────┘

The trap: CUDA only works on NVIDIA hardware
         Everyone learned CUDA
         Now everyone is locked in
```

### Competitors

```
AMD:    Has GPUs, has ROCm (CUDA alternative)
        Reality: Limited ML support, small ecosystem

Intel:  Has GPUs, has OneAPI
        Reality: Barely functional for ML

Apple:  Has M1/M2/M3, has MPS (Metal Performance Shaders)
        Reality: Growing support, limited to Apple ecosystem

Google: Has TPUs, has JAX
        Reality: Viable alternative at scale, different paradigm
```

---

## JAX Fundamentals

### What is JAX?

```
JAX = NumPy + Autograd + XLA + Functional Programming

NumPy:      Array operations (you know this)
Autograd:   Automatic differentiation (gradients for free)
XLA:        Compiler for accelerators (GPU/TPU)
Functional: Pure functions, no side effects
```

### Core Idea

```python
# NumPy (CPU only, no gradients)
import numpy as np
y = np.dot(x, W) + b  # Only runs on CPU

# JAX (GPU/TPU, automatic gradients, same syntax!)
import jax.numpy as jnp
y = jnp.dot(x, W) + b  # Runs on GPU/TPU automatically
```

### The Three Magic Functions

#### 1. `jit` - Just-In-Time Compilation

```python
from jax import jit

def slow_function(x, W):
    return jnp.dot(x, W)

fast_function = jit(slow_function)

# First call: compiles to optimized XLA
# All future calls: 10-100x faster
```

#### 2. `grad` - Automatic Differentiation

```python
from jax import grad

def loss(params, x, y):
    pred = model(params, x)
    return ((pred - y) ** 2).mean()

# Returns a FUNCTION that computes gradients
gradient_fn = grad(loss)

# Use it
grads = gradient_fn(params, x, y)  # No .backward(), no tape!
```

#### 3. `vmap` - Automatic Vectorization

```python
from jax import vmap

def process_single(x):
    return model(x)  # Handles one sample

# Automatically handles batches
process_batch = vmap(process_single)

# No manual batch dimension handling!
results = process_batch(batch_of_samples)
```

### JAX vs PyTorch

```
                    PyTorch                 JAX
─────────────────────────────────────────────────────
Style               Object-oriented         Functional
Gradients           .backward()             grad(fn)
Compilation         torch.compile (new)     jit (native)
Batching            Manual                  vmap
Random numbers      torch.rand()            jax.random.key()
State               Mutable tensors         Immutable arrays
Target hardware     NVIDIA (CUDA)           Google (TPU/XLA)
Ecosystem           Massive                 Growing
Debugging           Easy (eager mode)       Harder (compiled)
```

---

## When to Use What

### Decision Tree

```
Are you a student/researcher doing experiments?
  └─► PyTorch (more tutorials, easier debugging)

Are you training at massive scale (billions of params)?
  └─► Consider JAX + TPUs (cheaper at scale)

Are you at a company using TPUs?
  └─► JAX (it's what TPUs want)

Are you building production systems?
  └─► PyTorch (more deployment options)

Do you want a job at Spellbrush?
  └─► Know both, show JAX familiarity
```

### Cost Comparison at Scale

```
Training a large diffusion model:

PyTorch + NVIDIA A100 cluster:
  - 100 GPUs × $2/hr × 1000 hours = $200,000

JAX + Google TPU pod:
  - Equivalent TPUs × $1/hr × 800 hours = $80,000

At Spellbrush/Google scale:
  - Training niji・journey
  - Savings of millions of dollars
  - JAX makes economic sense
```

---

## Practical JAX Examples

### Basic Neural Network

```python
import jax.numpy as jnp
from jax import grad, jit, vmap
import jax.random as random

# Initialize parameters
def init_params(key, layer_sizes):
    params = []
    for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
        key, subkey = random.split(key)
        W = random.normal(subkey, (in_size, out_size)) * 0.01
        b = jnp.zeros(out_size)
        params.append((W, b))
    return params

# Forward pass (pure function!)
def forward(params, x):
    for W, b in params[:-1]:
        x = jnp.tanh(jnp.dot(x, W) + b)
    W, b = params[-1]
    return jnp.dot(x, W) + b

# Loss function
def loss_fn(params, x, y):
    pred = forward(params, x)
    return jnp.mean((pred - y) ** 2)

# Get gradients
grad_fn = jit(grad(loss_fn))

# Training step
@jit
def train_step(params, x, y, lr=0.01):
    grads = grad_fn(params, x, y)
    return [(W - lr * dW, b - lr * db)
            for (W, b), (dW, db) in zip(params, grads)]
```

### Key Differences from PyTorch

```python
# PyTorch style (mutable, object-oriented)
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

model = Model()
optimizer = Adam(model.parameters())

for x, y in dataloader:
    loss = criterion(model(x), y)
    loss.backward()        # Mutates internal state
    optimizer.step()       # Mutates parameters
    optimizer.zero_grad()  # Mutates gradients


# JAX style (immutable, functional)
def model(params, x):
    return jnp.dot(x, params['W']) + params['b']

@jit
def train_step(params, x, y):
    loss, grads = value_and_grad(loss_fn)(params, x, y)
    params = tree_map(lambda p, g: p - 0.01 * g, params, grads)
    return params, loss  # Return NEW params, don't mutate

for x, y in dataloader:
    params, loss = train_step(params, x, y)  # Explicit state
```

---

## Spellbrush Context

### Why Spellbrush Uses JAX

```
Product:        niji・journey (anime image generation)
Scale:          Billions of images, massive models
Infrastructure: Google Cloud TPUs

Economics:
  - TPUs are cheaper than NVIDIA at their scale
  - JAX is the native language for TPUs
  - Tight research-to-production loop

Their stack:
  - JAX for large-scale TPU training
  - PyTorch for some components
  - Custom infrastructure
```

### What They're Looking For

```
From the job posting:

"PROFICIENT IN PYTORCH (AND MAYBE JAX TOO)"
  └─► Know PyTorch well, have JAX exposure

"TIGHT LOOP BETWEEN RESEARCH AND PRODUCTION"
  └─► Can ship models, not just train them

"SMALL, FAST-PACED TEAMS"
  └─► 4 people on AI, high ownership

"ON-SITE TOKYO OR SF"
  └─► Akihabara office (!)
```

### How Your Touhou Project Helps

```
Demonstrates:
  ✓ PyTorch familiarity (your classifier)
  ✓ Audio ML experience (complementary to vision)
  ✓ Anime/doujin domain knowledge
  ✓ Research → working model pipeline

To strengthen:
  ○ Port one component to JAX
  ○ Add diffusion (Phase 2 of your project)
  ○ Show you can work at scale
```

---

## Learning Resources

### JAX

- [JAX Quickstart](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)
- [Thinking in JAX](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html)
- [JAX 101 Tutorial](https://jax.readthedocs.io/en/latest/jax-101/index.html)

### TPUs

- [Google Cloud TPU Docs](https://cloud.google.com/tpu/docs)
- [TPU Research Cloud](https://sites.research.google/trc/) (free TPU access for researchers)

### Diffusion + JAX

- [Flax (JAX neural network library)](https://github.com/google/flax)
- [JAX implementation of diffusion models](https://github.com/google-research/vdm)

---

*Document created for interview prep. Last updated: 2024-12*
