# tensor_math_demo_jl

This repository contains an interactive **Julia script (`tensor_ops.jl`)** that demonstrates the differences between:
- Element-wise operations (`.*`)
- Matrix multiplication (`*`)
- Tensor dot products (`sum(B .* C)` and `dot(vec(B), vec(C))`)
- Per-slice (batched) matrix multiplication for 3D tensors

It’s designed as a learning sandbox for understanding how Julia handles **arrays, tensors, broadcasting, and batched linear algebra** 

---

## Features
- **Interactive menu** to explore operations step-by-step.
- **Side-by-side comparisons** of element-wise vs matrix vs tensor operations.
- **Per-slice (batched) multiplication** example similar to PyTorch’s `torch.bmm`.
- Demonstrations of **broadcasting, comprehensions, and constructors**.

---
