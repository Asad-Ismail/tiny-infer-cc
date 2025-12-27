# AI Coding Agent Instructions for tiny-infer-cc

## Project Overview
This is a static inference compiler that transforms restricted neural network graphs into optimized C++ code. The compiler performs operator fusion and memory reuse optimization to generate efficient inference code.

## Architecture Principles (To Be Established)

### Compiler Pipeline
When implementing the compiler, follow a multi-stage architecture:
1. **Graph IR**: Neural network graph representation with ops, tensors, and dependencies
2. **Analysis Phase**: Detect fusion opportunities, memory lifetime analysis, compute scheduling
3. **Optimization Pass**: Operator fusion, memory pool allocation, kernel selection
4. **Code Generation**: Emit optimized C++ with SIMD/vectorization where applicable

### Core Components (Planned)
- **Graph Parser**: Load models from ONNX/PyTorch/TensorFlow formats
- **IR Builder**: Construct internal graph representation with type checking
- **Fusion Engine**: Pattern matching for conv+relu, matmul+bias+activation chains
- **Memory Allocator**: Graph coloring or liveness analysis for tensor memory reuse
- **Code Emitter**: Template-based C++ codegen with operator implementations

## Development Workflow

### Build System
- Use CMake with separate targets for compiler binary and runtime library
- Keep generated code runtime dependencies minimal (standard library + optional BLAS)
- Provide both static and shared library options for generated inference code

### Testing Strategy
- Unit tests for each compiler pass (fusion, memory allocation, codegen)
- Integration tests comparing numerical accuracy against reference implementations
- Benchmark suite measuring inference latency and memory footprint of generated code
- Test with small networks first (2-5 layers) before scaling to larger models

### Code Organization
```
src/
  ir/           # Graph IR and tensor definitions
  parser/       # Model format importers
  passes/       # Optimization passes (fusion, memory, etc)
  codegen/      # C++ code generation
  runtime/      # Minimal runtime for generated code
tests/
  unit/         # Per-component tests
  integration/  # End-to-end model tests
  models/       # Test networks (small CNNs, MLPs)
examples/       # Example usage and generated code samples
```

## Coding Conventions (Recommendations)

### IR Design
- Immutable graph nodes with explicit data dependencies
- Use visitor pattern for graph traversals and transformations
- Represent tensors with shape, dtype, and memory location metadata
- Tag nodes with attributes for fusion eligibility and kernel types

### Optimization Passes
- Each pass should be idempotent and composable
- Validate graph correctness after each transformation
- Log fusion decisions with operator names and performance estimates
- Provide pass ordering configuration (some fusions may block others)

### Code Generation
- Generate readable C++ with clear variable names and comments
- Use `std::array` for static shapes, `std::vector` for dynamic batching
- Emit assertions for shape/dtype mismatches in generated code
- Include timing instrumentation hooks (optional at generation time)

### Memory Management
- Implement arena allocator for intermediate tensors in generated code
- Document memory layout (row-major, NCHW vs NHWC) clearly
- Reuse buffers when tensor lifetimes don't overlap
- Pre-allocate all memory at initialization in generated code

## Integration Points

### Input Formats
- Start with ONNX for maximum compatibility
- Add PyTorch JIT or TorchScript support if needed
- Consider a simple JSON-based IR format for testing

### Runtime Dependencies
- Generated code should work standalone with minimal dependencies
- Optional: link against Eigen, OpenBLAS, or BLIS for GEMM operations
- Consider providing pure C++ fallback implementations

### Output Formats
- Primary: Header + implementation file with inference function
- Include serialization/deserialization for input/output tensors
- Optionally generate CMakeLists.txt for building generated code

## Performance Considerations

- Profile generated code with real workloads, not synthetic benchmarks
- Focus on reducing memory allocations before micro-optimizations
- Document expected speedups from fusion (e.g., conv+relu: 1.3-1.8x)
- Consider parallel execution for independent subgraphs

## Key References
- README.md: Project description and goals
- .gitignore: C++ build artifacts excluded

## Notes for AI Agents
- This is a compiler project - prioritize correctness over premature optimization
- Start with a minimal working pipeline before adding advanced optimizations
- Test with tiny networks (3-5 ops) to validate the full pipeline quickly
- Generated C++ code is the product; it should be human-readable and debuggable
- When adding new operator support, implement reference version first, optimize later
