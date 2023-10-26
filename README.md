Saffira
---

Systolic Array Framework for Fault Injection and Robust Accelerators


The library resides into the systolic_injector directory.
A simple test with a convolution and a matrix multiplication can be found in test.py.
A sample injection is provided.

Furthremore, a prototype for the visualization of the equations is given in systolic_visualization.py

**TODO**:

- [ ] implement im2col and zeroPadding as generally as possible
- [ ] formalize the injection process


# How to use

In order to use Saffira, you just need to include the directory ./src/ to your project (or to your
PYTHONPATH environment variable).

You can find some examples in the files ./test.py.

In general, we have the feollowing classes:

- `SystolicArray` - it is the main class, it configures the systolic array simulation: array size, 
type of projection, etc... The main method is `matmul` and it just performs the matrix multiplication
between the two operands. On this object you can use the methods `add_fault`, `clear_all_faults` and
`clear_single_fault` to manage the faults acting on the array.
- `Fault` - actually this is supposed to be 