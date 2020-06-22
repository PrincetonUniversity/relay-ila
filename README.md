# Relay ILA

The ILA model of TVM/Relay.

- vector operations (add, multiply, tanh, sigmoid)
- nn dense operation (matrix-vector multiplication)
- maxpooling-2d
- lstm


# Requirements

- TVM/Relay Python interface https://tvm.apache.org/docs/install/from_source.html
- ILAng https://github.com/Bo-Yuan-Huang/ILAng
    - `-DILANG_INSTALL_DEV=ON`
- SystemC https://www.accellera.org/downloads/standards/systemc
- CMake
- C++11 compiler

# Build instructions

To build the Relay ILA:

``` bash
mkdir build
cd build
cmake ..
make
```

To generate the SystemC simulation model, in `<project-root>/build`:

``` bash
./relay
cp ../app/sim_main.cc sim_model/app/main.cc
cp ../uninterpreted_func/uninterpreted_func.cc sim_model/extern/
cd sim_model
mkdir build
cd build
cmake ..
make
```

To run sanity checking simulation, in `<project-root>/build/sim_model/build`:

``` bash
./<project-root>/script/lstm_test.py 64 lstm.bin
cp <project-root>/flex-sim/* .
mv flex-64x64-step1.bin lstm.bin
./relay_sim
mv flex-64x64-step2.bin lstm.bin
./relay_sim
```
    
# Input/Output sizes

for input size of I and output size of O

`input` shoud be vectors of size `I`

`cell`, `next_cell`, `hidden`, `next_hidden` shoud be vectors of size `O`

`12h_weight` should be matrix of size `O*4 x I`

`i2h_bias` should be vector of size `O*4`

`h2h_weight` should be matrix of size `O*4 x O`

`h2h_bias` should be vector of size `O*4`

`temp_vector0` `temp_vector1` `temp_vector2` should be vectors of size `O*4`

# instruction sequence

- i2h
    - `O*4` groups of `relay_nn_dense_loop_fma_instr`, `I` per group (`O*I*4` total)
- h2h
    - `O*4` groups of `relay_nn_dense_loop_fma_instr`, `O` per group (`O*O*4` total)
- add dense results
    - `O*4` number of `relay_vector_add_child_instr`
- sigmoid 
    - `O*3` number of `relay_vector_sigmoid_child_instr`
- cell tanh
    - `O` number of `relay_vector_tanh_child_instr`
- forget gate
    - `O` number of `relay_vector_multiply_child_instr`
- input gate
    - `O` number of `relay_vector_multiply_child_instr`
- next cell
    - `O` number of `relay_vector_add_child_instr`
- next cell tanh
    - `O` number of `relay_vector_tanh_child_instr`
- output gate
    - `O` number of `relay_vector_multiply_child_instr`


