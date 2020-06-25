// =============================================================================
// MIT License
//
// Copyright (c) 2020 Princeton University
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// =============================================================================

// File: relay_vector_op.h

#ifndef RELAY_VECTOR_OP_H__
#define RELAY_VECTOR_OP_H__

namespace ilang {

namespace relay {

#define RELAY_MEMORY "relay_memory"

#define RELAY_VECTOR_DATA_BW 32
#define RELAY_VECTOR_DATA_BYTES 4
#define RELAY_WORD_ADDR_SHIFT 2
#define RELAY_VECTOR_DATA_ZERO 0

#define RELAY_VECTOR_ADDR_BW 32
#define RELAY_VECTOR_SIZE_BW 32

#define RELAY_LOAD_WORD(__memory, __byte_addr)                                 \
  Load(__memory, (__byte_addr) >> 2)
#define RELAY_STORE_WORD(__memory, __byte_addr, __word_val)                    \
  Store(__memory, (__byte_addr) >> 2, (__word_val))

#define RELAY_FLAG_BW 1
#define RELAY_FLAG_ON 1
#define RELAY_FLAG_OFF 0

#define RELAY_ITE_FLAG(expr)                                                   \
  Ite((expr), BvConst(RELAY_FLAG_ON, RELAY_FLAG_BW),                           \
      BvConst(RELAY_FLAG_OFF, RELAY_FLAG_BW))

#define RELAY_VECTOR_OP_CHILD "relay_vector_op_child_module"
#define RELAY_VECTOR_OP_SIZE "relay_vector_op_size"
#define RELAY_VECTOR_OP_SIZE_BW RELAY_VECTOR_SIZE_BW
#define RELAY_VECTOR_OP_CNTR "relay_vector_op_cntr"
#define RELAY_VECTOR_OP_CNTR_BW RELAY_VECTOR_SIZE_BW

// common input/output vector address
#define RELAY_VECTOR_OP0_ADDR "relay_vector_op0_addr"
#define RELAY_VECTOR_OP1_ADDR "relay_vector_op1_addr"
#define RELAY_VECTOR_OUTPUT_ADDR "relay_vector_output_addr"

// vector add
#define RELAY_VECTOR_ADD "relay_vector_add_instr"
#define RELAY_VECTOR_ADD_ENABLE "relay_vector_add_enable"

#define RELAY_VECTOR_ADD_START "relay_vector_add_start"

#define RELAY_VECTOR_ADD_CHILD "relay_vector_add_child_module"
#define RELAY_VECTOR_ADD_CHILD_INSTR "relay_vector_add_child_instr"

#define RELAY_VECTOR_ADD_OP0_ADDR RELAY_VECTOR_OP0_ADDR
#define RELAY_VECTOR_ADD_OP1_ADDR RELAY_VECTOR_OP1_ADDR
#define RELAY_VECTOR_ADD_OUTPUT_ADDR RELAY_VECTOR_OUTPUT_ADDR

// vector multiply
#define RELAY_VECTOR_MULTIPLY "relay_vector_multiply_instr"
#define RELAY_VECTOR_MULTIPLY_ENABLE "relay_vector_multiply_enable"

#define RELAY_VECTOR_MULTIPLY_START "relay_vector_multiply_start"

#define RELAY_VECTOR_MULTIPLY_CHILD "relay_vector_multiply_child_module"
#define RELAY_VECTOR_MULTIPLY_CHILD_INSTR "relay_vector_multiply_child_instr"

#define RELAY_VECTOR_MULTIPLY_OP0_ADDR RELAY_VECTOR_OP0_ADDR
#define RELAY_VECTOR_MULTIPLY_OP1_ADDR RELAY_VECTOR_OP1_ADDR
#define RELAY_VECTOR_MULTIPLY_OUTPUT_ADDR RELAY_VECTOR_OUTPUT_ADDR

// vector sigmoid
#define RELAY_VECTOR_SIGMOID "relay_vector_sigmoid_instr"
#define RELAY_VECTOR_SIGMOID_ENABLE "relay_vector_sigmoid_enable"
#define RELAY_VECTOR_SIGMOID_START "relay_vector_sigmoid_start"

#define RELAY_VECTOR_SIGMOID_CHILD "relay_vector_sigmoid_child_module"
#define RELAY_VECTOR_SIGMOID_CHILD_INSTR "relay_vector_sigmoid_child_instr"

#define RELAY_VECTOR_SIGMOID_OP0_ADDR RELAY_VECTOR_OP0_ADDR
#define RELAY_VECTOR_SIGMOID_OUTPUT_ADDR RELAY_VECTOR_OUTPUT_ADDR

// vector tanh

#define RELAY_VECTOR_TANH "relay_vector_tanh_instr"
#define RELAY_VECTOR_TANH_ENABLE "relay_vector_tanh_enable"
#define RELAY_VECTOR_TANH_START "relay_vector_tanh_start"

#define RELAY_VECTOR_TANH_CHILD "relay_vector_tanh_child_module"
#define RELAY_VECTOR_TANH_CHILD_INSTR "relay_vector_tanh_child_instr"

#define RELAY_VECTOR_TANH_OP0_ADDR RELAY_VECTOR_OP0_ADDR
#define RELAY_VECTOR_TANH_OUTPUT_ADDR RELAY_VECTOR_OUTPUT_ADDR

static auto bv_sort_out = SortRef::BV(RELAY_VECTOR_DATA_BW);
static auto bv_sort_in0 = SortRef::BV(RELAY_VECTOR_DATA_BW);
static auto bv_sort_in1 = SortRef::BV(RELAY_VECTOR_DATA_BW);
static FuncRef bv_sigmoid("bv_sigmoid", bv_sort_out, bv_sort_in0);
static FuncRef bv_tanh("bv_tanh", bv_sort_out, bv_sort_in0);
static FuncRef bv_multiply("bv_multiply", bv_sort_out, bv_sort_in0,
                           bv_sort_in1);
static FuncRef bv_add("bv_add", bv_sort_out, bv_sort_in0, bv_sort_in1);

} // namespace relay

} // namespace ilang

#endif
