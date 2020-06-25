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

// File: relay_arch_states.cc

#include <ilang/util/log.h>

#include <relay/relay_top.h>

namespace ilang {

namespace relay {

void DefineArchState(Ila& m) {
  // tensor memory
  m.NewMemState(RELAY_TENSOR_MEM, RELAY_FUNC_ADDR_IN_BITWIDTH,
                RELAY_FUNC_DATA_IN_BITWIDTH);

  // memory space used by lstm/vector_op/nn_dense
  m.NewMemState(RELAY_MEMORY, RELAY_LSTM_ADDR_BW, RELAY_VECTOR_DATA_BW);
}

} // namespace relay

} // namespace ilang
