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

// File: relay_vector_op.cc

#include <relay/relay_top.h>

namespace ilang {

namespace relay {

void DefineVectorAdd(Ila& m) {
  auto vector_child = m.child(RELAY_VECTOR_OP_CHILD);
  auto instr = vector_child.NewInstr(RELAY_VECTOR_ADD);

  auto vector_add_enable = m.state(RELAY_VECTOR_ADD_ENABLE);
  auto child_start = m.state(RELAY_VECTOR_ADD_START);

  instr.SetDecode(
      (vector_add_enable == RELAY_FLAG_ON) &
      (m.state(RELAY_VECTOR_OP_SIZE) != BvConst(0, RELAY_VECTOR_OP_SIZE_BW)) &
      (child_start == RELAY_FLAG_OFF));
  auto cntr = m.state(RELAY_VECTOR_OP_CNTR);

  auto memory = m.state(RELAY_MEMORY);

  instr.SetUpdate(child_start, BvConst(RELAY_FLAG_ON, RELAY_FLAG_BW));
  instr.SetUpdate(cntr, BvConst(0, RELAY_VECTOR_OP_CNTR_BW));

  {
    auto child = vector_child.NewChild(RELAY_VECTOR_ADD_CHILD);
    auto child_started = (child_start == RELAY_FLAG_ON);
    child.SetValid(child_started);
    {
      auto child_instr = child.NewInstr(RELAY_VECTOR_ADD_CHILD_INSTR);
      child_instr.SetDecode(child_started);

      auto addr_offset = cntr * RELAY_VECTOR_DATA_BYTES;
      auto op0_addr = m.state(RELAY_VECTOR_ADD_OP0_ADDR) + addr_offset;
      auto op1_addr = m.state(RELAY_VECTOR_ADD_OP1_ADDR) + addr_offset;
      auto output_addr = m.state(RELAY_VECTOR_ADD_OUTPUT_ADDR) + addr_offset;
      // uninterpreted add function
      auto result = bv_add(RELAY_LOAD_WORD(memory, op0_addr),
                           RELAY_LOAD_WORD(memory, op1_addr));

      auto next_cntr = cntr + BvConst(1, RELAY_VECTOR_OP_CNTR_BW);
      auto continue_cond = (next_cntr != m.state(RELAY_VECTOR_OP_SIZE));
      auto next_child_start = RELAY_ITE_FLAG(continue_cond);
      auto next_vector_add_enable = RELAY_ITE_FLAG(continue_cond);
      auto lstm_state = m.state(RELAY_LSTM_STATE);
      auto next_lstm_state =
          Ite(continue_cond, lstm_state, m.state(RELAY_LSTM_RETURN_STATE));

      child_instr.SetUpdate(memory,
                            RELAY_STORE_WORD(memory, output_addr, result));
      child_instr.SetUpdate(cntr, next_cntr);
      child_instr.SetUpdate(child_start, next_child_start);
      child_instr.SetUpdate(vector_add_enable, next_vector_add_enable);
      child_instr.SetUpdate(lstm_state, next_lstm_state);
    }
  }
}

void DefineVectorMultiply(Ila& m) {
  auto vector_child = m.child(RELAY_VECTOR_OP_CHILD);
  auto instr = vector_child.NewInstr(RELAY_VECTOR_MULTIPLY);

  auto vector_multiply_enable = m.state(RELAY_VECTOR_MULTIPLY_ENABLE);
  auto child_start = m.state(RELAY_VECTOR_MULTIPLY_START);

  instr.SetDecode(
      (vector_multiply_enable == RELAY_FLAG_ON) &
      (m.state(RELAY_VECTOR_OP_SIZE) != BvConst(0, RELAY_VECTOR_OP_SIZE_BW)) &
      (child_start == RELAY_FLAG_OFF));

  auto cntr = m.state(RELAY_VECTOR_OP_CNTR);

  auto memory = m.state(RELAY_MEMORY);

  instr.SetUpdate(child_start, BvConst(RELAY_FLAG_ON, RELAY_FLAG_BW));
  instr.SetUpdate(cntr, BvConst(0, RELAY_VECTOR_OP_CNTR_BW));

  {
    auto child = vector_child.NewChild(RELAY_VECTOR_MULTIPLY_CHILD);
    auto child_started = (child_start == RELAY_FLAG_ON);
    child.SetValid(child_started);
    {
      auto child_instr = child.NewInstr(RELAY_VECTOR_MULTIPLY_CHILD_INSTR);
      child_instr.SetDecode(child_started);

      auto addr_offset = cntr * RELAY_VECTOR_DATA_BYTES;
      auto op0_addr = m.state(RELAY_VECTOR_MULTIPLY_OP0_ADDR) + addr_offset;
      auto op1_addr = m.state(RELAY_VECTOR_MULTIPLY_OP1_ADDR) + addr_offset;
      auto output_addr =
          m.state(RELAY_VECTOR_MULTIPLY_OUTPUT_ADDR) + addr_offset;
      // uninterpreted add function
      auto result = bv_multiply(RELAY_LOAD_WORD(memory, op0_addr),
                                RELAY_LOAD_WORD(memory, op1_addr));

      auto next_cntr = cntr + BvConst(1, RELAY_VECTOR_OP_CNTR_BW);
      auto continue_cond = (next_cntr != m.state(RELAY_VECTOR_OP_SIZE));
      auto next_child_start = RELAY_ITE_FLAG(continue_cond);
      auto next_vector_multiply_enable = RELAY_ITE_FLAG(continue_cond);
      auto lstm_state = m.state(RELAY_LSTM_STATE);
      auto next_lstm_state =
          Ite(continue_cond, lstm_state, m.state(RELAY_LSTM_RETURN_STATE));

      child_instr.SetUpdate(memory,
                            RELAY_STORE_WORD(memory, output_addr, result));
      child_instr.SetUpdate(cntr, next_cntr);
      child_instr.SetUpdate(child_start, next_child_start);
      child_instr.SetUpdate(vector_multiply_enable,
                            next_vector_multiply_enable);
      child_instr.SetUpdate(lstm_state, next_lstm_state);
    }
  }
}

void DefineVectorSigmoid(Ila& m) {
  auto vector_child = m.child(RELAY_VECTOR_OP_CHILD);
  auto instr = vector_child.NewInstr(RELAY_VECTOR_SIGMOID);

  auto vector_sigmoid_enable = m.state(RELAY_VECTOR_SIGMOID_ENABLE);
  auto child_start = m.state(RELAY_VECTOR_SIGMOID_START);

  instr.SetDecode(
      (vector_sigmoid_enable == RELAY_FLAG_ON) &
      (m.state(RELAY_VECTOR_OP_SIZE) != BvConst(0, RELAY_VECTOR_OP_SIZE_BW)) &
      (child_start == RELAY_FLAG_OFF));

  auto cntr = m.state(RELAY_VECTOR_OP_CNTR);

  auto memory = m.state(RELAY_MEMORY);

  instr.SetUpdate(child_start, BvConst(RELAY_FLAG_ON, RELAY_FLAG_BW));
  instr.SetUpdate(cntr, BvConst(0, RELAY_VECTOR_OP_CNTR_BW));

  {
    auto child = vector_child.NewChild(RELAY_VECTOR_SIGMOID_CHILD);
    auto child_started = (child_start == RELAY_FLAG_ON);
    child.SetValid(child_started);
    {
      auto child_instr = child.NewInstr(RELAY_VECTOR_SIGMOID_CHILD_INSTR);
      child_instr.SetDecode(child_started);

      auto addr_offset = cntr * RELAY_VECTOR_DATA_BYTES;
      auto op0_addr = m.state(RELAY_VECTOR_SIGMOID_OP0_ADDR) + addr_offset;
      auto output_addr =
          m.state(RELAY_VECTOR_SIGMOID_OUTPUT_ADDR) + addr_offset;
      // uninterpreted sigmoid function
      auto result = bv_sigmoid(RELAY_LOAD_WORD(memory, op0_addr));

      auto next_cntr = cntr + 1;
      auto continue_cond = (next_cntr != m.state(RELAY_VECTOR_OP_SIZE));
      auto next_child_start = RELAY_ITE_FLAG(continue_cond);
      auto next_vector_sigmoid_enable = RELAY_ITE_FLAG(continue_cond);

      auto lstm_state = m.state(RELAY_LSTM_STATE);
      auto next_lstm_state =
          Ite(continue_cond, lstm_state, m.state(RELAY_LSTM_RETURN_STATE));

      child_instr.SetUpdate(memory,
                            RELAY_STORE_WORD(memory, output_addr, result));
      child_instr.SetUpdate(cntr, next_cntr);
      child_instr.SetUpdate(child_start, next_child_start);
      child_instr.SetUpdate(vector_sigmoid_enable, next_vector_sigmoid_enable);
      child_instr.SetUpdate(lstm_state, next_lstm_state);
    }
  }
}

void DefineVectorTanh(Ila& m) {
  auto vector_child = m.child(RELAY_VECTOR_OP_CHILD);
  auto instr = vector_child.NewInstr(RELAY_VECTOR_TANH);

  auto vector_tanh_enable = m.state(RELAY_VECTOR_TANH_ENABLE);
  auto child_start = m.state(RELAY_VECTOR_TANH_START);

  instr.SetDecode(
      (vector_tanh_enable == RELAY_FLAG_ON) &
      (m.state(RELAY_VECTOR_OP_SIZE) != BvConst(0, RELAY_VECTOR_OP_SIZE_BW)) &
      (child_start == RELAY_FLAG_OFF));

  auto cntr = m.state(RELAY_VECTOR_OP_CNTR);

  auto memory = m.state(RELAY_MEMORY);

  instr.SetUpdate(child_start, BvConst(RELAY_FLAG_ON, RELAY_FLAG_BW));
  instr.SetUpdate(cntr, BvConst(0, RELAY_VECTOR_OP_CNTR_BW));

  {
    auto child = vector_child.NewChild(RELAY_VECTOR_TANH_CHILD);
    auto child_started = (child_start == RELAY_FLAG_ON);
    child.SetValid(child_started);
    {
      auto child_instr = child.NewInstr(RELAY_VECTOR_TANH_CHILD_INSTR);
      child_instr.SetDecode(child_started);

      auto addr_offset = cntr * RELAY_VECTOR_DATA_BYTES;
      auto op0_addr = m.state(RELAY_VECTOR_TANH_OP0_ADDR) + addr_offset;
      auto output_addr = m.state(RELAY_VECTOR_TANH_OUTPUT_ADDR) + addr_offset;
      // uninterpreted sigmoid function
      auto result = bv_tanh(RELAY_LOAD_WORD(memory, op0_addr));

      auto next_cntr = cntr + 1;
      auto continue_cond = (next_cntr != m.state(RELAY_VECTOR_OP_SIZE));
      auto next_child_start = RELAY_ITE_FLAG(continue_cond);
      auto next_vector_tanh_enable = RELAY_ITE_FLAG(continue_cond);

      auto lstm_state = m.state(RELAY_LSTM_STATE);
      auto next_lstm_state =
          Ite(continue_cond, lstm_state, m.state(RELAY_LSTM_RETURN_STATE));

      child_instr.SetUpdate(memory,
                            RELAY_STORE_WORD(memory, output_addr, result));
      child_instr.SetUpdate(cntr, next_cntr);
      child_instr.SetUpdate(child_start, next_child_start);
      child_instr.SetUpdate(vector_tanh_enable, next_vector_tanh_enable);
      child_instr.SetUpdate(lstm_state, next_lstm_state);
    }
  }
}

} // namespace relay

} // namespace ilang
