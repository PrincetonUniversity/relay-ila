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

// File: relay_lstm.h

#ifndef RELAY_LSTM_H__
#define RELAY_LSTM_H__

namespace ilang {

namespace relay {

#define RELAY_LSTM_ADDR_BW 32

#define RELAY_LSTM_STATE "relay_lstm_state"
#define RELAY_LSTM_STATE_BW 8
#define RELAY_WAIT_STATE 0
#define RELAY_LSTM_DENSE_I2H_STATE 1
#define RELAY_LSTM_DENSE_H2H_STATE 2
#define RELAY_LSTM_ADD_DENSE_STATE 3
#define RELAY_LSTM_SIGMOID_STATE 4
#define RELAY_LSTM_CELL_TANH_STATE 5
#define RELAY_LSTM_OUTPUT_GATE_STATE 6
#define RELAY_LSTM_FORGET_GATE_STATE 7
#define RELAY_LSTM_INPUT_GATE_STATE 8
#define RELAY_LSTM_NEXT_CELL_STATE 9
#define RELAY_LSTM_NEXT_CELL_TANH_STATE 10
#define RELAY_LSTM_OUTPUT_STATE 11
#define RELAY_LSTM_END_STATE 12

#define RELAY_LSTM_MATRIX_VECTOR "relay_lstm_matrix_vector_module"
#define RELAY_LSTM_DENSE_I2H_INSTR "relay_lstm_dense_i2h_instr"
#define RELAY_LSTM_DENSE_H2H_INSTR "relay_lstm_dense_h2h_instr"
#define RELAY_LSTM_ADD_DENSE_INSTR "relay_lstm_add_dense_instr"
#define RELAY_LSTM_SIGMOID_INSTR "relay_lstm_sigmoid_instr"
#define RELAY_LSTM_CELL_TANH_INSTR "relay_lstm_cell_tanh_instr"
#define RELAY_LSTM_OUTPUT_GATE_INSTR "relay_lstm_output_gate_instr"
#define RELAY_LSTM_FORGET_GATE_INSTR "relay_lstm_forget_gate_instr"
#define RELAY_LSTM_INPUT_GATE_INSTR "relay_lstm_input_gate_instr"
#define RELAY_LSTM_NEXT_CELL_INSTR "relay_lstm_next_cell_instr"
#define RELAY_LSTM_NEXT_CELL_TANH_INSTR "relay_lstm_next_cell_tanh_instr"
#define RELAY_LSTM_OUTPUT_INSTR "relay_lstm_output_instr"

#define RELAY_LSTM_FLAG_BW 1
#define RELAY_LSTM_FLAG_ON 1
#define RELAY_LSTM_FLAG_OFF 0

#define RELAY_LSTM_RETURN_STATE "relay_lstm_return_state"

// inputs

#define RELAY_LSTM_IN_SIZE "relay_lstm_in_size"
#define RELAY_LSTM_OUT_SIZE "relay_lstm_out_size"
#define RELAY_LSTM_INPUT_ADDR "relay_lstm_input_addr"
#define RELAY_LSTM_CELL_ADDR "relay_lstm_cell_addr"
#define RELAY_LSTM_NEXT_CELL_ADDR "relay_lstm_next_cell_addr"
#define RELAY_LSTM_HIDDEN_ADDR "relay_lstm_hidden_addr"
#define RELAY_LSTM_NEXT_HIDDEN_ADDR "relay_lstm_next_hidden_addr"
#define RELAY_LSTM_I2H_WEIGHT_ADDR "relay_lstm_i2h_weight_addr"
#define RELAY_LSTM_H2H_WEIGHT_ADDR "relay_lstm_h2h_weight_addr"

#define RELAY_LSTM_I2H_BIAS_ADDR "relay_lstm_i2h_bias_addr"
#define RELAY_LSTM_H2H_BIAS_ADDR "relay_lstm_h2h_bias_addr"

#define RELAY_LSTM_TEMP_VECTOR0_ADDR "relay_lstm_temp_vector0_addr"
#define RELAY_LSTM_TEMP_VECTOR1_ADDR "relay_lstm_temp_vector1_addr"
#define RELAY_LSTM_TEMP_VECTOR2_ADDR "relay_lstm_temp_vector2_addr"

// states

#define RELAY_LSTM_START "relay_lstm_start"
#define RELAY_LSTM_STATE "relay_lstm_state"

} // namespace relay

} // namespace ilang

#endif // RELAY_LSTM_H__
