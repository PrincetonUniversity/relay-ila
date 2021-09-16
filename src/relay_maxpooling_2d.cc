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

// File: relay_maxpooling_2d.cc

// tvm.relay.nn.max_pool2d(data, pool_size=(1, 1), strides=(1, 1), padding=(0,
// 0), layout='NCHW', ceil_mode=False)
// This file contains the model of maxpooling 2d instruction of Relay IR

#include <ilang/util/log.h>

#include <relay/relay_top.h>
#include <relay/uninterpreted_func.h>

namespace ilang {

namespace relay {

void AddChild_Loop_Op(Ila& m);
void AddChild_Find_Max(Ila& m);

void DefineMaxpooling2D(Ila& m) {

  // function arguments
  auto height_in = m.input(DATA_IN_Y); // 32
  auto width_in = m.input(DATA_IN_X);

  auto pool_y_in = m.input(POOL_SIZE_Y_IN); // 8
  auto pool_x_in = m.input(POOL_SIZE_X_IN);

  auto stride_y_in = m.input(STRIDES_Y_IN); // 8
  auto stride_x_in = m.input(STRIDES_X_IN);

  auto pool_y = m.state(MAXPOOLING_POOL_Y); // 32
  auto pool_x = m.state(MAXPOOLING_POOL_X);

  auto stride_y = m.state(MAXPOOLING_STRIDE_Y); // 32
  auto stride_x = m.state(MAXPOOLING_STRIDE_X);

  // states used for child
  auto flag_start = m.state(MAXPOOLING_START_FLAG); // ON/OFF

  // maxpooling state machine
  auto state = m.state(MAXPOOLING_STATE);

  auto cntr_X = m.state(MAXPOOLING_X_LOOP_CNTR);
  auto cntr_Y = m.state(MAXPOOLING_Y_LOOP_CNTR);

  auto height_out = m.state(MAXPOOLING_DATA_OUT_HEIGHT);
  auto width_out = m.state(MAXPOOLING_DATA_OUT_WIDTH);

  {
    auto instr = m.NewInstr(F_MAXPOOING_2D);

    auto func_id_match = (m.input(RELAY_FUNC_ID_IN) == F_MAXPOOLING_2D_ID);
    auto func_run = (m.input(RELAY_FUNC_RUN_IN) == RELAY_FUNC_RUN_ON);

    instr.SetDecode(func_id_match & func_run);

    auto stride_y_32 = ZExt(stride_y_in, 32);
    auto stride_x_32 = ZExt(stride_x_in, 32);
    // calculate the output tensor size
    auto height_out_tmp = height_in / stride_y_32;
    auto width_out_tmp = width_in / stride_x_32;

    // states update for child
    instr.SetUpdate(flag_start,
                    BvConst(FLAG_ON, MAXPOOLING_START_FLAG_BITWIDTH));

    instr.SetUpdate(
        state, BvConst(MAXPOOLING_STATE_FIND_MAX, MAXPOOLING_STATE_BITWIDTH));

    instr.SetUpdate(cntr_X, BvConst(0, MAXPOOLING_X_LOOP_CNTR_BITWIDTH));
    instr.SetUpdate(cntr_Y, BvConst(0, MAXPOOLING_Y_LOOP_CNTR_BITWIDTH));

    instr.SetUpdate(height_out, height_out_tmp);
    instr.SetUpdate(width_out, width_out_tmp);

    instr.SetUpdate(pool_x, ZExt(pool_x_in, RELAY_FUNC_ADDR_IN_BITWIDTH));
    instr.SetUpdate(pool_y, ZExt(pool_y_in, RELAY_FUNC_ADDR_IN_BITWIDTH));

    instr.SetUpdate(stride_x, stride_x_32);
    instr.SetUpdate(stride_y, stride_y_32);

    // add child to do the loop
    AddChild_Loop_Op(m);
  }
}

void AddChild_Loop_Op(Ila& m) {

  auto child = m.NewChild("maxpooling_loop_op");

  auto flag_start = m.state(MAXPOOLING_START_FLAG); // ON/OFF

  // maxpooling state machine
  auto state = m.state(MAXPOOLING_STATE);

  child.SetValid(flag_start == FLAG_ON);
  child.SetFetch(BvConst(1, 1));

  auto cntr_X = m.state(MAXPOOLING_X_LOOP_CNTR);
  auto cntr_Y = m.state(MAXPOOLING_Y_LOOP_CNTR);

  auto height_out = m.state(MAXPOOLING_DATA_OUT_HEIGHT);
  auto width_out = m.state(MAXPOOLING_DATA_OUT_WIDTH);

  auto pool_y = m.state(MAXPOOLING_POOL_Y);
  auto pool_x = m.state(MAXPOOLING_POOL_X);

  auto stride_y = m.state(MAXPOOLING_STRIDE_Y);
  auto stride_x = m.state(MAXPOOLING_STRIDE_X);

  // tensor memory state
  auto tensor = m.state(RELAY_TENSOR_MEM);

  // child states for find max
  auto cntr_max_X = child.NewBvState(MAXPOOLING_FIND_MAX_CNTR_X,
                                     MAXPOOLING_FIND_MAX_CNTR_BITWIDTH);
  auto cntr_max_Y = child.NewBvState(MAXPOOLING_FIND_MAX_CNTR_Y,
                                     MAXPOOLING_FIND_MAX_CNTR_BITWIDTH);
  auto result_max = child.NewBvState(MAXPOOLING_FIND_MAX_RESULT,
                                     MAXPOOLING_FIND_MAX_RESULT_BITWIDTH);

  // child instruction 1 -- X loop parameter update
  {
    auto instr = child.NewInstr("child_loop_X_update");

    auto cond_flag = (flag_start == FLAG_ON);
    auto cond_state = (state == MAXPOOLING_STATE_INC_X);

    instr.SetDecode(cond_flag & cond_state);

    auto end_of_X = (cntr_X == width_out);

    auto cntr_X_new =
        Ite(end_of_X, BvConst(0, MAXPOOLING_X_LOOP_CNTR_BITWIDTH), cntr_X + 1);

    auto next_state = Ite(
        end_of_X, BvConst(MAXPOOLING_STATE_INC_Y, MAXPOOLING_STATE_BITWIDTH),
        BvConst(MAXPOOLING_STATE_FIND_MAX, MAXPOOLING_STATE_BITWIDTH));

    instr.SetUpdate(cntr_X, cntr_X_new);
    instr.SetUpdate(state, next_state);
  }

  // child instruction 2 -- Y loop parameters update
  {
    auto instr = child.NewInstr("child_loop_Y_update");

    auto cond_flag = (flag_start == FLAG_ON);
    auto cond_state = (state == MAXPOOLING_STATE_INC_Y);

    instr.SetDecode(cond_flag & cond_state);

    auto end_of_Y = (cntr_Y == (height_out - 1));
    auto cntr_Y_new = cntr_Y + 1;
    auto next_state =
        Ite(end_of_Y,
            BvConst(MAXPOOLING_STATE_DONE, MAXPOOLING_STATE_BITWIDTH),
            BvConst(MAXPOOLING_STATE_FIND_MAX, MAXPOOLING_STATE_BITWIDTH));

    instr.SetUpdate(cntr_Y, cntr_Y_new);
    instr.SetUpdate(state, next_state);
  }

  // child instruction 3 -- find max in the given coordinates
  {
    auto instr = child.NewInstr("child_call_find_max");

    auto cond_flag = (flag_start == FLAG_ON);
    auto cond_state = (state == MAXPOOLING_STATE_FIND_MAX);

    instr.SetDecode(cond_flag & cond_state);

    auto next_state =
        BvConst(MAXPOOLING_STATE_FIND_MAX_CHILD, MAXPOOLING_STATE_BITWIDTH);

    instr.SetUpdate(cntr_max_X,
                    BvConst(0, MAXPOOLING_FIND_MAX_CNTR_BITWIDTH));
    instr.SetUpdate(cntr_max_Y,
                    BvConst(0, MAXPOOLING_FIND_MAX_CNTR_BITWIDTH));
    instr.SetUpdate(state, next_state);

    AddChild_Find_Max(m);
  }

  // child instruction 4 -- write the max value back into the memory
  {
    auto instr = child.NewInstr("child_write_max_value");
    auto cond_flag = (flag_start == FLAG_ON);
    auto cond_state = (state == MAXPOOLING_STATE_WRITE);

    instr.SetDecode(cond_flag & cond_state);

    auto addr = cntr_X + cntr_Y * width_out;
    auto next_state =
        BvConst(MAXPOOLING_STATE_INC_X, MAXPOOLING_STATE_BITWIDTH);

    instr.SetUpdate(tensor, Store(tensor, addr, result_max));
    instr.SetUpdate(state, next_state);
  }
}

void AddChild_Find_Max(Ila& m) {

  auto child_loop = m.child("maxpooling_loop_op");
  auto child_find_max = child_loop.NewChild("maxpooling_find_max_loop");

  auto state = m.state(MAXPOOLING_STATE);

  child_find_max.SetValid(state == MAXPOOLING_STATE_FIND_MAX_CHILD);
  child_find_max.SetFetch(BvConst(1, 1));

  auto height_in = m.input(DATA_IN_Y); // 32
  auto width_in = m.input(DATA_IN_X);

  auto pool_y = m.state(MAXPOOLING_POOL_Y);
  auto pool_x = m.state(MAXPOOLING_POOL_X);

  auto stride_y = m.state(MAXPOOLING_STRIDE_Y);
  auto stride_x = m.state(MAXPOOLING_STRIDE_X);

  auto out_y = m.state(MAXPOOLING_Y_LOOP_CNTR);
  auto out_x = m.state(MAXPOOLING_X_LOOP_CNTR); // 32

  auto cntr_max_y = child_loop.state(MAXPOOLING_FIND_MAX_CNTR_Y);
  auto cntr_max_x = child_loop.state(MAXPOOLING_FIND_MAX_CNTR_X);

  auto result = child_loop.state(MAXPOOLING_FIND_MAX_RESULT);

  // tensor memory state
  auto tensor = m.state(RELAY_TENSOR_MEM);

  // instruction finding the max value in the pooling window
  {
    auto instr = child_find_max.NewInstr("maxpooling_find_max_op");

    auto cntr_cond = (cntr_max_y < pool_y) | (cntr_max_x < pool_x);
    auto state_cond = (state == MAXPOOLING_STATE_FIND_MAX_CHILD);

    instr.SetDecode(cntr_cond & state_cond);

    // base coordinates of the pooling window
    auto win_x_base = out_x * stride_x;
    auto win_y_base = out_y * stride_y;

    // coordinates within the pooling window

    // coordinates in the 2D tensor
    auto tensor_x = win_x_base + cntr_max_x;
    auto tensor_y = win_y_base + cntr_max_y;

    // calculate the memory address according to the tensor coordinates
    auto addr = tensor_y * width_in + tensor_x;

    // fetch the data in the memory
    auto data = Load(tensor, addr);

    // auto result_tmp = Ite(cntr_find_max == 0, data,
    //                       Ite(data > result, data, result));

    // use uninterpreted function
    auto result_tmp = Ite(
      (cntr_max_x == 0) & (cntr_max_y == 0), data, adpfloat_max(result, data)
    );

    // state updates
    
    auto x_done = cntr_max_x == pool_x - 1;
    auto y_done = cntr_max_y == pool_y - 1;

    auto next_state = Ite(
        y_done & x_done, BvConst(MAXPOOLING_STATE_WRITE, MAXPOOLING_STATE_BITWIDTH),
        BvConst(MAXPOOLING_STATE_FIND_MAX_CHILD, MAXPOOLING_STATE_BITWIDTH));

    auto zero = BvConst(0, RELAY_FUNC_ADDR_IN_BITWIDTH);
    instr.SetUpdate(cntr_max_x, Ite(x_done, zero, cntr_max_x + 1));
    instr.SetUpdate(cntr_max_y, Ite(x_done, cntr_max_y + 1, cntr_max_y));

    instr.SetUpdate(result, result_tmp);
    instr.SetUpdate(state, next_state);
  }
}

} // namespace relay

} // namespace ilang
