#include <stdint.h>
#include <stdlib.h>
#include "bsg_manycore.h"
#include "bsg_set_tile_x_y.h"

#define BSG_TILE_GROUP_X_DIM bsg_tiles_X
#define BSG_TILE_GROUP_Y_DIM bsg_tiles_Y
#include "bsg_tile_group_barrier.h"
INIT_TILE_GROUP_BARRIER(r_barrier, c_barrier, 0, bsg_tiles_X-1, 0, bsg_tiles_Y-1);

typedef struct {
    /* float *state; */
    /* float *next_state; */
    float state[84*84*4];
    float next_state[84*84*4];
    float reward;
    uint32_t action;
    uint32_t done;
} Transition;

int  __attribute__ ((noinline)) kernel_optimizer(int flag, float *remem_state, float *remem_next_state, float *remem_reward, uint32_t *remem_action, uint32_t *remem_done,
        uint32_t position, float *state, float *next_state, float *others,
        float *weight, float *weight_T, float *weight_grad, float *bias_grad, int layer, int N, int block_size_x) {

    int start_x = block_size_x * (__bsg_tile_group_id_y * __bsg_grid_dim_x + __bsg_tile_group_id_x); 
    int step = bsg_tiles_X * bsg_tiles_Y;

    int state_size = 84*84*4;

    if (flag == 0) {
        // push replay memory        
        if (__bsg_id == 1) {
            for (int i = 0; i < state_size; i++) {
                remem_state[position*state_size+i] = state[i];
                remem_next_state[position*state_size+i] = next_state[i];
            }
            remem_reward[position] = others[0];
            remem_action[position] = others[1];
            remem_done[position] = others[2];
        }
    }
    else if (flag == 1) {
        // sample replay memory, now the position means the max size
        uint32_t sample_position = rand()%position;
        if (__bsg_id == 1) {
            for (int i = 0; i < state_size; i++) {
                state[i] = remem_state[sample_position*state_size+i];
                next_state[i] = remem_next_state[sample_position*state_size+i];
            }
            others[0] = remem_reward[sample_position];
            others[1] = remem_action[sample_position];
            others[2] = remem_done[sample_position];
        }
    }
    else {
        // optimizer 
        float lr = 0.001;
        float w, wT, dw;
        int i, j, k, z;
        int w_offset, wT_offset, dw_offset;
        if (layer == 4) {
            // fc2 
            for (int out_n = 0; out_n < 4; out_n++) {
                for (int in_n = start_x + __bsg_id; in_n < 512; in_n += step) {
                    // weight for fp
                    i = in_n/18;
                    k = in_n%18;
                    j = out_n;
                    w_offset = 4+i + i*(18*4) + j*18 + k;
                    dw_offset = in_n + out_n*540;
                    weight[w_offset] -= lr*weight_grad[dw_offset];
                    // weight for bp
                    i = out_n;
                    j = in_n/32;
                    z = in_n%32;
                    wT_offset = z*(16+16*18) + 16 + j*18 + i;
                    weight_T[wT_offset] = weight[w_offset];
                }
            }
            // bias
            if (__bsg_id==0)
                for (int index = 0; index < 4; index++)
                    weight[index] -= lr*bias_grad[index];
        }
        else if (layer == 3) {
            // fc1
            for (int out_n = 0; out_n < 512; out_n++) {
                for (int in_n = start_x + __bsg_id; in_n < 3136; in_n += step) {
                    // weight for fp
                    i = in_n/18;
                    k = in_n%18;
                    j = out_n/32;
                    z = out_n%32;
                    if (i==0) 
                        w_offset = z*(16+16*18) + 16 + j*18 + k;
                    else
                        w_offset = 32*(16+16*18) + (i-1)*(1+32*(16*18)) +  1+z*(16*18) + j*18 + k;
                    dw_offset = in_n + out_n*3168;
                    weight[w_offset] -= lr*weight_grad[dw_offset];
                    // weight for bp
                    i = out_n%18;
                    int slides = out_n/18;
                    int repeat = in_n/512;
                    z = (in_n-repeat*512)%32;
                    j = (in_n-repeat*512)/32;
                    if (slides==0)
                        wT_offset = repeat*(32*(16+16*18)+28*(1+32*(16*18))) + z*(16+16*18) + 16 + j*18 + i;
                    else
                        wT_offset = repeat*(32*(16+16*18)+28*(1+32*(16*18))) + 32*(16+16*18) + (slides-1)*(1+32*(16*18)) + 1 + z*(16*18) + j*18 + i;
                    weight_T[wT_offset] = weight[w_offset];
                }
            }
            // bias
            for (int index = start_x + __bsg_id; index < 512; index += step) {
                i = index/16;
                k = index%16;
                w_offset = i*(16+16*18) + k;
                weight[w_offset] -= lr*bias_grad[index];
            }
        }
        else if (layer == 2) {
            // conv3
            for (int d4 = start_x + __bsg_id; d4 < 64; d4 += step) {
                for (int d3 = 0; d3 < 64; d3++) {
                    for (int d2 = 0; d2 < 3; d2++) {
                        for (int d1 = 0; d1 < 3; d1++) {
                            // weight for fp
                            i = d4%16;
                            j = d4/16;
                            k = d3%2;
                            z = d3/2;
                            w_offset = j*(16+32*16*3*2*3) + 16 + z*(16*3*2*3) + i*(3*2*3) + (2-d2)*(2*3) + k*3 + d1;
                            dw_offset = d4*64*9 + d3*9 + d2 + d1*3;
                            weight[w_offset] -= lr*weight_grad[dw_offset];
                            // weight for bp
                            wT_offset = (d3/16)*(16+32*16*3*2*3) + 16 + (d4/2)*(16*3*2*3) + (d3%16)*(3*2*3) + (2-(2-d2))*(2*3) + (d4%2)*3 + (2-d1);
                            weight_T[wT_offset] = weight[w_offset];
                        }
                    }
                }
            }
            // bias
            for (int index = start_x + __bsg_id; index < 64; index += step) {
                i = index/16;
                k = index%16;
                w_offset = i*(16+32*16*3*2*3) + k;
                weight[w_offset] -= lr*bias_grad[index];
            }
        }
        else if (layer == 1) {
            // conv2
            for (int d4 = start_x + __bsg_id; d4 < 64; d4 += step) {
                for (int d3 = 0; d3 < 32; d3++) {
                    for (int d2 = 0; d2 < 4; d2++) {
                        for (int d1 = 0; d1 < 4; d1++) {
                            // weight for fp
                            i = d4%16;
                            j = d4/16;
                            w_offset = j*(16+32*16*4*4) + 16 + d3*(16*4*4) + i*(4*4) + (3-d2)*4 + d1;
                            dw_offset = d4*32*4*4 + d3*4*4 + d2 + d1*4;
                            weight[w_offset] -= lr*weight_grad[dw_offset];
                            // weight for bp 
                            i=d1%2;
                            j=d1/2;
                            k=d2%2;
                            z=d2/2;
                            wT_offset = (i+k*2)*(32+2*16*16*2*2*4) + (d3/16)*(16+16*16*2*2*4) + 16 + (d4/4)*(16*2*2*4) + (d3%16)*(2*2*4) + z*(2*4) + j*4 + (d4%4);
                            weight_T[wT_offset] = weight[w_offset];
                        }
                    }
                }
            }
            // bias
            for (int index = start_x + __bsg_id; index < 64; index += step) {
                i = index/16;
                k = index%16;
                w_offset = i*(16+32*16*4*4) + k;
                weight[w_offset] -= lr*bias_grad[index];
            }
        }
        else if (layer == 0) {
            // conv1
            for (int d4 = start_x + __bsg_id; d4 < 32; d4 += step) {
                for (int d3 = 0; d3 < 4; d3++) {
                    for (int d2 = 0; d2 < 8; d2++) {
                        for (int d1 = 0; d1 < 8; d1++) {
                            i = d4%16;
                            j = d4/16;
                            k = d1/4;
                            z = d2/4;
                            w_offset = j*(16+4*4*16*4*4) + 16 + (k*2+z)*(4*16*4*4) + d3*(16*4*4) + i*(4*4) + (3-d2+z*4)*4 + (d1-k*4);
                            dw_offset = d4*8*8 + d3*32*8*8 + d2 + d1*8;
                            weight[w_offset] -= lr*weight_grad[dw_offset];
                        }
                    }
                }
            }
            // bias
            for (int index = start_x + __bsg_id; index < 32; index += step) {
                i = index/16;
                k = index%16;
                w_offset = i*(16+4*4*16*4*4) + k;
                weight[w_offset] -= lr*bias_grad[index];
            }
        }
        else {
            // test; these shouldn't happend
            for (int out_n = 0; out_n < 4; out_n++) {
                for (int in_n = __bsg_id; in_n < 16; in_n += step) {
                    w_offset = out_n + in_n*4;
                    dw_offset = in_n + out_n*16;
                    weight[start_x + w_offset] -= 0.001*weight_grad[start_x + dw_offset];
                }
            }
        }
    }

    bsg_tile_group_barrier(&r_barrier, &c_barrier); 

  return 0;
}
