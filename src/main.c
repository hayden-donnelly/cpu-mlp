#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "mnist.h"

#define INPUT_DIM 784 // 28*28
#define NUM_HIDDEN_LAYERS 3
#define HIDDEN_DIM 256
#define OUTPUT_DIM 10
#define BATCH_SIZE 1

const int input_layer_weight_count = INPUT_DIM * HIDDEN_DIM;
const int hidden_layer_weight_count = HIDDEN_DIM * HIDDEN_DIM;
const int output_layer_weight_count = HIDDEN_DIM * OUTPUT_DIM;
 
const double two_pi = 2.0*3.14159265358979323846;

typedef struct
{
    float* weights;
    float* weight_grads;
    float* activations_out;
    float* activations;
} params_t;

float random_normal()
{
    const double epsilon = DBL_MIN;
    static float z1, z2;
    static int generate;
    
    generate = !generate;
    if(!generate)
    {
        return z2;
    }

    double u1, u2;
    do
    {
       u1 = rand() * (1.0 / RAND_MAX);
       u2 = rand() * (1.0 / RAND_MAX);
    }
    while (u1 <= epsilon);

    z1 = (float)(sqrt(-2.0 * log(u1)) * cos(two_pi * u2));
    z2 = (float)(sqrt(-2.0 * log(u1)) * sin(two_pi * u2));
    return z1;
}

params_t* init_params()
{
    const int total_weight_count = 
        (NUM_HIDDEN_LAYERS * hidden_layer_weight_count)
        + input_layer_weight_count + output_layer_weight_count;
    
    params_t* params = (params_t*)malloc(sizeof(params_t));
    const size_t weights_size = sizeof(float) * total_weight_count;
    params->weights = (float*)malloc(weights_size);
    params->weight_grads = (float*)malloc(weights_size);
    for(int i = 0; i < total_weight_count; i++)
    {
        params->weights[i] = random_normal();
        params->weight_grads[i] = 0.0f;
    }

    params->activations_out = (float*)malloc(sizeof(float) * OUTPUT_DIM);
    for(int i = 0; i < OUTPUT_DIM; i++)
    {
        params->activations_out[i] = 0.0f;
    }

    const int num_inter_activations = (NUM_HIDDEN_LAYERS+1) * HIDDEN_DIM;
    params->activations = (float*)malloc(sizeof(float) * num_inter_activations);
    for(int i = 0; i < num_inter_activations; i++)
    {
        params->activations[i] = 0.0f;
    }
    return params;
}

static inline void scalar_relu(float* a)
{
    *a = (*a > 0.0f) ? *a : 0.0f;
}

static inline void sigmoid(float *a)
{
    *a = 1.0f / (1.0f + (float)exp(-(double)*a));
}

static inline void vec_mat_mul_sigmoid(
    float* mat, float* vec_in, float* vec_out, 
    const int in_dim, const int out_dim
){
    int mat_offset = 0;
    for(int out_idx = 0; out_idx < out_dim; out_idx++)
    {
        vec_out[out_idx] = 0.0f;
        for(int in_idx = 0; in_idx < in_dim; in_idx++)
        {
            vec_out[out_idx] += mat[mat_offset++] * vec_in[in_idx];
        }
        sigmoid(vec_out + out_idx);
    }
}

static inline void vec_mat_mul_sigmoid_backward(
    float* mat, float* vec_in, 
    float* vec_out, float* vec_out_grad,
    float* mat_grad, float* vec_in_grad,
    const int in_dim, const int out_dim
){
    // Notation: f: R^(vec_in) -> R^(vec_out), f(x) = Wx.
    float vec_mat_mul_grad[out_dim];
    for(int i = 0; i < out_dim; i++)
    {
        // dsig_i/df_i = (sig(f_i) * (1 - sig(f_i)).
        // dL/df_i = dsig_i/df_i * dL/dsig_i.
        vec_mat_mul_grad[i] = (vec_out[i] * (1.0f - vec_out[i])) * vec_out_grad[i];
    }
    for(int in_idx = 0; in_idx < in_dim; in_idx++)
    {
        vec_in_grad[in_idx] = 0.0f;
        for(int out_idx = 0; out_idx < out_dim; out_idx++)
        {
            // df_j/dx_i = W_ij.
            // dL/dx_i = [df_1/dx_i * dL/df_1, df_2/dx_i * dL/df_2, ... , df_n/dx_i * dL/df_n.
            // Take a transposed view of the matrix.
            vec_in_grad[in_idx] += mat[out_idx * in_dim + in_idx] * vec_mat_mul_grad[out_idx];
        }
    }
    int mat_grad_offset = 0;
    for(int out_idx = 0; out_idx < out_dim; out_idx++)
    {
        for(int in_idx = 0; in_idx < in_dim; in_idx++)
        {
            // dL/dW = dL/df * x^T.
            // dL/dW_ij = x_j * dL/df_i.
            mat_grad[mat_grad_offset++] = vec_in[in_idx] * vec_mat_mul_grad[out_idx];
        }
    }
}

static inline void vec_mat_mul_softmax(
    float* mat, float* vec_in, float* vec_out, 
    const int in_dim, const int out_dim
){
    int mat_offset = 0;
    double exp_layer_sum = 0.0f;
    for(int out_idx = 0; out_idx < out_dim; out_idx++)
    {
        vec_out[out_idx] = 0.0f;
        for(int in_idx = 0; in_idx < in_dim; in_idx++)
        {
            vec_out[out_idx] += mat[mat_offset++] * vec_in[in_idx];
        }
        exp_layer_sum += exp((double)vec_out[out_idx]);
    }
    for(int i = 0; i < out_dim; i++)
    {
        vec_out[i] = (float)(exp((double)vec_out[i]) / exp_layer_sum); 
    }
}

void mse(float* label, float* pred_label, float* loss, const int num_classes)
{
    float error_sum = 0.0f;
    for(int i = 0; i < num_classes; i++)
    {
        const float error = label[i] - pred_label[i];
        error_sum += error * error;
    }
    *loss = error_sum / (float)num_classes; 
}

void mse_backward(float* label, float* pred_label, float* pred_label_grad, const int num_classes)
{
    for(int i = 0; i < num_classes; i++)
    {
        pred_label_grad[i] = 2.0f * (pred_label[i] - label[i]) / (float)num_classes;
    }
}

void print_output(float* out, const int n)
{
    for(int i = 0; i < n; i++)
    {
        printf("%f ", out[i]);
    }
    printf("\n");
}

void forward_pass(float* weights, float* in, float* out, float* activations)
{
    vec_mat_mul_sigmoid(weights, in, activations, INPUT_DIM, HIDDEN_DIM);
    printf("Input layer:\n");
    print_output(activations, HIDDEN_DIM);
    int activations_offset = 0;
    int weight_offset = input_layer_weight_count;
    printf("num hidden layers %d\n", NUM_HIDDEN_LAYERS);
    for(int i = 0; i < NUM_HIDDEN_LAYERS; i++)
    {
        const int next_activations_offset = activations_offset + HIDDEN_DIM;
        vec_mat_mul_sigmoid(
            weights + weight_offset, 
            activations + activations_offset, 
            activations + next_activations_offset,
            HIDDEN_DIM,
            HIDDEN_DIM
        );
        if(i == 2) {return;}
        printf("Hidden layer %d:\n", i);
        print_output(activations + next_activations_offset, HIDDEN_DIM);
        activations_offset = next_activations_offset;
        weight_offset += hidden_layer_weight_count;
    }
    vec_mat_mul_sigmoid(
        weights + weight_offset, 
        activations + activations_offset, 
        out,
        HIDDEN_DIM,
        OUTPUT_DIM
    );
    printf("Output layer:\n");
    print_output(out, OUTPUT_DIM);
}

int main()
{
    params_t* params = init_params();
    float in[INPUT_DIM] = {1.0f};
    printf("init done\n");
    forward_pass(
        params->weights, 
        in,
        params->activations_out, 
        params->activations
    );
    printf("forward done\n");
    int activation_offset = 0;
    for(int i = 0; i < NUM_HIDDEN_LAYERS; i++)
    {
        printf("layer %d\n", i);
        print_output(params->activations + activation_offset, HIDDEN_DIM);
        activation_offset += HIDDEN_DIM;
    }
    for(int i = 0; i < OUTPUT_DIM; i++)
    {
        printf("%f ", params->activations_out[i]);
    }
    free(params->activations_out);
    free(params->activations);
    free(params->weights);
    free(params->weight_grads);
    free(params);
}
