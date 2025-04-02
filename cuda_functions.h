
#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include <cuda_runtime.h>

__device__ double function1(double x1, double x2);
__device__ double function2(double x1, double x2);
__device__ double function3(double x1, double x2);
__device__ double callFunction(int func_id, double x1, double x2);

__global__ void integrateKernel(int func_id, double x1_start, double x1_end,
                                double x2_start, double x2_end, double *result,
                                int steps_x, int steps_y);

void integrateCUDA(int func_num, double x_start, double x_end, double y_start, double y_end,
                   int steps_x, int steps_y, double &result);

double adaptiveIntegrateCUDA(int func_num, double x_start, double x_end, double y_start, double y_end,
                             double abs_err, double rel_err, int init_steps_x, int init_steps_y, int max_iter,
                             double &achieved_abs_err, double &achieved_rel_err);

void readConfig(const char* filename, double& abs_err, double& rel_err, double& x_start, double& x_end,
                double& y_start, double& y_end, int& init_steps_x, int& init_steps_y, int& max_iter);

#endif // CUDA_FUNCTIONS_H
