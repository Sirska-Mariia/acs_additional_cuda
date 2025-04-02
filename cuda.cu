#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <fstream>

__device__ double function1(double x1, double x2) {
    double result = 0.002;

    for (int i = -2; i <= 2; ++i) {
        for (int j = -2; j <= 2; ++j) {
            double x1_diff = x1 - 16 * j;
            double x2_diff = x2 - 16 * i;

            double x1_diff_sq = x1_diff * x1_diff;
            double x1_diff_6 = x1_diff_sq * x1_diff_sq * x1_diff_sq;

            double x2_diff_sq = x2_diff * x2_diff;
            double x2_diff_6 = x2_diff_sq * x2_diff_sq * x2_diff_sq;

            result += 1.0 / (5 * (i + 2) + j + 3 + x1_diff_6 + x2_diff_6);
        }
     }
    return 1.0 / result;
}

__device__ double function2(double x1, double x2) {
    const double a = 20.0, b = 0.2, c = 2.0 * M_PI;
    double sum_squares = x1 * x1 + x2 * x2;
    return -a * exp(-b * sqrt(0.5 * sum_squares)) - exp(0.5 * (cos(c*x1) + cos(c*x2))) + a + exp(1.0);
}

__device__ double function3(double x1, double x2) {
    const int m = 5;
    const double a1[] = {1, 2, 1, 1, 5}, a2[] = {4, 5, 1, 2, 4}, c[] = {2, 1, 4, 7, 2};
    double result = 0.0;
    for (int i = 0; i < m; ++i) {
        double dist2 = pow(x1 - a1[i], 2) + pow(x2 - a2[i], 2);
        result += c[i] * exp(-dist2 / M_PI) * cos(M_PI * dist2);
    }
    return -result;
}

__device__ double callFunction(int func_id, double x1, double x2) {
    switch (func_id) {
        case 1: return function1(x1, x2);
        case 2: return function2(x1, x2);
        case 3: return function3(x1, x2);
        default: return 0.0;
    }
}

//__global__ void integrateKernel(int func_id, double x1_start, double x1_end, double x2_start, double x2_end, double *result, int steps_x, int steps_y) {
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    int j = blockIdx.y * blockDim.y + threadIdx.y;
//
//    if (i >= steps_x || j >= steps_y) return;
//
//    double dx = (x1_end - x1_start) / steps_x;
//    double dy = (x2_end - x2_start) / steps_y;
//    double x = x1_start + (i + 0.5) * dx;
//    double y = x2_start + (j + 0.5) * dy;
//
//    double val = callFunction(func_id, x, y);
 //   atomicAdd(result, val * dx * dy);
//}

__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}


__global__ void integrateKernel(int func_id, double x1_start, double x1_end,
                                double x2_start, double x2_end, double *result,
                                int steps_x, int steps_y) {
    __shared__ double sharedResult[256];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int threadIdx_in_block = threadIdx.x + threadIdx.y * blockDim.x;

    if (threadIdx_in_block < 256) {
        sharedResult[threadIdx_in_block] = 0.0;
    }
    __syncthreads();

    if (i >= steps_x || j >= steps_y) return;

    double dx = (x1_end - x1_start) / (double)steps_x;
    double dy = (x2_end - x2_start) / (double)steps_y;
    double x = x1_start + (i + 0.5) * dx;
    double y = x2_start + (j + 0.5) * dy;

    double val = callFunction(func_id, x, y) * dx * dy;

    atomicAdd(&sharedResult[threadIdx_in_block], val);
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        double blockSum = 0.0;
        for (int idx = 0; idx < blockDim.x * blockDim.y; ++idx) {
            blockSum += sharedResult[idx];
        }

        atomicAdd(result, blockSum);
    }
}

double integrateCUDA(int func_id, double x1_start, double x1_end,
                     double x2_start, double x2_end, int steps_x, int steps_y) {
    double *d_result, h_result = 0.0;
    cudaMalloc((void**)&d_result, sizeof(double));
    cudaMemset(d_result, 0, sizeof(double));

    dim3 blockSize(16, 16);
    dim3 gridSize((steps_x + 15) / 16, (steps_y + 15) / 16);

    integrateKernel<<<gridSize, blockSize>>>(func_id, x1_start, x1_end, x2_start, x2_end, d_result, steps_x, steps_y);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    return h_result;
}

double adaptiveIntegrateCUDA(int func_id, double x1_start, double x1_end,
                             double x2_start, double x2_end, double abs_err, double rel_err,
                             int init_steps_x, int init_steps_y, int max_iter,
                             double& achieved_abs_err, double& achieved_rel_err) {
    int steps_x = init_steps_x;
    int steps_y = init_steps_y;

    double prev_result = integrateCUDA(func_id, x1_start, x1_end, x2_start, x2_end, steps_x, steps_y);

    for (int iter = 0; iter < max_iter; ++iter) {
        steps_x *= 2;
        steps_y *= 2;

        double current_result = integrateCUDA(func_id, x1_start, x1_end, x2_start, x2_end, steps_x, steps_y);

        achieved_abs_err = std::fabs(current_result - prev_result);
        achieved_rel_err = achieved_abs_err / (std::fabs(current_result) + 1e-10);

        if (achieved_abs_err < abs_err && achieved_rel_err < rel_err) {
            return current_result;
        }

        prev_result = current_result;
    }

    std::cout << "Final result after max iterations: " << prev_result << std::endl;
    std::cout << "Final achieved absolute error: " << achieved_abs_err << std::endl;
    std::cout << "Final achieved relative error: " << achieved_rel_err << std::endl;
    throw std::runtime_error("Could not achieve desired accuracy");
    return prev_result;
}

//void integrateCUDA(int func_id, double x1_start, double x1_end, double x2_start, double x2_end, int steps_x, int steps_y, double &result) {
//    double *d_result;
//    cudaMalloc((void**)&d_result, sizeof(double));
//    cudaMemset(d_result, 0, sizeof(double));
//
//    dim3 blockSize(16, 16);
//    dim3 gridSize((steps_x + 15) / 16, (steps_y + 15) / 16);
//
//    integrateKernel<<<gridSize, blockSize>>>(func_id, x1_start, x1_end, x2_start, x2_end, d_result, steps_x, steps_y);
//    cudaDeviceSynchronize();
//
//    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
//    cudaFree(d_result);
//}
//
////
void readConfig(const char* filename, double& abs_err, double& rel_err, double& x_start, double& x_end,
                double& y_start, double& y_end, int& init_steps_x, int& init_steps_y, int& max_iter) {

    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening configuration file");
    }

    bool found_abs_err = false;
    bool found_rel_err = false;
    bool found_x_start = false;
    bool found_x_end = false;
    bool found_y_start = false;
    bool found_y_end = false;
    bool found_init_steps_x = false;
    bool found_init_steps_y = false;
    bool found_max_iter = false;

    std::string line;
    int line_number = 0;

    while (std::getline(file, line)) {
        line_number++;

        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }

        if (line.find_first_not_of(" \t") == std::string::npos) {
            continue;
        }

        size_t equal_pos = line.find('=');
        if (equal_pos == std::string::npos) {
            throw std::runtime_error("Invalid configuration format on line " + std::to_string(line_number) + ": " + line);
        }

        std::string name = line.substr(0, equal_pos);
        std::string value = line.substr(equal_pos + 1);

        name.erase(0, name.find_first_not_of(" \t"));
        name.erase(name.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        if (name == "abs_err") {
            if (found_abs_err) {
                throw std::runtime_error("Duplicate parameter 'abs_err' on line " + std::to_string(line_number));
            }
            abs_err = std::stod(value);
            found_abs_err = true;
        } else if (name == "rel_err") {
            if (found_rel_err) {
                throw std::runtime_error("Duplicate parameter 'rel_err' on line " + std::to_string(line_number));
            }
            rel_err = std::stod(value);
            found_rel_err = true;
        } else if (name == "x_start") {
            if (found_x_start) {
                throw std::runtime_error("Duplicate parameter 'x_start' on line " + std::to_string(line_number));
            }
            x_start = std::stod(value);
            found_x_start = true;
        } else if (name == "x_end") {
            if (found_x_end) {
                throw std::runtime_error("Duplicate parameter 'x_end' on line " + std::to_string(line_number));
            }
            x_end = std::stod(value);
            found_x_end = true;
        } else if (name == "y_start") {
            if (found_y_start) {
                throw std::runtime_error("Duplicate parameter 'y_start' on line " + std::to_string(line_number));
            }
            y_start = std::stod(value);
            found_y_start = true;
        } else if (name == "y_end") {
            if (found_y_end) {
                throw std::runtime_error("Duplicate parameter 'y_end' on line " + std::to_string(line_number));
            }
            y_end = std::stod(value);
            found_y_end = true;
        } else if (name == "init_steps_x") {
            if (found_init_steps_x) {
                throw std::runtime_error("Duplicate parameter 'init_steps_x' on line " + std::to_string(line_number));
            }
            init_steps_x = std::stoi(value);
            found_init_steps_x = true;
        } else if (name == "init_steps_y") {
            if (found_init_steps_y) {
                throw std::runtime_error("Duplicate parameter 'init_steps_y' on line " + std::to_string(line_number));
            }
            init_steps_y = std::stoi(value);
            found_init_steps_y = true;
        } else if (name == "max_iter") {
            if (found_max_iter) {
                throw std::runtime_error("Duplicate parameter 'max_iter' on line " + std::to_string(line_number));
            }
            max_iter = std::stoi(value);
            found_max_iter = true;
        } else {
            throw std::runtime_error("Unknown parameter '" + name + "' on line " + std::to_string(line_number));
        }
    }

    if (!found_abs_err) {
        throw std::runtime_error("Missing parameter 'abs_err'");
    }
    if (!found_rel_err) {
        throw std::runtime_error("Missing parameter 'rel_err'");
    }
    if (!found_x_start) {
        throw std::runtime_error("Missing parameter 'x_start'");
    }
    if (!found_x_end) {
        throw std::runtime_error("Missing parameter 'x_end'");
    }
    if (!found_y_start) {
        throw std::runtime_error("Missing parameter 'y_start'");
    }
    if (!found_y_end) {
        throw std::runtime_error("Missing parameter 'y_end'");
    }
    if (!found_init_steps_x) {
        throw std::runtime_error("Missing parameter 'init_steps_x'");
    }
    if (!found_init_steps_y) {
        throw std::runtime_error("Missing parameter 'init_steps_y'");
    }
    if (!found_max_iter) {
        throw std::runtime_error("Missing parameter 'max_iter'");
    }
}
