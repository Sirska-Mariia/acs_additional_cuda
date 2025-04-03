//#include <iostream>
//#include <cmath>
//#include <CL/cl.h>
//#include <vector>
//
//// OpenCL initialization and setup functions
//cl_context create_context();
//cl_command_queue create_command_queue(cl_context context, cl_device_id device);
//cl_program create_program(cl_context context, cl_device_id device, const char* source);
//cl_kernel create_kernel(cl_program program, const char* kernel_name);
//void check_cl_error(cl_int err, const char* message);
//// Check for OpenCL compilation errors
//
//void check_cl_build_errors(cl_program program, cl_device_id device) {
//    size_t log_size;
//    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
//    char* log = new char[log_size];
//    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, nullptr);
//    std::cerr << "OpenCL Build Log: " << std::endl << log << std::endl;
//    delete[] log;
//}
//
//
//double integrate_with_opencl(double (*func)(double, double), double x1_min, double x1_max, double x2_min, double x2_max, int num_steps, int func_id) {
//    // OpenCL Setup
//    cl_int err;
//
//    // Step 1: Set up OpenCL environment
//    cl_platform_id platform;
//    err = clGetPlatformIDs(1, &platform, nullptr);
//    check_cl_error(err, "Unable to get platform ID.");
//
//    cl_device_id device;
//    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
//    check_cl_error(err, "Unable to get device ID.");
//
//    cl_context context = create_context();
//    cl_command_queue queue = create_command_queue(context, device);
//
//    // Step 2: Write OpenCL kernel code as a string (you can put this in a separate .cl file too)
//    const char* kernel_source = R"(
//__kernel void integrate_kernel(__global const double* x1_values,
//                                __global const double* x2_values,
//                                __global double* result,
//                                const int num_steps,
//                                const double x1_min, const double x1_max,
//                                const double x2_min, const double x2_max,
//                                const int func_id) {
//    int idx = get_global_id(0);
//    int idy = get_global_id(1);
//
//    if (idx >= num_steps || idy >= num_steps) return;
//
//    // Calculate the (x1, x2) values at this index
//    double x1 = x1_min + idx * ((x1_max - x1_min) / num_steps);
//    double x2 = x2_min + idy * ((x2_max - x2_min) / num_steps);
//
//    // Variable to store the result of the function
//    double result_value = 0.0;
//
//    // Select which function to calculate based on func_id
//    if (func_id == 1) {
//        // function1
//        result_value = 0.002;
//        for (int i = -2; i <= 2; ++i) {
//            double term_i = 5 * (i + 2);
//            double x2_16i = x2 - 16 * i;
//            double x2_16i_2 = x2_16i * x2_16i;
//            double x2_16i_4 = x2_16i_2 * x2_16i_2;
//            double x2_16i_6 = x2_16i_4 * x2_16i_2;
//
//            for (int j = -2; j <= 2; ++j) {
//                double x1_16j = x1 - 16 * j;
//                double x1_16j_2 = x1_16j * x1_16j;
//                double x1_16j_4 = x1_16j_2 * x1_16j_2;
//                double x1_16j_6 = x1_16j_4 * x1_16j_2;
//
//                result_value += 1.0 / (term_i + j + 3 + x1_16j_6 + x2_16i_6);
//            }
//        }
//        result_value = 1.0 / result_value;
//
//    } else if (func_id == 2) {
//        // function2
//        const double a = 20.0;
//        const double b = 0.2;
//        const double c = 2.0 * M_PI;
//
//        double sum_squares = x1*x1 + x2*x2;
//        double sqrt_term = sqrt(0.5 * sum_squares);  // OpenCL sqrt() function
//        double exp_term1 = exp(-b * sqrt_term);     // OpenCL exp() function
//        double term1 = -a * exp_term1;
//
//        double cos_sum = cos(c*x1) + cos(c*x2);     // OpenCL cos() function
//        double exp_term2 = exp(0.5 * cos_sum);      // OpenCL exp() function
//        double term2 = -exp_term2;
//
//        result_value = term1 + term2 + a + exp(1.0); // OpenCL exp() function
//
//    } else if (func_id == 3) {
//        // function3
//        const int m = 5;
//        const double a1[] = {1, 2, 1, 1, 5};
//        const double a2[] = {4, 5, 1, 2, 4};
//        const double c[] = {2, 1, 4, 7, 2};
//
//        result_value = 0.0;
//        for (int i = 0; i < m; ++i) {
//            double dist2 = (x1 - a1[i]) * (x1 - a1[i]) + (x2 - a2[i]) * (x2 - a2[i]);
//            double exp_term = exp(-dist2 / M_PI);  // OpenCL exp() function
//            double cos_term = cos(M_PI * dist2);  // OpenCL cos() function
//            result_value += c[i] * exp_term * cos_term;
//        }
//
//        result_value = -result_value;
//    }
//
//    // Store the result in the global memory
//    result[idx * num_steps + idy] = result_value;
//}
//
//)";
//
//    // Step 3: Compile the program and create kernel
//    cl_program program = create_program(context, device, kernel_source);
//    cl_kernel kernel = create_kernel(program, "integrate_kernel");
//
//    // Step 4: Prepare data buffers
//    std::vector<double> x1_values(num_steps);
//    std::vector<double> x2_values(num_steps);
//    std::vector<double> result(num_steps * num_steps);
//
//    // Fill x1_values and x2_values with the respective values
//    for (int i = 0; i < num_steps; ++i) {
//        x1_values[i] = x1_min + i * (x1_max - x1_min) / num_steps;
//        x2_values[i] = x2_min + i * (x2_max - x2_min) / num_steps;
//    }
//
//    cl_mem x1_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * num_steps, x1_values.data(), &err);
//    check_cl_error(err, "Unable to create buffer for x1 values.");
//    cl_mem x2_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * num_steps, x2_values.data(), &err);
//    check_cl_error(err, "Unable to create buffer for x2 values.");
//    cl_mem result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * num_steps * num_steps, nullptr, &err);
//    check_cl_error(err, "Unable to create result buffer.");
//
//    // Step 5: Set kernel arguments
//    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &x1_buffer);
//    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &x2_buffer);
//    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &result_buffer);
//    err |= clSetKernelArg(kernel, 3, sizeof(int), &num_steps);
//    err |= clSetKernelArg(kernel, 4, sizeof(double), &x1_min);
//    err |= clSetKernelArg(kernel, 5, sizeof(double), &x1_max);
//    err |= clSetKernelArg(kernel, 6, sizeof(double), &x2_min);
//    err |= clSetKernelArg(kernel, 7, sizeof(double), &x2_max);
//    err |= clSetKernelArg(kernel, 8, sizeof(int), &func_id);
//    check_cl_error(err, "Unable to set kernel arguments.");
//
//    // Step 6: Execute the kernel
//    size_t global_work_size[] = { num_steps, num_steps };
//    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
//    check_cl_error(err, "Unable to enqueue kernel.");
//
//    // Step 7: Read back the result
//    err = clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0, sizeof(double) * num_steps * num_steps, result.data(), 0, nullptr, nullptr);
//    check_cl_error(err, "Unable to read result buffer.");
//
//    // Step 8: Compute the final integral (sum of results)
//    double integral = 0.0;
//    for (int i = 0; i < num_steps; ++i) {
//        for (int j = 0; j < num_steps; ++j) {
//            integral += result[i * num_steps + j];
//        }
//    }
//
//    // Clean up OpenCL resources
//    clReleaseMemObject(x1_buffer);
//    clReleaseMemObject(x2_buffer);
//    clReleaseMemObject(result_buffer);
//    clReleaseKernel(kernel);
//    clReleaseProgram(program);
//    clReleaseCommandQueue(queue);
//    clReleaseContext(context);
//
//    return integral;
//}
//
//void check_cl_error(cl_int err, const char* message) {
//    if (err != CL_SUCCESS) {
//        std::cerr << message << " Error code: " << err << std::endl;
//        exit(1);
//    }
//}
//
//cl_context create_context() {
//    cl_int err;
//    cl_platform_id platform;
//    err = clGetPlatformIDs(1, &platform, nullptr);
//    check_cl_error(err, "Unable to get platform ID.");
//
//    cl_device_id device;
//    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
//    check_cl_error(err, "Unable to get device ID.");
//
//    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
//    check_cl_error(err, "Unable to create context.");
//    return context;
//}
//
//cl_command_queue create_command_queue(cl_context context, cl_device_id device) {
//    cl_int err;
//    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
//    check_cl_error(err, "Unable to create command queue.");
//    return queue;
//}
//
//cl_program create_program(cl_context context, cl_device_id device, const char* source) {
//    cl_int err;
//    cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &err);
//    check_cl_error(err, "Unable to create program.");
//
//    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
//    // In integrate_with_opencl() after clBuildProgram() is called
//
// //   cl_int err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
//    if (err != CL_SUCCESS) {
//        std::cerr << "Unable to build program. Error code: " << err << std::endl;
//        check_cl_build_errors(program, device); // This will print the actual error
//        exit(1);
//    }
//
//    check_cl_error(err, "Unable to build program.");
//
//    return program;
//}
//
//cl_kernel create_kernel(cl_program program, const char* kernel_name) {
//    cl_int err;
//    cl_kernel kernel = clCreateKernel(program, kernel_name, &err);
//    check_cl_error(err, "Unable to create kernel.");
//    return kernel;
//}
//
////int main() {
////    double result = integrate_with_opencl(nullptr, 0.0, 10.0, 0.0, 10.0, 100);
////    std::cout << "Parallel integration result: " << result << std::endl;
////    return 0;
////}
//
//double function1(double x1, double x2) {
//    double result = 0.002;
//
//    for (int i = -2; i <= 2; ++i) {
//        double term_i = 5 * (i + 2);
//        double x2_16i = x2 - 16 * i;
//        double x2_16i_2 = x2_16i * x2_16i;
//        double x2_16i_4 = x2_16i_2 * x2_16i_2;
//        double x2_16i_6 = x2_16i_4 * x2_16i_2;
//
//        for (int j = -2; j <= 2; ++j) {
//            double x1_16j = x1 - 16 * j;
//            double x1_16j_2 = x1_16j * x1_16j;
//            double x1_16j_4 = x1_16j_2 * x1_16j_2;
//            double x1_16j_6 = x1_16j_4 * x1_16j_2;
//
//            result += 1.0 / (term_i + j + 3 + x1_16j_6 + x2_16i_6);
//        }
//    }
//    return 1.0 / result;
//}
//
//double function2(double x1, double x2) {
//    const double a = 20.0;
//    const double b = 0.2;
//    const double c = 2.0 * M_PI;
//
//    double sum_squares = x1*x1 + x2*x2;
//    double sqrt_term = std::sqrt(0.5 * sum_squares);
//    double exp_term1 = std::exp(-b * sqrt_term);
//    double term1 = -a * exp_term1;
//
//    double cos_sum = std::cos(c*x1) + std::cos(c*x2);
//    double exp_term2 = std::exp(0.5 * cos_sum);
//    double term2 = -exp_term2;
//
//    return term1 + term2 + a + std::exp(1.0);
//}
//
//double function3(double x1, double x2) {
//    const int m = 5;
//    const double a1[] = {1, 2, 1, 1, 5};
//    const double a2[] = {4, 5, 1, 2, 4};
//    const double c[] = {2, 1, 4, 7, 2};
//
//    double result = 0.0;
//    for (int i = 0; i < m; ++i) {
//        double dist2 = std::pow(x1 - a1[i], 2) + std::pow(x2 - a2[i], 2);
//        double exp_term = std::exp(-dist2 / M_PI);
//        double cos_term = std::cos(M_PI * dist2);
//        result += c[i] * exp_term * cos_term;
//    }
//
//    return -result;
//}
//
//// Main function for performing integration
//int main() {
//    // Integration parameters
//    const double x1_min = -5.0;
//    const double x1_max = 5.0;
//    const double x2_min = -5.0;
//    const double x2_max = 5.0;
//    const int num_steps = 100;  // Number of steps in each dimension (you can adjust this)
//
//    // Integrating function 1 (func_id = 1)
//    std::cout << "Integrating function 1..." << std::endl;
//    double integral1 = integrate_with_opencl(function1, x1_min, x1_max, x2_min, x2_max, num_steps, 1);
//    std::cout << "Integral of function 1: " << integral1 << std::endl;
//
//    // Integrating function 2 (func_id = 2)
//    std::cout << "Integrating function 2..." << std::endl;
//    double integral2 = integrate_with_opencl(function2, x1_min, x1_max, x2_min, x2_max, num_steps, 2);
//    std::cout << "Integral of function 2: " << integral2 << std::endl;
//
//    // Integrating function 3 (func_id = 3)
//    std::cout << "Integrating function 3..." << std::endl;
//    double integral3 = integrate_with_opencl(function3, x1_min, x1_max, x2_min, x2_max, num_steps, 3);
//    std::cout << "Integral of function 3: " << integral3 << std::endl;
//
//    return 0;
//}

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <chrono>
#include <cmath>
#include <CL/cl.h>
#include <cstring>

// OpenCL helper functions
void check_cl_error(cl_int err, const std::string& message) {
    if (err != CL_SUCCESS) {
        throw std::runtime_error(message + " Error code: " + std::to_string(err));
    }
}

cl_context create_context(cl_device_id device) {
    cl_int err;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    check_cl_error(err, "Failed to create context.");
    return context;
}

cl_command_queue create_command_queue(cl_context context, cl_device_id device) {
    cl_int err;
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    check_cl_error(err, "Failed to create command queue.");
    return queue;
}

cl_program create_program(cl_context context, cl_device_id device, const char* source) {
    cl_int err;
    size_t source_len = strlen(source);
    cl_program program = clCreateProgramWithSource(context, 1, &source, &source_len, &err);
    check_cl_error(err, "Failed to create program.");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Get the build log
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::string log_str(log.data(), log_size);
        throw std::runtime_error("Failed to build program: " + log_str);
    }

    return program;
}

cl_kernel create_kernel(cl_program program, const char* kernel_name) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, kernel_name, &err);
    check_cl_error(err, "Failed to create kernel.");
    return kernel;
}

// Configuration reader
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

    // Validate that all required parameters were found
    if (!found_abs_err) throw std::runtime_error("Missing parameter 'abs_err'");
    if (!found_rel_err) throw std::runtime_error("Missing parameter 'rel_err'");
    if (!found_x_start) throw std::runtime_error("Missing parameter 'x_start'");
    if (!found_x_end) throw std::runtime_error("Missing parameter 'x_end'");
    if (!found_y_start) throw std::runtime_error("Missing parameter 'y_start'");
    if (!found_y_end) throw std::runtime_error("Missing parameter 'y_end'");
    if (!found_init_steps_x) throw std::runtime_error("Missing parameter 'init_steps_x'");
    if (!found_init_steps_y) throw std::runtime_error("Missing parameter 'init_steps_y'");
    if (!found_max_iter) throw std::runtime_error("Missing parameter 'max_iter'");
}

// OpenCL integration function using rectangle method
double integrate_opencl(int func_id, double x1_min, double x1_max, double x2_min, double x2_max,
                        int num_steps_x, int num_steps_y) {
    cl_int err;

    // Step 1: Set up OpenCL environment
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, nullptr);
    check_cl_error(err, "Unable to get platform ID.");

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        // Try CPU if GPU not available
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
        check_cl_error(err, "Unable to get device ID.");
    }

    cl_context context = create_context(device);
    cl_command_queue queue = create_command_queue(context, device);

    // Step 2: Write OpenCL kernel code for rectangle method
    const char* kernel_source = R"(
__kernel void integrate_kernel(__global double* result_buffer,
                              const int num_steps_x, const int num_steps_y,
                              const double x1_min, const double x1_max,
                              const double x2_min, const double x2_max,
                              const int func_id) {
    int gid_x = get_global_id(0);
    int gid_y = get_global_id(1);

    if (gid_x >= num_steps_x || gid_y >= num_steps_y) return;

    // Calculate cell dimensions
    double dx = (x1_max - x1_min) / num_steps_x;
    double dy = (x2_max - x2_min) / num_steps_y;

    // Calculate midpoint of the rectangle
    double x1 = x1_min + (gid_x + 0.5) * dx;
    double x2 = x2_min + (gid_y + 0.5) * dy;

    // Calculate function value at the midpoint
    double f_value = 0.0;

    // Function selection
    if (func_id == 1) {
        // Function 1
        f_value = 0.002;
        for (int i = -2; i <= 2; ++i) {
            double term_i = 5 * (i + 2);
            double x2_16i = x2 - 16 * i;
            double x2_16i_2 = x2_16i * x2_16i;
            double x2_16i_4 = x2_16i_2 * x2_16i_2;
            double x2_16i_6 = x2_16i_4 * x2_16i_2;

            for (int j = -2; j <= 2; ++j) {
                double x1_16j = x1 - 16 * j;
                double x1_16j_2 = x1_16j * x1_16j;
                double x1_16j_4 = x1_16j_2 * x1_16j_2;
                double x1_16j_6 = x1_16j_4 * x1_16j_2;

                f_value += 1.0 / (term_i + j + 3 + x1_16j_6 + x2_16i_6);
            }
        }
        f_value = 1.0 / f_value;
    }
    else if (func_id == 2) {
        // Function 2
        const double a = 20.0;
        const double b = 0.2;
        const double c = 2.0 * M_PI;

        double sum_squares = x1*x1 + x2*x2;
        double sqrt_term = sqrt(0.5 * sum_squares);
        double exp_term1 = exp(-b * sqrt_term);
        double term1 = -a * exp_term1;

        double cos_sum = cos(c*x1) + cos(c*x2);
        double exp_term2 = exp(0.5 * cos_sum);
        double term2 = -exp_term2;

        f_value = term1 + term2 + a + exp(1.0);
    }
    else if (func_id == 3) {
        // Function 3
        const int m = 5;
        const double a1[5] = {1, 2, 1, 1, 5};
        const double a2[5] = {4, 5, 1, 2, 4};
        const double c[5] = {2, 1, 4, 7, 2};

        f_value = 0.0;
        for (int i = 0; i < m; ++i) {
            double dist2 = (x1 - a1[i]) * (x1 - a1[i]) + (x2 - a2[i]) * (x2 - a2[i]);
            double exp_term = exp(-dist2 / M_PI);
            double cos_term = cos(M_PI * dist2);
            f_value += c[i] * exp_term * cos_term;
        }

        f_value = -f_value;
    }

    // Store the result (function value × cell area)
    result_buffer[gid_x * num_steps_y + gid_y] = f_value * dx * dy;
}
)";

    // Step 3: Compile the program and create kernel
    cl_program program = create_program(context, device, kernel_source);
    cl_kernel kernel = create_kernel(program, "integrate_kernel");

    // Step 4: Create result buffer
    std::vector<double> result(num_steps_x * num_steps_y);
    cl_mem result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * num_steps_x * num_steps_y, nullptr, &err);
    check_cl_error(err, "Unable to create result buffer.");

    // Step 5: Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &result_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(int), &num_steps_x);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &num_steps_y);
    err |= clSetKernelArg(kernel, 3, sizeof(double), &x1_min);
    err |= clSetKernelArg(kernel, 4, sizeof(double), &x1_max);
    err |= clSetKernelArg(kernel, 5, sizeof(double), &x2_min);
    err |= clSetKernelArg(kernel, 6, sizeof(double), &x2_max);
    err |= clSetKernelArg(kernel, 7, sizeof(int), &func_id);
    check_cl_error(err, "Unable to set kernel arguments.");

    // Step 6: Execute the kernel
    size_t global_work_size[2] = { static_cast<size_t>(num_steps_x), static_cast<size_t>(num_steps_y) };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
    check_cl_error(err, "Unable to enqueue kernel.");

    // Step 7: Read back the result
    err = clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0, sizeof(double) * num_steps_x * num_steps_y,
                             result.data(), 0, nullptr, nullptr);
    check_cl_error(err, "Unable to read result buffer.");

    // Step 8: Sum all the results to get the final integral
    double integral = 0.0;
    for (int i = 0; i < num_steps_x * num_steps_y; ++i) {
        integral += result[i];
    }

    // Clean up OpenCL resources
    clReleaseMemObject(result_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return integral;
}

double integrate_opencl_tiled(int func_id, double x1_min, double x1_max, double x2_min, double x2_max,
                           int num_steps_x, int num_steps_y) {
    // Get device memory limits
    cl_platform_id platform;
    cl_device_id device;
    cl_int err = clGetPlatformIDs(1, &platform, nullptr);
    err |= clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
    }

    cl_ulong max_alloc_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc_size), &max_alloc_size, NULL);

    // Maximum number of elements per tile
    size_t max_elements = max_alloc_size / sizeof(double) / 2;  // Use only half of max memory

    // Calculate tile dimensions
    int tile_size_x = 1000;  // Starting point
    int tile_size_y = 1000;

    while (static_cast<size_t>(tile_size_x) * tile_size_y > max_elements) {
        tile_size_x /= 2;
        tile_size_y /= 2;
    }

    // Calculate number of tiles
    int num_tiles_x = (num_steps_x + tile_size_x - 1) / tile_size_x;
    int num_tiles_y = (num_steps_y + tile_size_y - 1) / tile_size_y;

    double total_integral = 0.0;

    // Process each tile
    for (int tile_x = 0; tile_x < num_tiles_x; ++tile_x) {
        for (int tile_y = 0; tile_y < num_tiles_y; ++tile_y) {
            // Calculate this tile's boundaries
            int start_x = tile_x * tile_size_x;
            int end_x = std::min(start_x + tile_size_x, num_steps_x);
            int start_y = tile_y * tile_size_y;
            int end_y = std::min(start_y + tile_size_y, num_steps_y);

            int this_tile_steps_x = end_x - start_x;
            int this_tile_steps_y = end_y - start_y;

            // Calculate physical boundaries for this tile
            double this_x_min = x1_min + (x1_max - x1_min) * start_x / num_steps_x;
            double this_x_max = x1_min + (x1_max - x1_min) * end_x / num_steps_x;
            double this_y_min = x2_min + (x2_max - x2_min) * start_y / num_steps_y;
            double this_y_max = x2_min + (x2_max - x2_min) * end_y / num_steps_y;

            // Integrate this tile
            double tile_integral = integrate_opencl(func_id, this_x_min, this_x_max,
                                                 this_y_min, this_y_max,
                                                 this_tile_steps_x, this_tile_steps_y);

            total_integral += tile_integral;
        }
    }

    return total_integral;
}
// Adaptive integration with OpenCL
double adaptiveIntegrate_opencl(int func_id, double x1_min, double x1_max, double x2_min, double x2_max,
                              double abs_err, double rel_err, int init_steps_x, int init_steps_y,
                              int max_iter, double& achieved_abs_err, double& achieved_rel_err) {

    int steps_x = init_steps_x;
    int steps_y = init_steps_y;

    // First integration
    double prev_result = integrate_opencl(func_id, x1_min, x1_max, x2_min, x2_max, steps_x, steps_y);

    for (int iter = 0; iter < max_iter; ++iter) {
        // Double the number of steps for each dimension
        steps_x *= 2;
        steps_y *= 2;

        double current_result = integrate_opencl_tiled(func_id, x1_min, x1_max, x2_min, x2_max, steps_x, steps_y);

        // Calculate errors
        achieved_abs_err = std::fabs(current_result - prev_result);
        achieved_rel_err = achieved_abs_err / (std::fabs(current_result) + 1e-10);

        // Check if desired accuracy is achieved
        if (achieved_abs_err < abs_err && achieved_rel_err < rel_err) {
            return current_result;
        }

        prev_result = current_result;
    }

    // If we get here, we couldn't achieve the desired accuracy
    throw std::runtime_error("Could not achieve desired accuracy");
    return prev_result;
}

// Main function that processes command line arguments and runs the integration
int main(int argc, char* argv[]) {
    using namespace std::chrono;

    if (argc != 4) {
        std::cerr << "Wrong number of arguments" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <function_number> <config_file> <num_threads>" << std::endl;
        return 1;
    }

    int func_id;
    try {
        func_id = std::stoi(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "Wrong function index" << std::endl;
        return 2;
    }

    if (func_id < 1 || func_id > 3) {
        std::cerr << "Wrong function index" << std::endl;
        return 2;
    }

    unsigned int num_threads;
    try {
        num_threads = std::stoi(argv[3]);
        if (num_threads <= 0) {
            std::cerr << "Number of threads must be positive" << std::endl;
            return 4;
        }
    } catch (const std::exception& e) {
        std::cerr << "Invalid number of threads" << std::endl;
        return 4;
    }

    unsigned int max_threads = std::thread::hardware_concurrency();
    if (max_threads > 0 && num_threads > max_threads) {
        num_threads = max_threads;
    }

    double abs_err, rel_err;
    double x_start, x_end, y_start, y_end;
    int init_steps_x, init_steps_y, max_iter;

    try {
        readConfig(argv[2], abs_err, rel_err, x_start, x_end, y_start, y_end, init_steps_x, init_steps_y, max_iter);
    } catch (const std::exception& e) {
        if (std::string(e.what()).find("Error opening configuration file") != std::string::npos) {
            std::cerr << e.what() << std::endl;
            return 3;
        } else {
            std::cerr << "Error reading configuration file: " << e.what() << std::endl;
            return 5;
        }
    }

    if (rel_err >= 0.001) {
        std::cerr << "Relative error must be less than 0.001" << std::endl;
        return 5;
    }

    double result, achieved_abs_err, achieved_rel_err;
    auto start_time = high_resolution_clock::now();

    try {
        // Note: num_threads is passed to OpenCL but not directly used since parallelism is handled by the GPU/device
        result = adaptiveIntegrate_opencl(func_id, x_start, x_end, y_start, y_end, abs_err, rel_err,
                                       init_steps_x, init_steps_y, max_iter, achieved_abs_err, achieved_rel_err);
    } catch (const std::exception& e) {
        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end_time - start_time).count();
        std::cout << result << std::endl;
        std::cout << achieved_abs_err << std::endl;
        std::cout << achieved_rel_err << std::endl;
        std::cout << duration << std::endl;
        std::cerr << e.what() << std::endl;
        return 16;
    }

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time).count();

    // Output the results
    std::cout << result << std::endl;
    std::cout << achieved_abs_err << std::endl;
    std::cout << achieved_rel_err << std::endl;
    std::cout << duration << std::endl;

    return 0;
}