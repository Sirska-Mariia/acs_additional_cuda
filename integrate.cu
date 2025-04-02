//#include <iostream>
//#include <cstring>
//#include <chrono>
//#include <stdexcept>
//#include <thread>
//#include "functions.h"

//using namespace std;
//using namespace std::chrono;
//unsigned int num_threads = 0;
//
//int main(int argc, char* argv[]) {
//    if (argc != 4) {
//        cerr << "Wrong number of arguments" << endl;
//        cerr << "Usage: " << argv[0] << " <function_number> <config_file> <num_threads>" << endl;
//        return 1;
//    }
//    int func_num;
//    try {
//        func_num = stoi(argv[1]);
//    } catch (const exception& e) {
//        cerr << "Wrong function index" << endl;
//        return 2;
//    }
//
//    if (func_num < 1 || func_num > 3) {
//        cerr << "Wrong function index" << endl;
//        return 2;
//    }
//    double (*func)(double, double);
//    switch (func_num) {
//        case 1: func = function1; break;
//        case 2: func = function2; break;
//        case 3: func = function3; break;
//        default:
//            cerr << "Wrong function index" << endl;
//            return 2;
//    }
//    try {
//        num_threads = stoi(argv[3]);
//        if (num_threads <= 0) {
//            cerr << "Number of threads must be positive" << endl;
//            return 4;
//        }
//    } catch (const exception& e) {
//        cerr << "Invalid number of threads" << endl;
//        return 4;
//    }
//    unsigned int max_threads = thread::hardware_concurrency();
//    if (max_threads > 0 && num_threads > max_threads) {
//        num_threads = max_threads;
//    }
//    double abs_err, rel_err;
//    double x_start, x_end, y_start, y_end;
//    int init_steps_x, init_steps_y, max_iter;
//
//    try {
//        readConfig(argv[2], abs_err, rel_err, x_start, x_end, y_start, y_end, init_steps_x, init_steps_y, max_iter);
//    } catch (const exception& e) {
//        if (strcmp(e.what(), "Error opening configuration file") == 0) {
//            cerr << e.what() << endl;
//            return 3;
//        } else {
//            cerr << "Error reading configuration file: " << e.what() << endl;
//            return 5;
//        }
//    }
//
//    if (rel_err >= 0.001) {
//        cerr << "Relative error must be less than 0.001" << endl;
//        return 5;
//    }
//
//    double result, achieved_abs_err, achieved_rel_err;
//    auto start_time = high_resolution_clock::now();
//
//    try {
//        result = adaptiveIntegrate(func, x_start, x_end, y_start, y_end, abs_err, rel_err,
//                                  init_steps_x, init_steps_y, max_iter, achieved_abs_err, achieved_rel_err);
//    } catch (const exception& e) {
//        auto end_time = high_resolution_clock::now();
//        auto duration = duration_cast<milliseconds>(end_time - start_time).count();
//        cout << result << endl;
//        cout << achieved_abs_err << endl;
//        cout << achieved_rel_err << endl;
//        cout << duration << endl;
//        cerr << e.what() << endl;
//        return 16;
//    }
//
//    auto end_time = high_resolution_clock::now();
//    auto duration = duration_cast<milliseconds>(end_time - start_time).count();
//    cout << result << endl;
//    cout << achieved_abs_err << endl;
//    cout << achieved_rel_err << endl;
//    cout << duration << endl;
//
//    return 0;
//}
//#include <iostream>
//#include <cstring>
//#include <chrono>
//#include <stdexcept>
//#include <thread>
//#include "cuda.cu"
#include <iostream>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include "cuda_functions.h"


using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Wrong number of arguments" << endl;
        cerr << "Usage: " << argv[0] << " <function_number> <config_file> <num_threads>" << endl;
        return 1;
    }

    int func_num;
    try {
        func_num = stoi(argv[1]);
    } catch (const exception& e) {
        cerr << "Wrong function index" << endl;
        return 2;
    }

    if (func_num < 1 || func_num > 3) {
        cerr << "Wrong function index" << endl;
        return 2;
    }

    double abs_err, rel_err;
    double x_start, x_end, y_start, y_end;
    int init_steps_x, init_steps_y, max_iter;

    try {
        readConfig(argv[2], abs_err, rel_err, x_start, x_end, y_start, y_end, init_steps_x, init_steps_y, max_iter);
    } catch (const exception& e) {
        cerr << "Error reading configuration file: " << e.what() << endl;
        return 5;
    }

    if (rel_err >= 0.001) {
        cerr << "Relative error must be less than 0.001" << endl;
        return 5;
    }

    double result = 0.0;
    double achieved_abs_err, achieved_rel_err;
    auto start_time = high_resolution_clock::now();

    try {
        result = adaptiveIntegrateCUDA(func_num, x_start, x_end, y_start, y_end,
                                       abs_err, rel_err, init_steps_x, init_steps_y, max_iter,
                                       achieved_abs_err, achieved_rel_err);
    } catch (const exception& e) {
        cerr << "CUDA integration failed: " << e.what() << endl;
        return 6;
    }

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time).count();

    cout << result << endl;
  //  cout << duration << endl;
    cout << achieved_abs_err << endl;
    cout << achieved_rel_err << endl;
    cout << duration << endl;

    return 0;
}
