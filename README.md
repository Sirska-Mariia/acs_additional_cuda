
# Lab work 3: Parallel-integral-CUDA
Authors (team): Mariia Sirska - https://github.com/Sirska-Mariia
Viktoria Koval - https://github.com/Vika-Koval 
## Prerequisites

cmake, gcc

### Compilation

mkdir <file_name> 

cd <file_name> 

cmake .. 

make 

./integrate_parallel <number_of_function> ../func<number_of_function>.cfg  number_of_threads


### Results

In this task, we paralelly calculated the integral of functions (such as Ackley's, Langermann, De Jong's) on CUDA using the usual method of cells (a generalization of the method of rectangles) and tested the results.
![image](https://github.com/user-attachments/assets/90b27794-5580-4364-9845-875ed7cc5e4d)

