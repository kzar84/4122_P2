#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

#define NUM_THREADS 256


/*********************
 * GLOBAL VARIABLES *
*********************/
float k;
float starting_temp;
int dimension;
int timesteps;
int width, height, depth;
int num_fixed_points;
std::vector<float> fixed_points;


/*******************
 * HOST FUNCTIONS *
/******************/
void read_file(std::string file_name);                                                                                // Reads configuration file
float* alloc_host_mem(int width, int height, int depth, float temp);                                                  // Allocates space for a "simulated" array on the host  
float* alloc_device_mem(int width, int height, int depth);                                                            // Allocates space for a "simulated" array on the device
void print_grid(float* grid, int width, int height, int depth);                                                       // Prints the grid to a csv file
void set_fixed_temps_2D(std::vector<float> fixed_points, int num_fixed_points, float* grid, int width);               // Sets the fixed temp blocks in the grid (2D)
void set_fixed_temps_3D(std::vector<float> fixed_points, int num_fixed_points, float* grid, int width, int height);   // Sets the fixed temp blocks in the grid (3D)

/*********************
 * DEVICE FUNCTIONS *
/********************/
__global__ void update_grid_2D(float* d_old_grid, float* d_new_grid, int width, int height, float k);                 // Updates 2D grid (one timestep)
__global__ void update_grid_3D(float* d_old_grid, float* d_new_grid, int width, int height, int depth, float k);      // Updates 3D grid (one timestep)


/*********************
  ****** MAIN *******
/********************/
int main(int argc, char** argv) {
    // Read in the file to get the configurations
    read_file(argv[1]);

    // Allocate space for grids
    float* grid = alloc_host_mem(width, height, depth, starting_temp);      // HOST MEMORY (set blocks to default temp)
    float* d_new_grid = alloc_device_mem(width, height, depth);             // DEVICE MEMORY
    float* d_old_grid = alloc_device_mem(width, height, depth);             // DEVICE MEMORY
    // Size var for size of array
    int size = width*height*depth*sizeof(float);
    // Determine number of blocks to launch
    int NUM_BLOCKS = (int)((width*height*depth)/NUM_THREADS) + 1;

    // 2D timestep loop
    if (dimension == 2) {
        // Set the fixed temps (first time)
        set_fixed_temps_2D(fixed_points, num_fixed_points, grid, width);
        /*** Timestep Loop ***/
        for (int i = 0; i < timesteps; ++i) {
            // Copy host grid to device
            cudaMemcpy(d_old_grid, grid, size, cudaMemcpyHostToDevice);
            // Call device function (updates grid for this timestep)
            update_grid_2D<<<NUM_BLOCKS, NUM_THREADS>>>(d_old_grid, d_new_grid, width, height, k);
            // Copy device grid to host
            cudaMemcpy(grid, d_new_grid, size, cudaMemcpyDeviceToHost);
            // Set the fixed temps (reset every timestep)
            set_fixed_temps_2D(fixed_points, num_fixed_points, grid, width);
        }
    }
    // 3D timestep loop
    if (dimension == 3) {
        // Set the fixed temps (reset every timestep)
        set_fixed_temps_3D(fixed_points, num_fixed_points, grid, width, height);
        /*** Timestep Loop ***/
        for (int i = 0; i < timesteps; ++i) {
            // Copy host grid to device
            cudaMemcpy(d_old_grid, grid, size, cudaMemcpyHostToDevice);
            // Call device function (updates grid for this timestep)
            update_grid_3D<<<NUM_BLOCKS, NUM_THREADS>>>(d_old_grid, d_new_grid, width, height, depth, k);
            // Copy device grid to host
            cudaMemcpy(grid, d_new_grid, size, cudaMemcpyDeviceToHost);
            // Set the fixed temps (reset every timestep)
            set_fixed_temps_3D(fixed_points, num_fixed_points, grid, width, height);
        }
    }

    // Print out the results (to the output file)
    print_grid(grid, width, height, depth);

    // Free up that memory
    free(grid);
    cudaFree(d_old_grid);
    cudaFree(d_new_grid);

    return 0;
}


/*******************
 * HOST FUNCTIONS *
/******************/
// Read in the configuration file and populate global variables
void read_file(std::string file_name) {
    // Use std namespace for function calls (string, etc...)
    using namespace std;
    // Open a file
    ifstream fh(file_name);
    // find the first line ignore comments/empty lines
    string line;
    getline(fh, line);
    line.erase(remove(line.begin(), line.end(), ' '), line.end());
    while (line[0] == '#' || line.empty()) { getline(fh, line); line.erase(remove(line.begin(), line.end(), ' '), line.end()); }
    // get rid of possible whitespaces in the line
    line.erase(remove(line.begin(), line.end(), ' '), line.end());
    // Set the dimension (2 or 3)
    if (line.compare("2D") == 0)
        dimension = 2;
    else
        dimension = 3;

    //Loop through 5 more times to get the remaing arguments
    for (int i = 0; i < 5; ++i) {
        // Find the next line that is not a comment or empty
        getline(fh, line);
        // get rid of possible whitespaces in the line
        line.erase(remove(line.begin(), line.end(), ' '), line.end());
        while (line[0] == '#' || line.empty()) { getline(fh, line); line.erase(remove(line.begin(), line.end(), ' '), line.end());}
        // Use these for tokenizing line 3 and 5
        stringstream data(line);
        vector<string> tokens;
        string tok; 
        switch (i)
        {
            case 0:
                k = stof(line);
                break;
            case 1:
                timesteps = stoi(line);
                break;
            case 2:
                while(getline(data, tok, ',')) {
                    tokens.push_back(tok);
                }
                // Set width and height to the tokens
                width = stoi(tokens[0]);
                height = stoi(tokens[1]);
                if (dimension == 3) { depth = stoi(tokens[2]); }
                else { depth = 1; }
                break;
            case 3:
                starting_temp = stof(line);
                break;
            case 4:
                num_fixed_points++;
                while(getline(data, tok, ',')) {
                    fixed_points.push_back(stof(tok));
                }
                while(getline(fh, line)) {
                    if (line[0] != '#' && !line.empty()) {
                        stringstream tmp(line);
                        num_fixed_points++;
                        while (getline(tmp, tok, ',')) {
                            fixed_points.push_back(stof(tok));
                        }
                    }
                }
                break;
        }
    }

    fh.close();
}

// Allocates space for a "simulated" array on the host
float* alloc_host_mem(int width, int height, int depth, float temp) {
    float* grid = (float*)calloc(width*height*depth, sizeof(float));
    for (int i = 0; i < width*height*depth; ++i)
        grid[i] = temp;
    
    return grid;
}

// Allocates space for a "simulated" array on the device
float* alloc_device_mem(int width, int height, int depth) {
    float* d_grid; 
    cudaMalloc((void**)&d_grid, width*height*depth*sizeof(float));

    return d_grid;
}

// Print the grid to the output file
void print_grid(float* grid, int width, int height, int depth) {
    // Open a file
    //int size = width*height;
    std::ofstream fh("heatOutput.csv");
    for (int d = 0; d < depth; ++d) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                fh << grid[d*width*height + i*width + j];
                // Print a comma if youre not the last element
                if (j != width - 1) 
                    fh << ", ";
            }
            //Move to a new line each row
            fh << "\n";
        }
        // Put a blank line in between each slice
        fh << "\n\n";
    }
    fh.close();
}

// Take a grid and vector of fixed points and fill in the fixed temps (2D)
void set_fixed_temps_2D(std::vector<float> fixed_points, int num_fixed_points, float* grid, int width) {
    // Fill in fixed temp spots
    for (int i = 0; i < num_fixed_points; ++i) {
        // Extract the current fixed point
        int f_x = fixed_points[i*5 + 0];
        int f_y = fixed_points[i*5 + 1];
        int f_width = fixed_points[i*5 + 2];            
        int f_height = fixed_points[i*5 + 3];
        int f_temp = fixed_points[i*5 + 4];

        for (int j = 0; j < f_height; ++j) {
            for (int k = 0; k < f_width; ++k) {
                grid[(f_y+j)*width + f_x + k] = f_temp;
            }
        }
    }
}

// Take a grid and vector of fixed points and fill in the fixed temps (3D)
void set_fixed_temps_3D(std::vector<float> fixed_points, int num_fixed_points, float* grid, int width, int height) {
    // Fill in fixed temp spots
    for (int i = 0; i < num_fixed_points; ++i) {
        // Extract the current fixed point
        int f_x = fixed_points[i*7 + 0];
        int f_y = fixed_points[i*7 + 1];
        int f_z = fixed_points[i*7 + 2];
        int f_width = fixed_points[i*7 + 3];            
        int f_height = fixed_points[i*7 + 4];
        int f_depth = fixed_points[i*7 + 5];
        int f_temp = fixed_points[i*7 + 6];

        for (int d = 0; d < f_depth; ++d) {
            for (int j = 0; j < f_height; ++j) {
                for (int k = 0; k < f_width; ++k) {
                    grid[(f_z+d)*width*height + (f_y+j)*width + (f_x + k)] = f_temp;       // (f_z+d) + (f_y+j)*depth + (f_x + k)*height*depth
                }
            }
        }
    }
}

/*********************
 * DEVICE FUNCTIONS *
/********************/
// Temp diffusion function that updates the 2D grid each timestep
__global__ void update_grid_2D(float* d_old_grid, float* d_new_grid, int width, int height, float k) {
    // Get the current index in the device array
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // Make sure the index is within the array (blockDim is not always a perfect multiple)
    if (index < width * height) {
        // Extract the (x,y) cooridnates to check for edge cases
        int x, y;
        x = index % width;
        y = (int)(index/width);

        // Values for calculating next temp
        float T_old, T_top, T_bottom, T_left, T_right;
        T_old = d_old_grid[index];

        // Get T_left
        if (x != 0)
            T_left = d_old_grid[index - 1];
        else
            T_left = T_old;
        // Get T_top
        if (y != 0)
            T_top = d_old_grid[index - width] ;
        else
            T_top = T_old;
        // Get T_right
        if (x != width - 1)
            T_right = d_old_grid[index + 1];
        else
            T_right = T_old;
        // Get T_bottom
        if (y != height - 1)
            T_bottom = d_old_grid[index + width];
        else
            T_bottom = T_old;

        // Update d_new_grid with new value
        d_new_grid[index] = T_old + k*(T_top + T_bottom + T_left + T_right - 4*T_old);
    }
}

// Temp diffusion function that updates the 3D grid each timestep
__global__ void update_grid_3D(float* d_old_grid, float* d_new_grid, int width, int height, int depth, float k) {
    // Get the current index in the device array
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // Make sure the index is within the array (blockDim is not always a perfect multiple)
    if (index < width*height*depth) {
        // Extract the (x,y,z) cooridnates to check for edge cases
        int x, y, z;
        z = (int)(index/(width*height));
        y = (int)((index-width*height*z)/width);
        x = (index-width*height*z) % width;

        // Values for calculating next temp
        float T_old, T_top, T_bottom, T_left, T_right, T_front, T_back;
        T_old = d_old_grid[index];

        // Get T_left
        if (x != 0)
            T_left = d_old_grid[index - 1];
        else
            T_left = T_old;
        // Get T_top
        if (y != 0)
            T_top = d_old_grid[index - width] ;
        else
            T_top = T_old;
        // Get T_right
        if (x != width - 1)
            T_right = d_old_grid[index + 1];
        else
            T_right = T_old;
        // Get T_bottom
        if (y != height - 1)
            T_bottom = d_old_grid[index + width];
        else
            T_bottom = T_old;
        // Get T_front
        if (z != 0)
            T_front = d_old_grid[index - width*height];
        else 
            T_front = T_old;
        // Get T_back
        if (z != depth - 1)
            T_back = d_old_grid[index + width*height];
        else
            T_back = T_old;

        // Update d_new_grid with new value
        d_new_grid[index] = T_old + k*(T_top + T_bottom + T_left + T_right + T_front + T_back - 6*T_old);
    }
}