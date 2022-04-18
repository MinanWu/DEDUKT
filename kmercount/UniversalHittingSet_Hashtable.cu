#include "UniversalHittingSet_Hashtable.hpp"

#include <fstream>



KeyValue* new_uhs_hashtable(int hashtable_capacity, int myrank) {

    int gpu_count;
    cudaGetDeviceCount(&gpu_count);
    cudaSetDevice(myrank % gpu_count);

    KeyValue* hashtable;
    size_t hashtable_size = (hashtable_capacity * sizeof(KeyValue));
    cudaMalloc(&hashtable, hashtable_size);
    cudaMemset(&hashtable, 0, hashtable_size);

    return hashtable;

}



std::string uhs_string_from_file(char* uhs_file_path) {

    std::string uhs_string;

    std::ifstream uhs_file(uhs_file_path);
	if (!(uhs_file.is_open())) {
		printf("Error: Failed to open UHS file '%s'\n", uhs_file_path);
        return uhs_string;
	}

	// std::string uhs_string(std::istreambuf_iterator<char>{uhs_file}, {});
	std::string uhs_file_line;
	while (!(uhs_file.eof())) {
		getline(uhs_file, uhs_file_line);
		uhs_string.append(uhs_file_line);
	}
	uhs_file.close();

	return uhs_string;

}



__device__ keyType mmer_long_at_gpu(int start_index, int m, const char* uhs_string) {
	keyType mmer_long = 0;
	for (int offset = 0; offset < m; offset += 1) {
		char s = uhs_string[start_index + offset];
		int j = (offset % 32);
		size_t x = ((s & 4) >> 1);
    	mmer_long |= ((x + ((x ^ (s & 2)) >> 1)) << (2 * (31 - j)));
	}
	return mmer_long;
}


__global__ void populate_uhs_hashtable_keys_gpu(const char* uhs_string, int MINIMIZER_LENGTH, int mmers_count, KeyValue* uhs_hashtable, uint64_t uhs_hashtable_capacity) {

	unsigned int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thread_id >= mmers_count) {
        return;
    }

	int mmer_start_index = (thread_id * MINIMIZER_LENGTH);
	keyType mmer_long = mmer_long_at_gpu(mmer_start_index, MINIMIZER_LENGTH, uhs_string);

    keyType slot_index = (cuda_murmur3_64(mmer_long) % uhs_hashtable_capacity);
    while (true) {
        keyType old_key = atomicCAS(&(uhs_hashtable[slot_index].key), kEmpty, mmer_long);
        if (old_key == kEmpty || old_key == mmer_long) {
            return;
        }
        slot_index = ((slot_index + 1) % uhs_hashtable_capacity);
    }

}


KeyValue* initialize_uhs_frequencies_hashtable(char* uhs_file_path, int MINIMIZER_LENGTH, int myrank, uint64_t* output_uhs_hashtable_capacity) {

    std::string uhs_string = uhs_string_from_file(uhs_file_path);

    char* uhs_string_gpu;
	size_t uhs_string_size = ((uhs_string.length() + 1) * sizeof(char));
	cudaMalloc(((void**) &uhs_string_gpu), uhs_string_size);
	cudaMemcpy(uhs_string_gpu, ((char*) uhs_string.data()), uhs_string_size, cudaMemcpyHostToDevice);

	int mmers_count = (uhs_string.length() / MINIMIZER_LENGTH);

    uint64_t uhs_hashtable_capacity = ((uint64_t) (mmers_count * 5));

    KeyValue* uhs_frequencies_hashtable = new_uhs_hashtable(uhs_hashtable_capacity, myrank);

	int min_grid_size;
	int thread_block_size;
	cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size,
        &thread_block_size,
        populate_uhs_hashtable_keys_gpu,
        0,
        0
    );
	int grid_size = ((mmers_count + (thread_block_size - 1)) / thread_block_size);

	populate_uhs_hashtable_keys_gpu<<<grid_size, thread_block_size>>>(
        uhs_string_gpu,
        MINIMIZER_LENGTH,
        mmers_count,
        uhs_frequencies_hashtable,
        uhs_hashtable_capacity
    );
	cudaDeviceSynchronize();

    cudaFree(uhs_string_gpu);

    *output_uhs_hashtable_capacity = uhs_hashtable_capacity;
    return uhs_frequencies_hashtable;

}



__device__ uint32_t get_mmer_frequency_gpu(keyType mmer_long, KeyValue* uhs_frequencies_hashtable, uint64_t uhs_hashtable_capacity) {
    keyType slot_index = (cuda_murmur3_64(mmer_long) % uhs_hashtable_capacity);
    keyType stop_index = slot_index - 1;
    while (slot_index != stop_index) {
        keyType slot_key = uhs_frequencies_hashtable[slot_index].key;
        if (slot_key == mmer_long) {
            uint32_t frequency = uhs_frequencies_hashtable[slot_index].value;
            return frequency;
        }
        slot_index = ((slot_index + 1) % uhs_hashtable_capacity);
    }
    return 0;
}



__device__ void increment_mmer_frequency_gpu(keyType mmer_long, KeyValue* uhs_frequencies_hashtable, uint64_t uhs_hashtable_capacity) {
    // ...
    // (A helper function for Task 2 — given an m-mer represented numerically, look up that m-mer in the hash table, like in the function get_mmer_frequency_gpu, and then increment its frequency, or do nothing if it doesn't exist in the hashtable)
}


__global__ void set_uhs_frequencies_from_sample_gpu(char* sequence, double sample_percentage, int MINIMIZER_LENGTH, KeyValue* uhs_frequencies_hashtable, uint64_t uhs_hashtable_capacity) {
    // ... (The main function for Task 2)
}


void set_uhs_frequencies_from_sample(char* sequence, double sample_fraction, int MINIMIZER_LENGTH, KeyValue* uhs_frequencies_hashtable, uint64_t uhs_hashtable_capacity) {

    // ... (The function where Task 2 begins)

    // char* sequence:  A bunch of DNA reads, with the boundary between two reads denoted by a lowercase 'a'
    // double sample_fraction:  If sample_fraction == 0.01, for example, that means we want to sample 1% of the m-mers, so look at the m-mers beginning at every 100th character
    // int MINIMIZER_LENGTH:  The length of the m-mer to sample at each of the points we're looking at (so, the "m" in "m-mer")
    // KeyValue* uhs_frequencies_hashtable:  The hashtable of m-mer frequencies; at this point, entries for all of the m-mers in the hitting set have been created, and their frequencies are all initialized to 0
    // uint64_t uhs_hashtable_capacity:  The number of slots in the hashtable, which is usually more than the number of m-mers in the hitting set; this may change if we end up needing to experiment with different load factors to optimize performance

    // For examples of how to 

    int min_grid_size;
	int thread_block_size;
	cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size,
        &thread_block_size,
        set_uhs_frequencies_from_sample_gpu,
        0,
        0
    );
	int grid_size = 1;  // Definitely need to change this, we probably want as many threads as there are m-mers being sampled? e.g. Maybe don't create a thread for *every* position, just create enough threads for the positions being sampled?

	set_uhs_frequencies_from_sample_gpu<<<grid_size, thread_block_size>>>(
        sequence,
        sample_fraction,
        MINIMIZER_LENGTH,
        uhs_frequencies_hashtable,
        uhs_hashtable_capacity
    );
	cudaDeviceSynchronize();

}