#include "UniversalHittingSet_Hashtable.hpp"

#include <fstream>

#include <mpi.h>



uhs_hashtable_slot* new_uhs_hashtable(int hashtable_capacity, int myrank) {
	uhs_hashtable_slot* hashtable;
	size_t hashtable_size = (hashtable_capacity * sizeof(uhs_hashtable_slot));
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



__device__ uhs_key_t mmer_numeric_at_gpu(int start_index, int m, const char* uhs_string) {
	uhs_key_t mmer_numeric = 0;
	for (int offset = 0; offset < m; offset += 1) {
		char s = uhs_string[start_index + offset];
		size_t x = ((s & 4) >> 1);
		mmer_numeric |= ((x + ((x ^ (s & 2)) >> 1)) << ((2 * (m - 1)) - (2 * offset)));
	}
	return mmer_numeric;
}


__global__ void populate_uhs_hashtable_keys_gpu(const char* uhs_string, int MINIMIZER_LENGTH, uhs_hashtable_slot* uhs_hashtable, uint64_t uhs_hashtable_capacity, uhs_key_t* uhs_mmers, uint64_t uhs_mmers_count) {

	unsigned int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (thread_id >= uhs_mmers_count) {
		return;
	}

	int mmer_start_index = (thread_id * MINIMIZER_LENGTH);
	uhs_key_t mmer_numeric = mmer_numeric_at_gpu(mmer_start_index, MINIMIZER_LENGTH, uhs_string);

	uhs_mmers[thread_id] = mmer_numeric;

	uint32_t slot_index = (cuda_murmur3_64(mmer_numeric) & (uhs_hashtable_capacity - 1));

	while (true) {
		uhs_key_t old_key = atomicCAS(&(uhs_hashtable[slot_index].key), 0, mmer_numeric);
		if (old_key == 0 || old_key == mmer_numeric) {
			return;
		}
		slot_index = ((slot_index + 1) & (uhs_hashtable_capacity - 1));
	}

}


uhs_hashtable_slot* initialize_uhs_frequencies_hashtable(char* uhs_file_path, int MINIMIZER_LENGTH, int myrank, uint64_t* output_uhs_hashtable_capacity, uhs_key_t** output_uhs_mmers, uint64_t* output_uhs_mmers_count) {

	std::string uhs_string = uhs_string_from_file(uhs_file_path);

	char* uhs_string_gpu;
	size_t uhs_string_size = ((uhs_string.length() + 1) * sizeof(char));
	cudaMalloc(((void**) &uhs_string_gpu), uhs_string_size);
	cudaMemcpy(uhs_string_gpu, ((char*) uhs_string.data()), uhs_string_size, cudaMemcpyHostToDevice);

	int uhs_mmers_count = (uhs_string.length() / MINIMIZER_LENGTH);

	uhs_key_t* uhs_mmers = NULL;
	cudaMalloc(((void**) &uhs_mmers), (uhs_mmers_count * sizeof(uhs_key_t)));

	uint64_t uhs_hashtable_capacity = 1;
	while (uhs_hashtable_capacity < ((uint64_t) uhs_mmers_count)) {
		uhs_hashtable_capacity *= 2;
	}

	uhs_hashtable_slot* uhs_frequencies_hashtable = new_uhs_hashtable(uhs_hashtable_capacity, myrank);

	int min_grid_size;
	int thread_block_size;
	cudaOccupancyMaxPotentialBlockSize(
		&min_grid_size,
		&thread_block_size,
		populate_uhs_hashtable_keys_gpu,
		0,
		0
	);
	int grid_size = ((uhs_mmers_count + (thread_block_size - 1)) / thread_block_size);

	populate_uhs_hashtable_keys_gpu<<<grid_size, thread_block_size>>>(
		uhs_string_gpu,
		MINIMIZER_LENGTH,
		uhs_frequencies_hashtable,
		uhs_hashtable_capacity,
		uhs_mmers,
		uhs_mmers_count
	);
	cudaDeviceSynchronize();

	cudaFree(uhs_string_gpu);

	*output_uhs_hashtable_capacity = uhs_hashtable_capacity;
	*output_uhs_mmers = uhs_mmers;
	*output_uhs_mmers_count = uhs_mmers_count;
	return uhs_frequencies_hashtable;

}



__device__ uhs_value_t get_mmer_frequency_gpu(uhs_key_t mmer_numeric, uhs_hashtable_slot* uhs_frequencies_hashtable, uint64_t uhs_hashtable_capacity) {
	uint32_t slot_index = (cuda_murmur3_64(mmer_numeric) & (uhs_hashtable_capacity - 1));
	while (true) {
		uhs_key_t slot_key = uhs_frequencies_hashtable[slot_index].key;
		if (slot_key == mmer_numeric) {
			uhs_value_t frequency = uhs_frequencies_hashtable[slot_index].value;
			return frequency;
		}
		else if (slot_key == 0) {
			return 1000000000;
		}
		slot_index = ((slot_index + 1) & (uhs_hashtable_capacity - 1));
	}
	return 1000000000;
}


__device__ void set_mmer_frequency_gpu(uhs_key_t mmer_numeric, uhs_value_t new_frequency, uhs_hashtable_slot* uhs_frequencies_hashtable, uint64_t uhs_hashtable_capacity) {
	uint32_t slot_index = (cuda_murmur3_64(mmer_numeric) & (uhs_hashtable_capacity - 1));
	while (true) {
		keyType slot_key = uhs_frequencies_hashtable[slot_index].key;
		if (slot_key == mmer_numeric) {
			uhs_frequencies_hashtable[slot_index].value = new_frequency;
			return;
		}
		else if (slot_key == 0) {
			return;
		}
		slot_index = ((slot_index + 1) & (uhs_hashtable_capacity - 1));
	}
}



__device__ void increment_mmer_frequency_gpu(uhs_key_t mmer_numeric, uhs_hashtable_slot* uhs_frequencies_hashtable, uint64_t uhs_hashtable_capacity) {
	uint32_t slot_index = (cuda_murmur3_64(mmer_numeric) & (uhs_hashtable_capacity - 1));
	while (true) {
		uhs_key_t slot_key = uhs_frequencies_hashtable[slot_index].key;
		if (slot_key == mmer_numeric) {
			atomicAdd(&(uhs_frequencies_hashtable[slot_index].value), 1);
			return;
		}
		else if (slot_key == 0) {
			return;
		}
		slot_index = ((slot_index + 1) & (uhs_hashtable_capacity - 1));
	}
}


__global__ void set_uhs_frequencies_from_sample_gpu(char* sequence, unsigned int sequence_length, double sample_percentage, int MINIMIZER_LENGTH, uhs_hashtable_slot* uhs_frequencies_hashtable, uint64_t uhs_hashtable_capacity) {
	unsigned int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int sample_gap = std::ceil(1.0 / sample_percentage);
	int mmer_start_index = (thread_id * sample_gap);
	if ((mmer_start_index + MINIMIZER_LENGTH) > sequence_length) {
		return;
	}
	uhs_key_t mmer_numeric = mmer_numeric_at_gpu(mmer_start_index, MINIMIZER_LENGTH, ((const char*) sequence));
	increment_mmer_frequency_gpu(mmer_numeric, uhs_frequencies_hashtable, uhs_hashtable_capacity);
}


void set_uhs_frequencies_from_sample(char* sequence, unsigned int sequence_length, double sample_fraction, int MINIMIZER_LENGTH, uhs_hashtable_slot* uhs_frequencies_hashtable, uint64_t uhs_hashtable_capacity) {

	int min_grid_size;
	int thread_block_size;
	cudaOccupancyMaxPotentialBlockSize(
		&min_grid_size,
		&thread_block_size,
		set_uhs_frequencies_from_sample_gpu,
		0,
		0
	);
	int grid_size = ((std::ceil(sequence_length * sample_fraction) + (thread_block_size - 1)) / thread_block_size);

	set_uhs_frequencies_from_sample_gpu<<<grid_size, thread_block_size>>>(
		sequence,
		sequence_length,
		sample_fraction,
		MINIMIZER_LENGTH,
		uhs_frequencies_hashtable,
		uhs_hashtable_capacity
	);
	cudaDeviceSynchronize();

}



__global__ void reset_uhs_frequencies_gpu(uhs_hashtable_slot* uhs_frequencies_hashtable, uint64_t uhs_hashtable_capacity, uhs_key_t* uhs_mmers, uint64_t uhs_mmers_count) {

	unsigned int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (thread_id >= uhs_mmers_count) {
		return;
	}

	uhs_key_t mmer_numeric = uhs_mmers[thread_id];
	set_mmer_frequency_gpu(mmer_numeric, 0, uhs_frequencies_hashtable, uhs_hashtable_capacity);

}


void reset_uhs_frequencies(uhs_hashtable_slot* uhs_frequencies_hashtable, uint64_t uhs_hashtable_capacity, uhs_key_t* uhs_mmers, uint64_t uhs_mmers_count) {

	int min_grid_size;
	int thread_block_size;
	cudaOccupancyMaxPotentialBlockSize(
		&min_grid_size,
		&thread_block_size,
		reset_uhs_frequencies_gpu,
		0,
		0
	);
	int grid_size = ((uhs_mmers_count + (thread_block_size - 1)) / thread_block_size);

	reset_uhs_frequencies_gpu<<<grid_size, thread_block_size>>>(
		uhs_frequencies_hashtable,
		uhs_hashtable_capacity,
		uhs_mmers,
		uhs_mmers_count
	);

}



__global__ void populate_uhs_frequencies_array_gpu(uhs_hashtable_slot* uhs_frequencies_hashtable, uint64_t uhs_hashtable_capacity, uhs_key_t* uhs_mmers, uint64_t uhs_mmers_count, uhs_value_t* mmer_frequencies) {

	unsigned int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (thread_id >= uhs_mmers_count) {
		return;
	}

	uhs_key_t mmer_numeric = uhs_mmers[thread_id];
	mmer_frequencies[thread_id] = get_mmer_frequency_gpu(mmer_numeric, uhs_frequencies_hashtable, uhs_hashtable_capacity);

}

__global__ void repopulate_uhs_hashtable_values_from_array_gpu(uhs_hashtable_slot* uhs_frequencies_hashtable, uint64_t uhs_hashtable_capacity, uhs_key_t* uhs_mmers, uint64_t uhs_mmers_count, uhs_value_t* mmer_frequencies) {

	unsigned int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (thread_id >= uhs_mmers_count) {
		return;
	}

	uhs_key_t mmer_numeric = uhs_mmers[thread_id];
	set_mmer_frequency_gpu(mmer_numeric, mmer_frequencies[thread_id], uhs_frequencies_hashtable, uhs_hashtable_capacity);

}


void allreduce_uhs_frequencies(uhs_hashtable_slot* uhs_frequencies_hashtable, uint64_t uhs_hashtable_capacity, uhs_key_t* uhs_mmers, uint64_t uhs_mmers_count, long mpi_mode) {

    uhs_value_t mmer_frequencies_local[uhs_mmers_count];
	uhs_value_t mmer_frequencies_total[uhs_mmers_count];
	uhs_value_t* mmer_frequencies_gpu;
	size_t mmer_frequencies_size = (uhs_mmers_count * sizeof(uhs_value_t));
	cudaMalloc(&mmer_frequencies_gpu, mmer_frequencies_size);

	int min_grid_size;
	int thread_block_size;
	cudaOccupancyMaxPotentialBlockSize(
		&min_grid_size,
		&thread_block_size,
		populate_uhs_frequencies_array_gpu,
		0,
		0
	);
	int grid_size = ((uhs_mmers_count + (thread_block_size - 1)) / thread_block_size);

	populate_uhs_frequencies_array_gpu<<<grid_size, thread_block_size>>>(
		uhs_frequencies_hashtable,
		uhs_hashtable_capacity,
		uhs_mmers,
		uhs_mmers_count,
		mmer_frequencies_gpu
	);

	cudaMemcpy(mmer_frequencies_local, mmer_frequencies_gpu, mmer_frequencies_size, cudaMemcpyDeviceToHost);

	MPI_Allreduce(
		mmer_frequencies_local,
		mmer_frequencies_total,
		uhs_mmers_count,
		MPI_UINT32_T,
		MPI_SUM,
		MPI_COMM_WORLD
	);

	cudaMemcpy(mmer_frequencies_gpu, mmer_frequencies_total, mmer_frequencies_size, cudaMemcpyHostToDevice);

	repopulate_uhs_hashtable_values_from_array_gpu<<<grid_size, thread_block_size>>>(
		uhs_frequencies_hashtable,
		uhs_hashtable_capacity,
		uhs_mmers,
		uhs_mmers_count,
		mmer_frequencies_gpu
	);

	cudaFree(mmer_frequencies_gpu);
}
