#ifndef UHS_HASHTABLE_HPP
#define UHS_HASHTABLE_HPP

#include "common_gpu.h"



typedef uint32_t uhs_key_t;
typedef uint32_t uhs_value_t;

typedef struct uhs_hashtable_slot {
	uhs_key_t key;
	uhs_value_t value;
} uhs_hashtable_slot;

uhs_key_t uhs_key_empty = 0;
uhs_value_t uhs_value_empty = 1000000000;



uhs_hashtable_slot* initialize_uhs_frequencies_hashtable(
	char* uhs_file_path,
	int MINIMIZER_LENGTH,
	int myrank,
	uint64_t* output_uhs_hashtable_capacity,
	uhs_key_t* output_uhs_mmers,
	uint64_t* output_uhs_mmers_count
);

__device__ uhs_value_t get_mmer_frequency_gpu(
	uhs_key_t mmer_numeric,
	uhs_hashtable_slot* uhs_frequencies_hashtable,
	uint64_t uhs_hashtable_capacity
);

void reset_uhs_hashtable_frequencies(
	uhs_hashtable_slot* uhs_frequencies_hashtable,
	uint64_t uhs_hashtable_capacity,
	uhs_key_t* uhs_mmers,
	uint64_t uhs_mmers_count
);

void set_uhs_frequencies_from_sample(
	char* sequence,
	unsigned int sequence_length,
	double sample_fraction,
	int MINIMIZER_LENGTH,
	uhs_hashtable_slot* uhs_frequencies_hashtable,
	uint64_t uhs_hashtable_capacity
);



#endif