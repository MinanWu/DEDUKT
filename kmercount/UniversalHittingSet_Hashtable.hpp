#ifndef UHS_HASHTABLE_HPP
#define UHS_HASHTABLE_HPP

#include "common_gpu.h"



KeyValue* initialize_uhs_frequencies_hashtable(char* uhs_file_path, int MINIMIZER_LENGTH, int myrank, uint64_t* output_uhs_hashtable_capacity);

__device__ uint32_t get_mmer_frequency_gpu(keyType mmer_long, KeyValue* uhs_frequencies_hashtable, uint64_t uhs_hashtable_capacity);

void set_uhs_frequencies_from_sample(char* sequence, double sample_fraction, int MINIMIZER_LENGTH, KeyValue* uhs_frequencies_hashtable, uint64_t uhs_hashtable_capacity);


#endif