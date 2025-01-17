/* HEADER
 * TODO: add licensing
 *
 * Originally authored and developed for short read analysis by: Aydin Buluc, Evangelos Georganas, and Rob Egan.
 * Extended for long read analysis by Marquita Ellis.
 * GPU version by Israt Nisa.
 */

#include <cassert>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <libgen.h>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>
#include <sstream>
#include <limits>
#include <array>
#include <tuple>
#include <cstdint>
#include <functional>
#include <tuple>
#include <omp.h>
//#include "nvbio/basic/omp.h"
#include <mpi.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "algorithm"
#include "random"
#include "stdint.h"

#include "../version.h"
#include "../common/common.h"
#include "../common/file.h"
#include "../common/mpi_common.h"
#include "../common/stdlib_compatibility.hpp"
#include "../common/VectorMap.hpp"
#include "common.h"

int MINIMIZER_LENGTH=7;
#ifdef SINGLE_EXEC
extern StaticVars _sv;
#else
StaticVars _sv = NULL;
#endif

#include "../common/MPIUtils.h"
#include "DataStructures/hyperloglog.hpp"
extern "C" {
#ifdef HIPMER_BLOOM64
#include "DataStructures/libbloom/bloom64.h"
#else
#include "DataStructures/libbloom/bloom.h"
#endif
}

#ifdef HEAVYHITTERS

#ifndef MAXHITTERS
#define MAXHITTERS 32000
#endif

#endif

//#define DEBUG 1
#include "Kmer.hpp"
// #include "simple.cuh"
#include "kmrCnt_GPU.h"
#include "spmer_kmrCnt.h"
#include "KmerIterator.hpp"
#include "Deleter.h"
#include "ParallelFASTQ.h"
#include "Friends.h"
#include "MPIType.h"
#include "SimpleCount.h"
#include "FriendsMPI.h"
#include "Pack.h"
#include "common.h"
#include "../common/Buffer.h"
#include "../common/hash_funcs.h"
#ifdef KHASH
#include "khash.hh"
#endif
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include "../common/optlist.h"

// #include "../readoverlap/ReadOverlapper.h"

#include "kmermatch.h"
#include "kmerhashtable.h"

#ifdef HEAVYHITTERS
typedef SimpleCount<Kmer, KmerHash> HeavyHitters;
HeavyHitters *heavyhitters = NULL;
UFX2ReduceObj * Frequents;
size_t nfreqs;
#endif

using namespace std;

//#define BENCHMARKONLY // does not output a ufx file when defined
#define MEGA 1000000.0
#define MILLION 1000000
#define COUNT_THRESHOLD 300000
#define COUNT_THRESHOLD_HIGH 30300000
#define HIGH_BIN 100
#define HIGH_NUM_BINS ((COUNT_THRESHOLD_HIGH-COUNT_THRESHOLD)/HIGH_BIN)

int ERR_THRESHOLD;


typedef array<int,4> ACGT;
int nprocs;
int myrank;
int64_t nonerrorkmers;
int64_t kmersprocessed;
int64_t readsprocessed;
int totaltime;
int type;
int window;
int reliable_min = 2;
int reliable_max = MAX_NUM_READS;

#ifndef UPC_ALLOCATOR
#define UPC_ALLOCATOR 0
#endif

double tot_pack_GPU = 0.0, tot_exch_GPU = 0.0, tot_process_GPU = 0.0;


#include "UniversalHittingSet_Hashtable.hpp"

int minimizer_ordering = 0;
char* uhs_file_path;
float uhs_sample_fraction = 0.01;
long uhs_mpi_mode = 0;


READIDS newReadIdList() {
	READIDS toreturn = *(new READIDS);
	ASSERT(nullReadId == 0, "Read ID lists are being initialized to 0, despite the reserved null read ID being non-zero..."); // TODO
	std::memset(&toreturn, 0, MAX_NUM_READS*sizeof(ReadId));
	return toreturn;
}

POSITIONS newPositionsList() {
	POSITIONS toreturn = *(new POSITIONS);
	ASSERT(initPos == 0, "Position lists are being initialized to 0, despite the reserved null position being non-zero..."); // TODO
	std::memset(&toreturn, 0, MAX_NUM_READS*sizeof(PosInRead));
	return toreturn;
}

KmerCountsType *kmercounts = NULL;

void countTotalKmersAndCleanHash();

void writeMaxReadLengths(vector<int> &maxReadLengths, const vector<filedata> &allfiles ) {
	ASSERT( maxReadLengths.size() == allfiles.size(),"" );
	int *globalMaxReadLength = new int[allfiles.size()];
	CHECK_MPI( MPI_Reduce( &maxReadLengths[0], globalMaxReadLength, allfiles.size(), MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD) );
	if (!myrank) {
		for(int i = 0; i < allfiles.size(); i++) {
			char file[MAX_FILE_PATH];
			char *name = get_basename(file, allfiles[i].filename);
			strcat(name, ".maxReadLen.txt");
			FILE *f = fopen_rank_path(name, "w", -1);
			fprintf(f, "%d", globalMaxReadLength[i]);
			if (fclose_track(f) != 0) { DIE("Could now write %s! %s\n", name, strerror(errno)); }
		}
	}
	delete [] globalMaxReadLength;
}

inline void StoreReadName(const ReadId & readIndex, std::string name, std::unordered_map<ReadId, std::string>& readNameMap) {
	ASSERT( readNameMap.count(readIndex) == 0, "Rank "+ to_string(MYTHREAD) +": collision in readNameMap on key=" + to_string(readIndex) + ", count is " + to_string(readNameMap.count(readIndex)) );
	readNameMap[readIndex] = name;
}

KeyValue* d_hashTable;

// use:  km.toString(s);
// pre:  s has space for k+1 elements
// post: s[0,...,k-1] is the DNA string for the Kmer km and s[k] = '\0'
string toString_custm(int64_t longs) {

	char *s = new char[32];
	size_t i,j,l;

	for (i = 0; i < KMER_LENGTH; i++) {
		j = i % 32;
		l = i / 32;

		switch(((longs) >> (2*(31-j)) )& 0x03 ) {
			case 0x00: *s = 'A'; ++s; break;
			case 0x01: *s = 'C'; ++s; break;
			case 0x02: *s = 'G'; ++s; break;
			case 0x03: *s = 'T'; ++s; break;
		}
		// for (int ii = 0; ii < KMER_LENGTH; ii++)
		// 	cout << s[ii];
		// cout << " ";
	}

	*s = '\0';
	cout << longs << " - " << std::string(s) <<  endl;
	return std::string(s);
}
static int exchange_iter = 0;

int batch = -1;


void getSupermers_CPU_DEBUG(char* seq, int klen, int mlen, int nproc, int *owner_counter,
		keyType* h_send_smers, unsigned char* h_send_slens, int n_kmers, int rank ){

	unsigned int seq_len = strlen(seq);

	bool validKmer = false;

	int window = 32 - klen;// - mlen + 1 ;
	int grid = (seq_len + (128*window - 1) ) / (128*window);// * window;
	// int per_block_seq_len = 128;
	int order = 0;
	cout << "info " << grid << " " << window << " " << n_kmers << endl;

	uint64_t counter = 0, match = 0;

	for(int g = 0; g < grid; ++g) {

		int st_char_grid = g * 128 * window ;
		if((st_char_grid+klen - 1) >= n_kmers) break;

		for(int b = 0; b < 128; ++b) {
			validKmer = false;
			int st_char_block = st_char_grid + b * window;
			if((st_char_block + klen - 1) >= n_kmers) break;

			uint64_t comprs_Kmer = 0; keyType comprs_Smer = 0;
			keyType cur_mini = std::numeric_limits<uint64_t>::max();
			keyType prev_mini = cur_mini;
			int i = st_char_block; bool inserted = false;

			int w = 0;
			cout << "new window " << endl;
			for(; w < window; w++) {
				if((st_char_block + w +klen - 1) >= n_kmers) return;
				char s; validKmer = false;
				uint64_t comprs_Kmer = 0;
				for (int k = 0; k < KMER_LENGTH; ++k) {

					s = seq[ i + w + k];
					if(s == 'a' || s == 'N') {
						w+=KMER_LENGTH-1;
						validKmer = false; break;
					}
					else validKmer = true;
					int j = k % 32;
					size_t x = ((s) & 4) >> 1;
					comprs_Kmer |= ((x + ((x ^ (s & 2)) >>1)) << (2*(31-j))); //make it longs[] to support larger kmer
				}
				if(validKmer){
					if(w == 0) {
						cur_mini = find_minimizer(comprs_Kmer, order);
						comprs_Smer = comprs_Kmer; //slen = klen;
					}
					else  {
						cur_mini = find_minimizer(comprs_Kmer, order);

						if(prev_mini == cur_mini ){
							match++;
							// validKmer = true;
							if((i+w) < 50) cout << "GPU smer: matched: " << s << " " << cur_mini << " " << comprs_Kmer << " " << counter <<" " << w << endl;

							inserted = false;
						}
						else {
							counter++;
							if((i+w) < 50) cout << "GPU smer: new: " << s << " " << cur_mini << " " << comprs_Kmer << " " << counter <<" " << w << endl;

							// validKmer = false;
							inserted = true;
						}
					}
					prev_mini = cur_mini;
				}
				//         else{
				//         	cur_mini = std::numeric_limits<uint64_t>::max();
				// prev_mini = cur_mini;
				//         }
				//             if((i + w + klen) >= n_kmers) {
				//             	cout << "Similar CPU supermer: Total supermers: " <<  counter <<  " " << w <<endl;// if(i < 5)// toString_custm(kmers_compressed[i]);
				// 	return ;
				// }
				if(validKmer && w == window - 1  ) {
					counter++;
					if((i) < 50) cout << "outside Similar CPU supermer: Total supermers: " <<  counter <<  " " << w <<endl;


					// cout << i+window << " " <<  n_kmers << " " << counter << endl;
				}
			}
			cout << "Similar CPU supermer: Total supermers: " <<  counter <<  " " << match<<endl; // if(i < 5)// toString_custm(kmers_compressed[i]);



			// if(validKmer) counter++;
		}
	}

	cout << "Similar CPU supermer: Total supermers: " <<  counter <<  " " << match<<endl;// if(i < 5)// toString_custm(kmers_compressed[i]);

	return ;
}

keyType tmp_MurmurHash3_x64_128(const void* key, const uint32_t len, const uint32_t seed)
{
	const uint8_t * data = (const uint8_t*)key;
	const uint32_t nblocks = len / 16;
	int32_t i;

	uint64_t h1 = seed;
	uint64_t h2 = seed;

	uint64_t c1 = BIG_CONSTANT(0x87c37b91114253d5);
	uint64_t c2 = BIG_CONSTANT(0x4cf5ad432745937f);

	const uint8_t * tail = (const uint8_t*)(data + nblocks*16);

	uint64_t k1 = 0;
	uint64_t k2 = 0;

	switch(len & 15)
	{
		case 15: k2 ^= (uint64_t)(tail[14]) << 48;
		case 14: k2 ^= (uint64_t)(tail[13]) << 40;
		case 13: k2 ^= (uint64_t)(tail[12]) << 32;
		case 12: k2 ^= (uint64_t)(tail[11]) << 24;
		case 11: k2 ^= (uint64_t)(tail[10]) << 16;
		case 10: k2 ^= (uint64_t)(tail[ 9]) << 8;
		case  9: k2 ^= (uint64_t)(tail[ 8]) << 0;
			 k2 *= c2; k2  =  (k2 << 33) | (k2 >> (64 - 33)) ;//ROTL64(k2,33);
			 k2 *= c1; h2 ^= k2;

		case  8: k1 ^= (uint64_t)(tail[ 7]) << 56;
		case  7: k1 ^= (uint64_t)(tail[ 6]) << 48;
		case  6: k1 ^= (uint64_t)(tail[ 5]) << 40;
		case  5: k1 ^= (uint64_t)(tail[ 4]) << 32;
		case  4: k1 ^= (uint64_t)(tail[ 3]) << 24;
		case  3: k1 ^= (uint64_t)(tail[ 2]) << 16;
		case  2: k1 ^= (uint64_t)(tail[ 1]) << 8;
		case  1: k1 ^= (uint64_t)(tail[ 0]) << 0;
			 k1 *= c1; k1  = (k1 << 31) | (k1 >> (64 - 31)) ;//ROTL64(k1,31);
			 k1 *= c2; h1 ^= k1;
	};

	//----------
	// finalization

	h1 ^= len; h2 ^= len;

	h1 += h2;
	h2 += h1;

	// h1 = fmix64(h1);

	// /* regular murmur64 */
	keyType k  = h1;
	k ^= k >> 33;
	k *= BIG_CONSTANT(0xff51afd7ed558ccd);
	k ^= k >> 33;
	k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
	k ^= k >> 33;

	h1 = k;

	// h2 = fmix64(h2);

	k  = h2;
	k ^= k >> 33;
	k *= BIG_CONSTANT(0xff51afd7ed558ccd);
	k ^= k >> 33;
	k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
	k ^= k >> 33;

	h2 = k;

	h1 += h2;
	h2 += h1;

	// ((uint64_t*)out)[0] = h1;
	// ((uint64_t*)out)[1] = h2;

	return h1;
}

// std::unordered_map<uint64_t,uint64_t> kcounter_cpu;

// Kmer is of length k
// HyperLogLog counting, bloom filtering, and std::maps use Kmer as their key
int nkmers_thisBatch = 0;
int nkmers_processed = 0;



size_t ParseNPack(vector<string> & seqs, vector<string> names, vector<string> & quals, vector< vector<Kmer> > & outgoing, vector< vector<ReadId> > & readids,
		vector< vector<PosInRead> > & positions, ReadId & startReadIndex, vector<vector<array<char,2>>> & extquals, vector<vector<array<char,2>>> & extseqs,
		std::unordered_map<ReadId, std::string>& readNameMap, int pass, size_t offset)
{
	MPI_Pcontrol(1,"ParseNPack");
	size_t nreads = seqs.size();
	size_t nskipped = 0;
	size_t maxsending = 0, kmersthisbatch = 0;
	size_t bytesperkmer = Kmer::numBytes();
	size_t bytesperentry = bytesperkmer + 4;
	size_t memoryThreshold = (MAX_ALLTOALL_MEM / nprocs) * 2; // 2x any single rank
	DBG("ParseNPack(seqs %lld, qals %lld, out %lld, extq %lld, exts %lld, pass %d, offset %lld)\n", (lld) seqs.size(), (lld) quals.size(), (lld) outgoing.size(), (lld) extquals.size(), (lld) extseqs.size(), pass, (lld) offset);

	ReadId readIndex = startReadIndex;

	for(size_t i=offset; i< nreads; ++i)
	{
		size_t found  = seqs[i].length();
		// skip this sequence if the length is too short
		if (seqs[i].length() <= KMER_LENGTH) {
			//cerr << "seq is too short (" << seqs[i].length() << " < " << KMER_LENGTH << " : " << seqs[i] << endl;
			nskipped++;
			continue;
		}
		int nkmers = (seqs[i].length()-KMER_LENGTH+1);
		kmersprocessed += nkmers;
		kmersthisbatch += nkmers;

		std::vector<Kmer> kmers = Kmer::getKmers(seqs[i]); // calculate all the kmers

		ASSERT(kmers.size() == nkmers,"");
		size_t Nfound = seqs[i].find('N');

		for(size_t j=0; j< nkmers; ++j)
		{
			while (Nfound!=std::string::npos && Nfound < j) Nfound=seqs[i].find('N', Nfound+1);
			if (Nfound!=std::string::npos && Nfound < j+KMER_LENGTH) continue;	// if there is an 'N', toss it
			ASSERT(kmers[j] == Kmer(seqs[i].c_str() + j),"");

			size_t sending = PackEndsKmer(seqs[i], quals[i], j, kmers[j], readIndex, j, outgoing,
					readids, positions, extquals, extseqs, pass, found, KMER_LENGTH);
			if (sending > maxsending) maxsending = sending;
		}
		// if (pass == 2) { StoreReadName(readIndex, names[i], readNameMap); }
		readIndex++; // always start with next read index whether exiting or continuing the loop
		if (maxsending * bytesperentry >= (memoryThreshold) || (kmersthisbatch + seqs[i].length()) * bytesperentry >= (MAX_ALLTOALL_MEM)) {
			nreads = i+1; // start with next read
			// if (pass==2) { startReadIndex = readIndex; }
			break;
		}
	}

	// if (pass == 2) { startReadIndex = readIndex; }

	if(type > 0){ //not cpu based original dibella
		for (int i=0; i < nprocs; i++) {
			outgoing[i].clear(); // CPU parseNPack is populating this..remove when that call is removed
		}
	}

	// LOGF("ParseNPack got through %lld reads (of %lld) and skipped %lld reads, total %lld kmers\n", (lld) nreads, (lld) seqs.size(), (lld) nskipped, (lld) kmersthisbatch);
	MPI_Pcontrol(-1,"ParseNPack");
	return nreads;
}


size_t kmerCount_GPU(vector<string> & seqs, int pass, size_t offset, size_t endoffset, int nproc, struct bloom * bm)
{
	size_t nreads = endoffset;// seqs.size();
	size_t nskipped = 0, maxsending = 0, nkmerDelim_thisBatch = 0;
	size_t memoryThreshold = (MAX_ALLTOALL_MEM / nprocs) * 2; // 2x any single rank
	size_t nRecvdKmers = 0;
	std::string all_seqs;
	vector<keyType> kmers_GPU;
	vector<int> owner_counter (nprocs, 0);
	keyType *d_kmers;

	nkmers_thisBatch = 0;

	//*concat all reads before copying to GPU*
	for(size_t i=offset; i < nreads; ++i){

		if (seqs[i].length() <= KMER_LENGTH) continue;

		nkmers_thisBatch += seqs[i].length() - KMER_LENGTH + 1;
		int approx_maxsending = (4 * nkmers_thisBatch + nprocs - 1)/nprocs;
		all_seqs += seqs[i] + "a";

		// if (approx_maxsending * bytesperentry >= memoryThreshold || (nkmers_thisBatch + seqs[i].length()) * bytesperentry >= MAX_ALLTOALL_MEM) {
		// 	nreads = i+1; // start with next read
		// 	break;
		// }
	}

	if(all_seqs.length() == 0) return nreads;

	nkmerDelim_thisBatch = all_seqs.length() - KMER_LENGTH + 1;
	nkmers_processed += nkmers_thisBatch;

	//* Parse reads and pack to outgoing buffer on GPU *
	vector <uint64_t> h_outgoing(nkmerDelim_thisBatch * BUFF_SCALE);
	getKmers_GPU(all_seqs, h_outgoing, KMER_LENGTH, nprocs, owner_counter, myrank, BUFF_SCALE);
	// getKmers_test(all_seqs);

	//* Exchange kmers * TODO:: GPU direct communication *
	vector <keyType> recvbuf(nkmerDelim_thisBatch * BUFF_SCALE);
	int * recvcnt = new int[nprocs];

	double exch_t_GPU = GPU_Exchange(recvbuf, h_outgoing, pass, nprocs,
	nkmerDelim_thisBatch, owner_counter, nRecvdKmers, recvcnt); // outgoing arrays will be all empty, shouldn't crush
	tot_exch_GPU += exch_t_GPU;

	//* Build local kmer counter *
	int p_buff_len = ((nkmerDelim_thisBatch * BUFF_SCALE) + nprocs - 1)/nprocs;

	double process_t_GPU = GPU_buildCounter(d_hashTable, recvbuf, pass, bm, nRecvdKmers, recvcnt, p_buff_len);   // we might still receive data even if we didn't send any
	tot_process_GPU += process_t_GPU;
	delete[] recvcnt;
	if(nRecvdKmers > 0)  recvbuf.clear();
	return nreads;
}

/******* Performance reporting *******/
double perf_reporting(double exch_time, size_t totsend, size_t totrecv)
{
	double performance_report_time = MPI_Wtime();

	const int SND=0, RCV=1;
	int64_t local_counts[2];
	local_counts[SND] = totsend;
	local_counts[RCV] = totrecv;

	int64_t global_mins[2]={0,0};
	CHECK_MPI( MPI_Reduce(&local_counts, &global_mins, 2, MPI_LONG_LONG, MPI_MIN, 0, MPI_COMM_WORLD) );
	int64_t global_maxs[2]={0,0};
	CHECK_MPI( MPI_Reduce(&local_counts, &global_maxs, 2, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD) );
	int64_t global_sums[2] = {0,0};
	CHECK_MPI( MPI_Reduce(&local_counts, &global_sums, 2, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD) );

	double global_min_time = 0.0;
	CHECK_MPI( MPI_Reduce(&exch_time, &global_min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD) );
	double global_max_time = 0.0;
	CHECK_MPI( MPI_Reduce(&exch_time, &global_max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD) );
	double global_sum_time = 0.0;
	CHECK_MPI( MPI_Reduce(&exch_time, &global_sum_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) );

	int bytedata = 8;//sizeof(uint64_t);
	serial_printf("KmerMatch:%s sent min %lld bytes, sent max %lld bytes, sent avg %lld bytes, recv min %lld bytes, \
		recv max %lld bytes, recv avg %lld bytes, in min %.3f s, max %.3f s, avg %.3f s\n", __FUNCTION__, global_mins[SND]*bytedata, \
		global_maxs[SND]*bytedata, (global_sums[SND]/nprocs)*bytedata, global_mins[RCV]*bytedata, global_maxs[RCV]*bytedata, (global_sums[RCV]/nprocs)*bytedata, \
		global_min_time, global_max_time, global_sum_time/nprocs);
	performance_report_time = MPI_Wtime()-performance_report_time;

	return performance_report_time;
}

uint64_t nkmers_smer_all = 0;
uint64_t nSupermers_all = 0;
double tot_GPUsmer_build = 0, tot_GPUsmer_exch = 0, tot_GPU_smer_kcounter = 0;

size_t spmer_kmerCount_GPU(vector<string> & seqs, vector< vector<Kmer> > & outgoing, int pass, size_t offset, int endoffset, int minimizer_ordering, uhs_hashtable_slot* uhs_frequencies_hashtable, uint64_t uhs_hashtable_capacity, uhs_key_t* uhs_mmers, uint64_t uhs_mmers_count)
{
	double start_gpu_smer = MPI_Wtime();
	uint64_t HTsize_smer = 0, totalPairs_smer = 0, all_seq_size = 0;
	size_t nreads = endoffset;// seqs.size(), max_slen = 0;
	size_t maxsending = 0, nkmers_smer_batch = 0;
	size_t memoryThreshold = (MAX_ALLTOALL_MEM / nprocs) * 2; // 2x any single rank
	std::string all_seqs;

	int * recvcnt = new int[nprocs];
	int * sendcnt = new int[nprocs];
	memset(sendcnt, 0, sizeof(int) * nprocs);

	for(size_t i=offset; i< nreads; ++i){
		if (seqs[i].length() <= KMER_LENGTH) continue;
		all_seq_size += seqs[i].length();
	}
	if(all_seq_size == 0) return nreads;

	for(size_t i=offset; i < nreads; ++i){
		size_t found  = seqs[i].length();

		if (seqs[i].length() <= KMER_LENGTH) continue;

		nkmers_smer_batch += seqs[i].length() - KMER_LENGTH + 1;
		int approx_maxsending = (4 * nkmers_thisBatch + nprocs - 1)/nprocs;
		all_seqs += seqs[i] + "a";

		// if (approx_maxsending * bytesperentry >= memoryThreshold || (nkmers_thisBatch + seqs[i].length()) * bytesperentry >= MAX_ALLTOALL_MEM) {
		// 	nreads = i+1; // start with next read
		// 	break;
		// }
	}

	nkmers_smer_all += nkmers_smer_batch;
	int *owner_counter = (int*) malloc (nprocs * sizeof(int)) ;
	memset(owner_counter, 0, nprocs * sizeof(int));

	std::vector<keyType> h_send_smers(nkmers_smer_batch * BUFF_SCALE);
	std::vector<unsigned char> h_send_slens(nkmers_smer_batch * BUFF_SCALE);

	//* Build Supermers on GPU */

	// getSupermers_CPU_DEBUG(seqs_arr, KMER_LENGTH, MINIMIZER_LENGTH, nprocs, owner_counter, h_send_smers, h_send_slens, nkmers_smer_batch,  myrank);
	getSupermers_GPU(all_seqs, KMER_LENGTH, MINIMIZER_LENGTH, nprocs, owner_counter,
		h_send_smers, h_send_slens, nkmers_smer_batch,  myrank, BUFF_SCALE,
		minimizer_ordering, uhs_frequencies_hashtable, uhs_hashtable_capacity,
		uhs_sample_fraction, uhs_mmers, uhs_mmers_count, uhs_mpi_mode);
	tot_GPUsmer_build += MPI_Wtime() -  start_gpu_smer ;

	//* Exchange supermers on CPU */

	vector<keyType> recvbuf (nkmers_smer_batch * BUFF_SCALE );
	vector<unsigned char> recvbuf_len (nkmers_smer_batch * BUFF_SCALE );

	double exch_gpu_smer = Exchange_GPUsupermers(h_send_smers, h_send_slens, recvbuf,
	recvbuf_len, sendcnt, recvcnt, nkmers_smer_batch, owner_counter);
	tot_GPUsmer_exch += exch_gpu_smer;//MPI_Wtime() -  start_gpu_smer ;

	//* Parse supermers and build kcounter on GPU */

	start_gpu_smer = MPI_Wtime();
	size_t num_keys = 0;
	for(uint64_t i= 0; i < nprocs ; ++i)
		num_keys += recvcnt[i];
	nSupermers_all += num_keys;

	int p_buff_len = ((nkmers_smer_batch * BUFF_SCALE) + nprocs - 1)/nprocs;

	GPU_SP_buildCounter(d_hashTable, recvbuf, recvbuf_len, recvcnt, num_keys, KMER_LENGTH,
	 myrank, p_buff_len);
	tot_GPU_smer_kcounter += MPI_Wtime() -  start_gpu_smer ;

	//***** Correctness check of Kmer counter *****
	std::vector<KeyValue> h_pHashTable(kHashTableCapacity);
	cudaMemcpy(h_pHashTable.data(), d_hashTable, sizeof(KeyValue) * kHashTableCapacity, cudaMemcpyDeviceToHost);

	for (int i = 0; i < kHashTableCapacity; ++i){
		if (h_pHashTable[i].value > 0) { //kEmpty
			HTsize_smer++;
			totalPairs_smer += h_pHashTable[i].value;
		}
	}

	size_t allrank_hashsize = 0, allrank_totalPairs = 0;
	CHECK_MPI( MPI_Reduce(&HTsize_smer, &allrank_hashsize,  1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD) );
	CHECK_MPI( MPI_Reduce(&totalPairs_smer, &allrank_totalPairs, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD) );

	size_t allrank_kmersthisbatch = 0;
	size_t allrank_kmersprocessed = 0;
	// CHECK_MPI( MPI_Reduce(&nkmers_smer_batch, &allrank_kmersthisbatch, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD) );
	CHECK_MPI( MPI_Reduce(&nkmers_smer_all, &allrank_kmersprocessed, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD) );

	if(myrank == 0){
	  	cout << "\nBatch: " << batch <<" - Smer based: GPU HTsize: " << allrank_hashsize << ", #kmers from HT: " << allrank_totalPairs
		<< " expected #kmers " << allrank_kmersprocessed << endl;
	}

	return nreads;
}

double Exchange(vector< vector<Kmer> > & outgoing, vector< vector< ReadId > > & readids, vector< vector< PosInRead > > & positions, vector<vector<array<char,2>>> & extquals, vector<vector<array<char,2>>> & extseqs,
		vector<Kmer> & mykmers, vector< ReadId > & myreadids, vector< PosInRead > & mypositions, /*vector<array<char,2>> & myquals, vector<array<char,2>> & myseqs,*/ int pass, Buffer scratch1, Buffer scratch2)
{
	MPI_Pcontrol(1,"Exchange");
	double tot_exch_time = MPI_Wtime();
	double performance_report_time = 0.0;

	//
	// count and exchange number of bytes being sent
	// first pass: just k-mer (instances)
	// second pass: each k-mer (instance) with its source read (ID) and position
	//
	size_t bytesperkmer = Kmer::numBytes();
	size_t bytesperentry = bytesperkmer + (pass == 2 ? sizeof(ReadId) + sizeof(PosInRead) : 0);
	int * sendcnt = new int[nprocs];
	for(int i=0; i<nprocs; ++i) {
		sendcnt[i] = (int) outgoing[i].size() * bytesperentry;
		// cout << "From Exchage: " << outgoing[i].size() << " " << bytesperentry << endl;
		if (pass == 2) {
			ASSERT( outgoing[i].size() == readids[i].size(),"" );
			ASSERT( outgoing[i].size() == positions[i].size(),"" );
		} else {
			ASSERT (readids[i].size() == 0,"");
			ASSERT (positions[i].size() == 0,"");
		}
	}
	int * sdispls = new int[nprocs];
	int * rdispls = new int[nprocs];
	int * recvcnt = new int[nprocs];
	CHECK_MPI( MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, MPI_COMM_WORLD) );  // share the request counts

	sdispls[0] = 0;
	rdispls[0] = 0;
	for(int i=0; i<nprocs-1; ++i) {
		if (sendcnt[i] < 0 || recvcnt[i] < 0) {
			cerr << myrank << " detected overflow in Alltoall" << endl;
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		sdispls[i+1] = sdispls[i] + sendcnt[i];
		rdispls[i+1] = rdispls[i] + recvcnt[i];
		if (sdispls[i+1] < 0 || rdispls[i+1] < 0) {
			cerr << myrank << " detected overflow in Alltoall" << endl;
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}
	int64_t totsend = accumulate(sendcnt, sendcnt+nprocs, static_cast<int64_t>(0));
	if (totsend < 0) { cerr << myrank << " detected overflow in totsend calculation, line" << __LINE__ << endl; }
	int64_t totrecv = accumulate(recvcnt, recvcnt+nprocs, static_cast<int64_t>(0));
	if (totrecv < 0) { cerr << myrank << " detected overflow in totrecv calculation, line" << __LINE__ << endl; }
	DBG("totsend=%lld totrecv=%lld\n", (lld) totsend, (lld) totrecv);

	growBuffer(scratch1, sizeof(uint8_t) * totsend); // will exit if totsend is negative

	uint8_t * sendbuf = (uint8_t*) getStartBuffer(scratch1);
	for(int i=0; i<nprocs; ++i)  {
		size_t nkmers2send = outgoing[i].size();
		uint8_t * addrs2fill = sendbuf+sdispls[i];
		for(size_t j=0; j< nkmers2send; ++j) {
			ASSERT(addrs2fill == sendbuf+sdispls[i] + j*bytesperentry,"");
			(outgoing[i][j]).copyDataInto( addrs2fill );

			if (pass == 2) {
				ReadId* ptrRead = (ReadId*) (addrs2fill + bytesperkmer);
				ptrRead[0] = readids[i][j];
				PosInRead* ptrPos = (PosInRead*) (addrs2fill + bytesperkmer + sizeof(ReadId));
				ptrPos[0] = positions[i][j];
			}
			/* not exchanging extensions in longread version
			   if (pass == 2) {
			   char *ptr = ((char*) addrs2fill) + bytesperkmer;
			   ptr[0] = extquals[i][j][0];
			   ptr[1] = extquals[i][j][1];
			   ptr[2] = extseqs[i][j][0];
			   ptr[3] = extseqs[i][j][1];
			   }
			 */
			addrs2fill += bytesperentry;
		}
		outgoing[i].clear();
		readids[i].clear();
		positions[i].clear();
		extquals[i].clear();
		extseqs[i].clear();
	}

	growBuffer(scratch2, sizeof(uint8_t) * totrecv);
	uint8_t * recvbuf = (uint8_t*) getStartBuffer(scratch2);
	int total_count = 0;

	double exch_time = 0.0 - MPI_Wtime();
	CHECK_MPI( MPI_Alltoallv(sendbuf, sendcnt, sdispls, MPI_BYTE, recvbuf, recvcnt, rdispls, MPI_BYTE, MPI_COMM_WORLD) );
	exch_time += MPI_Wtime();

	/******* Performance reporting *******/
	performance_report_time = MPI_Wtime();
	const int SND=0, RCV=1;
	int64_t local_counts[2];
	local_counts[SND] = totsend;
	local_counts[RCV] = totrecv;

	int64_t global_mins[2]={0,0};
	CHECK_MPI( MPI_Reduce(&local_counts, &global_mins, 2, MPI_LONG_LONG, MPI_MIN, 0, MPI_COMM_WORLD) );
	int64_t global_maxs[2]={0,0};
	CHECK_MPI( MPI_Reduce(&local_counts, &global_maxs, 2, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD) );

	double global_min_time = 0.0;
	CHECK_MPI( MPI_Reduce(&exch_time, &global_min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD) );
	double global_max_time = 0.0;
	CHECK_MPI( MPI_Reduce(&exch_time, &global_max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD) );

	serial_printf("KMER based KmerMatch:%s exchange iteration %d pass %d: sent min %lld bytes, sent max %lld bytes, recv min %lld bytes, recv max %lld bytes, in min %.3f s, max %.3f s\n",
			__FUNCTION__, exchange_iter, pass, global_mins[SND], global_maxs[SND], global_mins[RCV], global_maxs[RCV], global_min_time, global_max_time);
	performance_report_time = MPI_Wtime()-performance_report_time;
	/*************************************/

	uint64_t nkmersrecvd = totrecv / bytesperentry;
	for(uint64_t i= 0; i < nkmersrecvd; ++i) {
		Kmer kk;
		kk.copyDataFrom(recvbuf + (i * bytesperentry));
		mykmers.push_back(kk);

		if (pass == 2) {
			ReadId *ptr = (ReadId*) (recvbuf + (i * bytesperentry) + bytesperkmer);
			ASSERT(ptr[0] > 0,"");
			myreadids.push_back(ptr[0]);
			PosInRead *posPtr = (PosInRead*) (recvbuf + (i * bytesperentry) + bytesperkmer + sizeof(ReadId));
			mypositions.push_back(posPtr[0]);
		}
		/* not exchanging extensions in longread version
		   char *ptr = ((char*) recvbuf) + (i * bytesperentry) + bytesperkmer;
		   if (pass == 2) {
		   array<char,2> qualexts = { ptr[0], ptr[1] };
		   array<char,2> seqexts = { ptr[2], ptr[3] };
		   myquals.push_back(qualexts);
		   myseqs.push_back(seqexts);
		   }
		 */
	}

	DBG("DeleteAll: recvcount=%lld, sendct=%lld\n", (lld) recvcnt, (lld) sendcnt);
	DeleteAll(rdispls, sdispls, recvcnt, sendcnt);

	//serial_printf("exchanged totsend=%lld, totrecv=%lld, pass=%d\n", (lld) totsend, (lld) totrecv, pass);
	exchange_iter++;
	tot_exch_time=MPI_Wtime()-tot_exch_time-performance_report_time;
	MPI_Pcontrol(-1,"Exchange");
	return tot_exch_time;
}

int64_t rsrv = 0; //TODO:: make it local

typedef struct {
	double duration, parsingTime, getKmerTime, lexKmerTime, hllTime, hhTime;
	double last;
} MoreHLLTimers;
inline double getDuration(MoreHLLTimers &t) {
	double delta = t.last;
	t.last = MPI_Wtime();
	return t.last - delta;
}
MoreHLLTimers InsertIntoHLL(vector<string> & seqs, vector<string> & quals, HyperLogLog & hll, bool extraTimers = false)
{
	size_t locreads = seqs.size();
	MoreHLLTimers t;
	if (extraTimers) {
		memset(&t, 0, sizeof(MoreHLLTimers));
		getDuration(t);
		t.duration = t.last;
	}

	readsprocessed += locreads;

	MPI_Pcontrol(1,"HLL_Parse");
	for(size_t i=0; i< locreads; ++i)
	{
		size_t found = seqs[i].length();
		if(found >= KMER_LENGTH) // otherwise size_t being unsigned will underflow
		{
			if (extraTimers) t.parsingTime += getDuration(t);
			std::vector<Kmer> kmers = Kmer::getKmers(seqs[i]); // calculate all the kmers
			if (extraTimers) t.getKmerTime += getDuration(t);

			ASSERT(kmers.size() >= found-KMER_LENGTH+1,"");
			size_t Nfound = seqs[i].find('N');
			for(size_t j=0; j< found-KMER_LENGTH+1; ++j)
			{
				ASSERT(kmers[j] == Kmer(seqs[i].c_str() + j),"");
				while (Nfound!=std::string::npos && Nfound < j) Nfound=seqs[i].find('N', Nfound+1);
				if (Nfound!=std::string::npos && Nfound < j+KMER_LENGTH) continue;	// if there is an 'N', toss it
				Kmer &mykmer = kmers[j];

				if (extraTimers) t.parsingTime += getDuration(t);
				Kmer lexsmall =  mykmer.rep();
				if (extraTimers) t.lexKmerTime += getDuration(t);
				hll.add((const char*) lexsmall.getBytes(), lexsmall.getNumBytes());
				if (extraTimers) t.hllTime += getDuration(t);
#ifdef HEAVYHITTERS
				if (heavyhitters) heavyhitters->Push(lexsmall);
				if (extraTimers) t.hhTime += getDuration(t);
#endif
			}
		}
	}
	MPI_Pcontrol(-1,"HLL_Parse");
	seqs.clear();
	quals.clear();
	if (extraTimers) {
		t.last = t.duration;
		t.duration = getDuration(t);
	}
	return t;
}

void ProudlyParallelCardinalityEstimate(const vector<filedata> & allfiles, double & cardinality, bool cached_io, const char* base_dir)
{

	HyperLogLog hll(12);
	int numFiles = allfiles.size();
	double *_times = (double*) calloc(numFiles*16, sizeof(double));
	double *pfq_times = _times;
	double *read_times = pfq_times + numFiles;
	double *HH_reduce_times = read_times + numFiles;
	double *HH_max_times = HH_reduce_times + numFiles;
	double *tot_times = HH_max_times + numFiles;
	double *max_times = tot_times + numFiles;
	double *min_times = max_times + numFiles;
	double *HLL_times = min_times + numFiles;
	double *HLL_avg_times = HLL_times + numFiles;
	double *HLL_max_times = HLL_avg_times + numFiles;
	double *avg_per_file_times = HLL_max_times + numFiles;
	double *max_per_file_times = avg_per_file_times + numFiles;
	double *tot_HH_times = max_per_file_times + numFiles;
	double *max_HH_times = tot_HH_times + numFiles;

	for(auto itr= allfiles.begin(); itr != allfiles.end(); itr++) {
		double start_t = MPI_Wtime();
		double hll_time = 0.0;
		int idx = itr - allfiles.begin();
		ParallelFASTQ pfq;
		pfq.open(itr->filename, cached_io, base_dir, itr->filesize);
		// The fastq file is read line by line, so the number of records processed in a block
		// shouldn't make any difference, so we can just set this to some arbitrary value.
		// The value 262144 is for records with read lengths of about 100.
		size_t upperlimit = MAX_ALLTOALL_MEM / 16;
		size_t totalsofar = 0;
		vector<string> names;
		vector<string> seqs;
		vector<string> quals;
		vector<Kmer> mykmers;
		int iterations = 0;

		while (1) {
			MPI_Pcontrol(1,"FastqIO");
			iterations++;
			size_t fill_status = pfq.fill_block(names, seqs, quals, upperlimit);
			// Sanity checks
			ASSERT(seqs.size() == quals.size(),"");
			for (int i = 0; i < seqs.size(); i++) {
				if (seqs[i].length() != quals[i].length()) {
					DIE("sequence length %lld != quals length %lld\n%s\n",
							(lld) seqs[i].length(), (lld) quals[i].length(), seqs[i].c_str());
				}
			}
			MPI_Pcontrol(-1,"FastqIO");
			if (!fill_status)
				break;
			double start_hll = MPI_Wtime();
			int64_t numSeqs = seqs.size();
			// this clears the vectors
			MoreHLLTimers mt = InsertIntoHLL(seqs, quals, hll, true);
			names.clear();
			hll_time += MPI_Wtime() - start_hll;
			LOGF("HLL timings: iteration %d, seqs %lld, duration %0.4f, parsing %0.4f, getKmer %0.4f, lexKmer %0.4f, hllTime %0.4f, hhTime %0.4f\n", iterations, (lld) numSeqs, mt.duration, mt.parsingTime, mt.getKmerTime, mt.lexKmerTime, mt.hllTime, mt.hhTime);
		}

		HLL_times[ idx ] = hll_time;
		pfq_times[ idx ] = pfq.get_elapsed_time();

#ifdef HEAVYHITTERS
		if (heavyhitters)
			HH_reduce_times[ idx ] = heavyhitters->GetReduceTime() - (idx == 0 ? 0.0 : HH_reduce_times[idx-1]);
		else
			HH_reduce_times[idx] = 0.0;

		HH_max_times[idx] = HH_reduce_times[idx];
#endif // HEAVYHITTERS

#ifdef DEBUG
		CHECK_MPI( MPI_Reduce(read_times+idx,tot_times+idx, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) );
		if (myrank == 0) {
			int num_ranks;
			double t2 = pfq.get_elapsed_time();
			CHECK_MPI( MPI_Comm_size(MPI_COMM_WORLD, &num_ranks) );
			cout << __FUNCTION__ << ": Total time taken for FASTQ reads is " << (tot_times[idx] / num_ranks) << ", elapsed " << t2 << endl;
		}
#endif // DEBUG
		read_times[ idx ] = MPI_Wtime() - start_t;

	}

#ifdef HEAVYHITTERS
	if (heavyhitters && myrank == 0) {
		cout << "Thread " << myrank << " HeavyHitters performance: " << heavyhitters->getPerformanceStats() << endl;
	}
#endif // HEAVYHITTERS

#ifndef DEBUG
	CHECK_MPI( MPI_Reduce(pfq_times,tot_times, allfiles.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) );
	CHECK_MPI( MPI_Reduce(pfq_times,max_times, allfiles.size(), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD) );
	CHECK_MPI( MPI_Reduce(pfq_times,min_times, allfiles.size(), MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD) );
	CHECK_MPI( MPI_Reduce(HH_reduce_times, tot_HH_times, allfiles.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) );
	CHECK_MPI( MPI_Reduce(HH_max_times, max_HH_times, allfiles.size(), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD) );
	CHECK_MPI( MPI_Reduce(HLL_times, HLL_avg_times, allfiles.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) );
	CHECK_MPI( MPI_Reduce(HLL_times, HLL_max_times, allfiles.size(), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD) );

	// maximum elapsed time per file
	CHECK_MPI( MPI_Reduce(read_times, avg_per_file_times, allfiles.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) );
	CHECK_MPI( MPI_Reduce(read_times, max_per_file_times, allfiles.size(), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD) );
	if (myrank == 0) {
		int num_ranks;
		CHECK_MPI( MPI_Comm_size(MPI_COMM_WORLD, &num_ranks) );
		for(int i = 0; i < allfiles.size(); i++) {
			cout << __FUNCTION__ << ": Total time taken for FASTQ file " << allfiles[i].filename << " size: "
				<< allfiles[i].filesize << " avg: " << tot_times[i] / num_ranks << " max: " << max_times[i]
				<< " min: " << min_times[i] << " HLL avg: " << HLL_avg_times[i]/num_ranks << " HLL max: "
				<< HLL_max_times[i] << " HH avg: " << HH_reduce_times[i] / num_ranks << " HH max: " << HH_max_times[i]
				<< " elapsed_avg: " << avg_per_file_times[i]/num_ranks << " elapsed_max: " << max_per_file_times[i]
				<< " seconds" << endl;
		}
	}
#endif // ndef DEBUG
	LOGF("My cardinality before reduction: %f\n", hll.estimate());

	// using MPI_UNSIGNED_CHAR because MPI_MAX is not allowed on MPI_BYTE
	int count = hll.M.size();
	CHECK_MPI( MPI_Allreduce(MPI_IN_PLACE, hll.M.data(), count, MPI_UNSIGNED_CHAR, MPI_MAX, MPI_COMM_WORLD) );
	CHECK_MPI( MPI_Allreduce(MPI_IN_PLACE, &readsprocessed, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD) );

	cardinality = hll.estimate();
	if(myrank == 0) {
		cout << __FUNCTION__ << ": Embarrassingly parallel k-mer count estimate is " << cardinality << endl;
		cout << __FUNCTION__ << ": Total reads processed over all processors is " << readsprocessed << endl;
		ADD_DIAG("%f", "cardinality", cardinality);
		ADD_DIAG("%lld", "total_reads", (lld) readsprocessed);
	}
	SLOG("%s: total cardinality %f\n", __FUNCTION__, cardinality);
#ifdef HEAVYHITTERS
	if (heavyhitters) {
		double hh_merge_start = MPI_Wtime();
		MPI_Pcontrol(1,"HeavyHitters");
		ParallelAllReduce(*heavyhitters);
		nfreqs = heavyhitters->Size();
		heavyhitters->CreateIndex(); // make it indexible
		double myMergeTime, avgMergeTime, maxMergeTime;
		myMergeTime = heavyhitters->GetMergeTime();
		CHECK_MPI( MPI_Reduce(&myMergeTime, &maxMergeTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD) );
		CHECK_MPI( MPI_Reduce(&myMergeTime, &avgMergeTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) );
		if (myrank == 0) {
			int num_ranks;
			CHECK_MPI( MPI_Comm_size(MPI_COMM_WORLD, &num_ranks) );
			cout << __FUNCTION__ << ":Total time taken for HH Merge " << (MPI_Wtime() - hh_merge_start)
				<< " avg HH merge: " << avgMergeTime / num_ranks << " max: " << maxMergeTime << endl;
		}
		Frequents = new UFX2ReduceObj[nfreqs]();    // default initialize
		MPI_Pcontrol(-1,"HeavyHitters");
	} else {
		SLOG("Skipping heavy hitter merge");
	}
#endif

	MPI_Barrier(MPI_COMM_WORLD);
	cardinality /= static_cast<double>(nprocs);	// assume a balanced distribution
	cardinality *= 1.1;	// 10% benefit of doubt
	LOGF("Adjusted per-process cardinality: %f\n", cardinality);
}

class KmerInfo {
	public:
		//typedef array<char,2> TwoChar;
		//typedef unsigned int ReadId;
	private:
		Kmer kmer;
		//TwoChar quals,seqs;
		ReadId readId;
		PosInRead position;
	public:
		KmerInfo() {}
		KmerInfo(Kmer k): kmer(k), readId( (ReadId) nullReadId), position( (PosInRead) initPos ) {}
		KmerInfo(Kmer k, ReadId r, PosInRead p): kmer(k), readId(r), position(p) {}
		//KmerInfo(Kmer k, ReadId r): kmer(k), quals(), seqs(), readId(r) {}
		//KmerInfo(Kmer k, TwoChar q, TwoChar s, ReadId r): kmer(k), quals(q), seqs(s), readId(r) {}
		KmerInfo(const KmerInfo &copy) {
			kmer = copy.kmer;
			//quals = copy.quals;
			//seqs = copy.seqs;
			readId = copy.readId;
			position = copy.position;
		}
		const Kmer& getKmer() const {
			return kmer;
		}
		int write(GZIP_FILE f) {
			int count = GZIP_FWRITE(this, sizeof(*this), 1, f);
#ifndef NO_GZIP
			if (count != sizeof(*this)*1) { DIE("There was a problem writing the kmerInfo file! %s\n", strerror(errno)); }
#else
			if (count != 1) { DIE("There was a problem writing the kmerInfo file! %s\n", strerror(errno)); }
#endif
			return count;
		}
		int read(GZIP_FILE f) {
			int count = GZIP_FREAD(this, sizeof(*this), 1, f);
#ifndef NO_GZIP
			if (count != sizeof(*this)*1 && !GZIP_EOF(f)) { DIE("There was a problem reading the kmerInfo file! %s\n", strerror(errno)); }
#else
			if (count != 1 && ! feof(f)) { DIE("There was a problem reading the kmerInfo file! %s\n", strerror(errno)); }
#endif
			return count;
		}
		// returns true if in bloom, does not modify
		bool checkBloom(struct bloom *bm) {
			MPI_Pcontrol(1,"BloomFilter");
			bool inBloom = bloom_check(bm, kmer.getBytes(), kmer.getNumBytes()) == 1;
			MPI_Pcontrol(-1,"BloomFilter");
			return inBloom;
		}
		// returns true if in bloom, inserts if not
		bool checkBloomAndRemember(struct bloom *bm) {
			bool inBloom = checkBloom(bm);
			if (!inBloom) {
				MPI_Pcontrol(1,"BloomFilter");
				bloom_add(bm, kmer.getBytes(), kmer.getNumBytes());
				MPI_Pcontrol(-1,"BloomFilter");
			}
			return inBloom;
		}
		// returns true when kmer is already in bloom, if in bloom, inserts into map, if andCount, increments map count
		bool checkBloomAndInsert(struct bloom *bm, bool andCount) {
			bool inBloom = true;//checkBloomAndRemember(bm);

			if (inBloom) {
				MPI_Pcontrol(1,"InsertOrUpdate");
				auto got = kmercounts->find(kmer.getArray());  // kmercounts is a global variable

				if(got == kmercounts->end())
				{
#ifdef KHASH
					kmercounts->insert(kmer.getArray(), make_tuple(newReadIdList(), newPositionsList(), 0));
#else

					// typedef VectorMap< Kmer::MERARR, KmerCountType, std::hash<Kmer::MERARR>, std::less<Kmer::MERARR>, std::equal_to<Kmer::MERARR> > KmerCountsType;
					kmercounts->insert(make_pair(kmer.getArray(), make_tuple(newReadIdList(), newPositionsList(), 0)));
#endif
					// if (andCount) includeCount(false);
				}
				else {
					// if (andCount) includeCount(got);
				}
				MPI_Pcontrol(-1,"InsertOrUpdate");
			}
			auto got = kmercounts->find(kmer.getArray());
			includeCount(got);
			return inBloom;
		}

		void updateReadIds(KmerCountsType::iterator got) {
#ifdef KHASH
			READIDS reads = get<0>(*got);  // ::value returns (const valtype_t &) but ::* returns (valtype_t &), which can be changed
			POSITIONS& positions = get<1>(*got);
#else
			READIDS& reads = get<0>(got->second);
			POSITIONS& positions = get<1>(got->second);
#endif
			ASSERT(readId > nullReadId,"");

			// never add duplicates, also currently doesn't support more than 1 positions per read ID
			int index;
			for (index = 0; index < reliable_max && reads[index] > nullReadId; index++) {
				if (reads[index] == readId) return;
			}
			// if the loop finishes without returning, the index is set to the next open space or there are no open spaces
			if (index >= reliable_max || reads[index] > nullReadId) return;
			ASSERT(reads[index] == nullReadId, "reads[index] does not equal expected value of nullReadId");
			reads[index] = readId;
			positions[index] = position;
		}

		bool includeCount(bool doStoreReadId) {
			auto got = kmercounts->find(kmer.getArray());  // kmercounts is a global variable
			auto tmp = kmer.getArray();
			// cout << "got array " << tmp[0] << endl;
			if ( doStoreReadId && (got != kmercounts->end()) ) { updateReadIds(got); }
			return includeCount(got);
		}

		bool includeCount(KmerCountsType::iterator got) {
			MPI_Pcontrol(1,"HashTable");
			bool inserted = false;
			if(got != kmercounts->end()) // don't count anything else
			{
				// count the kmer in mercount
#ifdef KHASH
				++(get<2>(*got));  // ::value returns (const valtype_t &) but ::* returns (valtype_t &), which can be changed
#else
				++(get<2>(got->second)); // increment the counter regardless of quality extensions
				// cout << get<0>(got->first) << " " << get<2>(got->second)  <<endl;

#endif
				inserted = true;
			}
			MPI_Pcontrol(-1,"HashTable");
			return inserted;
		}
};


		// at this point, no kmers include anything other than uppercase 'A/C/G/T'
void DealWithInMemoryData(vector<Kmer> & mykmers, int pass, struct bloom * bm, vector< ReadId > myreadids, vector< PosInRead > mypositions)
{
	// store kmer & extensions with confirmation of bloom
	// store to disk first time kmers with (first) insert into bloom
	// pass 1 - just store kmers in bloom and insert into count, pass 2 - count
	//LOGF("Dealing with memory pass %d: mykmers=%lld, myseqs=%lld\n", pass, (lld) mykmers.size(), (lld) myseqs.size());

	LOGF("Dealing with memory pass %d: mykmers=%lld\n", pass, (lld) mykmers.size());

	if (pass == 2) {
		ASSERT(myreadids.size() == mykmers.size(),"");
		ASSERT(mypositions.size() == mykmers.size(),"");
	}
	if (pass == 1) {
		ASSERT(myreadids.size() == 0,"");
		ASSERT(mypositions.size() == 0,"");
	}

	if(pass == 1)
	{
		assert(bm);
		MPI_Pcontrol(1,"BloomFilter");
		size_t count = mykmers.size();
		for(size_t i=0; i < count; ++i)
		{
			// there will be a second pass, just insert into bloom, and map, but do not count
			KmerInfo ki(mykmers[i]);
			ki.checkBloomAndInsert(bm, false);
		}

		MPI_Pcontrol(-1,"BloomFilter");
	} else {
		ASSERT(pass==2,"");
		MPI_Pcontrol(1,"CountKmers");
		size_t count = mykmers.size();
		for(size_t i=0; i < count; ++i)
		{
			KmerInfo ki(mykmers[i], myreadids[i], mypositions[i]);
#ifdef HEAVYHITTERS
			ASSERT( !heavyhitters || ! heavyhitters->IsMember(mykmers[i]),"" );
#endif
			ASSERT(!bm,"");
			ki.includeCount(true);
		}
		MPI_Pcontrol(-1,"CountKmers");
	}

}

size_t ProcessFiles(const vector<filedata> & allfiles, int pass, double & cardinality, bool cached_io,
		const char* base_dir, ReadId & readIndex, std::unordered_map<ReadId, std::string>& readNameMap)
{
	assert(base_dir != NULL);
	struct bloom * bm = NULL;

	int exchangeAndCountPass = pass;

	Buffer scratch1 = initBuffer(MAX_ALLTOALL_MEM);
	Buffer scratch2 = initBuffer(MAX_ALLTOALL_MEM);

	// initialize bloom filter
	if(pass == 1) {
		if(type == 1 || type == 3) // kcoutner on GPU
			d_hashTable = create_hashtable_GPU(myrank); // IN
		// Only require bloom in pass 1!!
		unsigned int random_seed = 0xA57EC3B2;
		const double desired_probability_of_false_positive = 0.05;
		bm = (struct bloom*) malloc(sizeof(struct bloom));
#ifdef HIPMER_BLOOM64
		bloom_init64(bm, cardinality, desired_probability_of_false_positive);
#else
		assert(cardinality < 1L<<32);
		bloom_init(bm, cardinality, desired_probability_of_false_positive);
#endif

		if(myrank == 0)
		{
			cout << __FUNCTION__ << " pass " << pass << ": Table size is: " << bm->bits << " bits, " << ((double)bm->bits)/8/1024/1024 << " MB" << endl;
			cout << __FUNCTION__ << " pass " << pass << ": Optimal number of hash functions is : " << bm->hashes << endl;
		}

		LOGF("Initialized bloom filter with %lld bits and %d hash functions\n", (lld) bm->bits, (int) bm->hashes);
	}
	std::vector<int> maxReadLengths;

	uhs_hashtable_slot* uhs_frequencies_hashtable = NULL;
	uint64_t uhs_hashtable_capacity = 0;
	uhs_key_t* uhs_mmers;
	uint64_t uhs_mmers_count;

	if (minimizer_ordering == 1) {
		uhs_frequencies_hashtable = initialize_uhs_frequencies_hashtable(
			uhs_file_path,
			MINIMIZER_LENGTH,
			myrank,
			&uhs_hashtable_capacity,
			&uhs_mmers,
			&uhs_mmers_count
		);
	}

	int nReads = 0;
	double pfqTime = 0.0;
	auto files_itr = allfiles.begin();
	int trunc_ret;
	double t01 = MPI_Wtime(), t02;
	double tot_pack = 0.0, tot_exch = 0.0, tot_process = 0.0, tot_raw = 0.0;

	while(files_itr != allfiles.end())
	{
		ParallelFASTQ *pfq = new ParallelFASTQ();
		pfq->open(files_itr->filename, cached_io, base_dir, files_itr->filesize);
		LOGF("Opened %s of %lld size\n", files_itr->filename, (lld) files_itr->filesize);
		files_itr++;
		// once again, arbitrarily chosen - see ProudlyParallelCardinalityEstimate
		size_t upperlimit = MAX_ALLTOALL_MEM / 16;

		vector<string> names;
		vector<string> seqs;
		vector<string> quals;
		vector< vector<Kmer> > outgoing(nprocs);
		vector< vector<array<char,2> > > extquals(nprocs);
		vector< vector<array<char,2> > > extseqs(nprocs);
		vector< vector< ReadId > > readids(nprocs);
		vector< vector< PosInRead> > positions(nprocs);

		vector<Kmer> mykmers;
		vector<array<char,2>> myquals;
		vector<array<char,2>> myseqs;
		vector< ReadId > myreadids;
		vector< PosInRead > mypositions;

		int moreflags[3], allmore2go[3], anymore2go;
		int &moreSeqs = moreflags[0], &moreToExchange = moreflags[1], &moreFiles = moreflags[2];
		int &allmoreSeqs = allmore2go[0], &allmoreToExchange = allmore2go[1], &allmoreFiles = allmore2go[2];
		moreSeqs = 1; // assume more as file is just open
		moreFiles = (files_itr != allfiles.end());
		moreToExchange = 0; // no data yet
		size_t fill_status;
		int exchanges = 0;

		do { // extract raw data into seqs and quals
			DBG("Starting new round: moreSeqs=%d, moreFiles=%d\n", moreSeqs, moreFiles);
			MPI_Pcontrol(1,"FastqIO");
			double t_temp = MPI_Wtime();
			do {
				DBG2("Filling a block: moreSeqs=%d, moreFiles=%d\n", moreSeqs, moreFiles);
				// fill a block, from this or the next file
				if (pfq && !moreSeqs) {
					assert(pfq != NULL);
					double t = pfq->get_elapsed_time();
					pfqTime += t;
					DBG2("Closed last file %.3f sec\n", t);
					maxReadLengths.push_back(pfq->get_max_read_len());
					delete pfq;
					pfq = NULL;
					if (files_itr != allfiles.end()) {
						// finished reading the last file, open the next file
						pfq = new ParallelFASTQ();
						pfq->open(files_itr->filename, cached_io, base_dir, files_itr->filesize);
						DBG2("Opened new file %s of %lld size\n", files_itr->filename, (lld) files_itr->filesize);
						files_itr++;
						moreSeqs = 1;
					}
				}
				moreFiles = (files_itr != allfiles.end());
				fill_status = 0;
				if(moreSeqs) { // fill another block from the same or newly opened file
					fill_status = pfq->fill_block(names, seqs, quals, upperlimit);  // file_status is 0 if fpos >= end_fpos
					long long llf = fill_status;
					DBG2("Filled block to %lld\n",(lld)  llf);
					assert(fill_status == seqs.size());
					assert(fill_status == quals.size());
				}
				moreSeqs = (fill_status > 0);
			} while (moreFiles && !moreSeqs); // exit when a block is full IOR there are no more files
			MPI_Pcontrol(-1,"FastqIO");

			nReads += seqs.size();
			tot_raw += MPI_Wtime() - t_temp; // pfqTime will be subtracted at end

			size_t offset = 0;
			do { // extract kmers and counts from read sequences (seqs)
				DBG("Starting Exchange - ParseNPack %lld (%lld)\n", (lld) offset, (lld) seqs.size());
				batch++;
				if(myrank == 0) printf("\n\n**** Starting batch:  %d\n",  batch);
				int tmp_offset = offset;

				double exch_start_t = MPI_Wtime();
				DBG("%s %d : before ParseNPack, readIndex=%lld\n", __FUNCTION__, __LINE__, readIndex);
				offset = ParseNPack(seqs, names, quals, outgoing, readids, positions, readIndex, extquals, extseqs, readNameMap, exchangeAndCountPass, offset);    // no-op if seqs.size() == 0
				DBG("%s %d : after ParseNPack, readIndex=%lld\n", __FUNCTION__, __LINE__, readIndex);
				double pack_t = MPI_Wtime() - exch_start_t;
				tot_pack += pack_t;

				double exch_t = 0;

				if(type == 0){ // Original diBella on CPU
					exch_t = Exchange(outgoing, readids, positions, extquals, extseqs, mykmers, myreadids, mypositions, /*myquals, myseqs,*/ exchangeAndCountPass, scratch1, scratch2); // outgoing arrays will be all empty, shouldn't crush
					tot_exch += exch_t;

					DealWithInMemoryData(mykmers, exchangeAndCountPass, bm, myreadids, mypositions);   // we might still receive data even if we didn't send any
				}

				else if(type == 1){ // diBella on GPU
					double exch_start_t_GPU = MPI_Wtime();
					kmerCount_GPU(seqs, exchangeAndCountPass, tmp_offset, offset, nprocs, bm);    // no-op if seqs.size() == 0
					double pack_t_GPU = MPI_Wtime() - exch_start_t_GPU;
					tot_pack_GPU += pack_t_GPU;
				}

				else if(type == 2) // Supermer based kcounter on CPU
					spmer_kmerCount(seqs, tmp_offset, offset, KMER_LENGTH, MINIMIZER_LENGTH);

				else if (type == 3) { // Supermer based kcounter on GPU
					spmer_kmerCount_GPU(seqs, outgoing, exchangeAndCountPass, tmp_offset, offset, minimizer_ordering, uhs_frequencies_hashtable, uhs_hashtable_capacity, uhs_mmers, uhs_mmers_count);    // no-op if seqs.size() == 0
				}

				double process_t = MPI_Wtime() - exch_start_t - pack_t - exch_t;
				tot_process += process_t;
				if (offset == seqs.size()) {
					seqs.clear();	// no need to do the swap trick as we will reuse these buffers in the next iteration
					quals.clear();
					offset = 0;
				}

				// #endif
				moreToExchange = offset < seqs.size(); // TODO make sure this isn't problematic in long read version

				mykmers.clear();
				myreadids.clear();
				mypositions.clear();
				myquals.clear();
				myseqs.clear();

				// double process_t_GPU = MPI_Wtime() - exch_start_t_GPU - pack_t_GPU - exch_t_GPU;


				DBG("Processed (%lld).  remainingToExchange=%lld %0.3f sec\n", (lld) mykmers.size(), (lld) seqs.size() - offset, process_t);

				DBG("Checking global state: moreSeqs=%d moreToExchange=%d moreFiles=%d\n", moreSeqs, moreToExchange, moreFiles);
				CHECK_MPI( MPI_Allreduce(moreflags, allmore2go, 3, MPI_INT, MPI_SUM, MPI_COMM_WORLD) );
				DBG("Got global state: allmoreSeqs=%d allmoreToExchange=%d allmoreFiles=%d\n", allmoreSeqs, allmoreToExchange, allmoreFiles);
				double now = MPI_Wtime();
				LOGF("Exchange timings pack: %0.3f exch: %0.3f process: %0.3f elapsed: %0.3f\n", pack_t, exch_t, process_t, now - t01);
			} while (moreToExchange);
			anymore2go = allmoreSeqs + allmoreToExchange + allmoreFiles;
			exchanges++;
		} while(anymore2go);

		if (pfq) {
			double t = pfq->get_elapsed_time();
			pfqTime += t;
			DBG( "Closing last file: %.3f sec\n", t);
			maxReadLengths.push_back(pfq->get_max_read_len());
			delete pfq;
			pfq = NULL;
		}
		// Calculate the max_read_len for each of the input files and write to marker files
		writeMaxReadLengths( maxReadLengths, allfiles );
	}	// end_of_loop_over_all_files

	t02 = MPI_Wtime();
	tot_raw = tot_raw - pfqTime;
	double tots[6], gtots[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	double tots_GPU[8], gtots_GPU[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	tots[0] = pfqTime;
	tots[1] = tot_pack;
	tots[2] = tot_exch;
	tots[3] = tot_process;
	tots[4] = tot_raw;
	tots[5] = t02 - t01;
	tots_GPU[0] = tot_pack_GPU;
	tots_GPU[1] = tot_exch_GPU; //- (tot_alltoallv_GPU * (COMM_ITER - 1) );
	tots_GPU[2] = tot_process_GPU;
	tots_GPU[3] = tot_alltoallv_GPU;
	tots_GPU[4] = tot_GPUsmer_build;
	tots_GPU[5] = tot_GPUsmer_exch;// - (tot_GPUsmer_alltoallv * (COMM_ITER - 1) );;
	tots_GPU[6] = tot_GPU_smer_kcounter;
	tots_GPU[7] = tot_GPUsmer_alltoallv;

	// LOGF("Process Total times: fastq: %0.3f pack: %0.3f exch: %0.3f process: %0.3f elapsed: %0.3f\n", pfqTime, tot_pack, tot_exch, tot_process, tots[4]);

	CHECK_MPI( MPI_Reduce(&tots_GPU, &gtots_GPU, 8, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) );
	CHECK_MPI( MPI_Reduce(&tots, &gtots, 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) );
	//   if (myrank == 0) {
	//       int num_ranks;
	//       CHECK_MPI( MPI_Comm_size(MPI_COMM_WORLD, &num_ranks) );
	//       // cout << __FUNCTION__ << " pass " << pass << ": Average time taken for FASTQ reads is " << (gtots[0] / num_ranks) << ", myelapsed " << tots[0] << endl;
	//       // cout << __FUNCTION__ << " pass " << pass << ": Average time taken for packing reads is " << (gtots[1] / num_ranks) << ", myelapsed " << tots[1] << endl;
	//       // cout << __FUNCTION__ << " pass " << pass << ": Average time taken for exchanging reads is " << (gtots[2] / num_ranks) << ", myelapsed " << tots[2] << endl;
	//       // cout << __FUNCTION__ << " pass " << pass << ": Average time taken for processing reads is " << (gtots[3] / num_ranks) << ", myelapsed " << tots[3] << endl;
	//       // cout << __FUNCTION__ << " pass " << pass << ": Average time taken for other FASTQ processing is " << (gtots[4] / num_ranks) << ", myelapsed " << tots[4] << endl;
	//       // cout << __FUNCTION__ << " pass " << pass << ": Average time taken for elapsed is " << (gtots[5] / num_ranks) << ", myelapsed " << tots[5] << endl;
	// //   	cout << "\nTimings on GPU KMER:\n";
	// //   	cout << __FUNCTION__ << " pass " << pass << ": Average time taken for packing reads on GPU (incl. memcpy) is " << (gtots_GPU[0] / num_ranks) << ", myelapsed " << tots[1] << endl;
	// //       cout << __FUNCTION__ << " pass " << pass << ": Average time taken for xchanging reads on GPU is " << (gtots_GPU[1] / num_ranks) << ", myelapsed " << tots[1] << endl;
	// //       cout << __FUNCTION__ << " pass " << pass << ": Average time taken for processing reads (Kmer count) on GPU is " << (gtots_GPU[2] / num_ranks) << ", myelapsed " << tots[1] << endl;
	//       cout << "\nTimings on GPU supermer:\n";
	//   	cout << __FUNCTION__ << " pass " << pass << ": Average time taken for build supermer on GPU (incl. memcpy) is " << (gtots_GPU[3] / num_ranks) << ", myelapsed " << tots[1] << endl;
	//       cout << __FUNCTION__ << " pass " << pass << ": Average time taken for xchanging smer on GPU is " << (gtots_GPU[4] / num_ranks) << ", myelapsed " << tots[1] << endl;
	//       cout << __FUNCTION__ << " pass " << pass << ": Average time taken for Kmer count on GPU is " << (gtots_GPU[5] / num_ranks) << ", myelapsed " << tots[1] << endl;


	//   	printf("IN: Finished Pass %d, Freeing bloom and other memory. kmercounts: %lld entries\n", pass, (lld) kmercounts->size());
	//   }



	//***** Correctness check of Kmer counter *****
	if(type == 0 ){
		int64_t maxcount = 0;
		int64_t globalmaxcount = 0;
		int64_t hashsize = 0;
		for(auto itr = kmercounts->begin(); itr != kmercounts->end(); ++itr) {
#ifdef KHASH
			if(!itr.isfilled()) continue;   // not all entries are full in khash
			int allcount = get<2>( itr.value() );
#else
			int allcount =  get<2>( itr->second );
#endif
			if(allcount > maxcount)  maxcount = allcount;
			nonerrorkmers += allcount;
			++hashsize;
		}

		int64_t totalnonerror;
		int64_t distinctnonerror;
		CHECK_MPI( MPI_Reduce(&nonerrorkmers, &totalnonerror, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD) );
		CHECK_MPI( MPI_Reduce(&hashsize, &distinctnonerror, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD) );
		CHECK_MPI( MPI_Allreduce(&maxcount, &globalmaxcount, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD) );
		if(myrank == 0) {
			int num_ranks;
			CHECK_MPI( MPI_Comm_size(MPI_COMM_WORLD, &num_ranks) );

			cout << "\nCPU - K-mer based: HTsize: " << distinctnonerror << ", #kmers from HT: " << totalnonerror
				<< " expected #kmers " << endl;
			cout << "Avg timings: parse&pack: " << (gtots[1] / num_ranks) << ", exchange K-mers: " <<  (gtots[2] / num_ranks) << ", build kcounter: " << (gtots[3] / num_ranks) << endl;

		}
	}

	else if(type == 1 || type == 3) {// kcoutner on GPU

		size_t HTsize_GPU = 0, kmersFromHT_GPU = 0;
		std::vector<KeyValue> h_pHashTable(kHashTableCapacity);
		cudaMemcpy(h_pHashTable.data(), d_hashTable, sizeof(KeyValue) * kHashTableCapacity, cudaMemcpyDeviceToHost);

		for (int i = 0; i < kHashTableCapacity; ++i){
			if (h_pHashTable[i].value > 0) { // if (hosthash[i].value > 1 && hosthash[i].value < 8) { //kEmpty
				HTsize_GPU++;
				kmersFromHT_GPU += h_pHashTable[i].value;
			}
			}

			size_t allrank_hashsize = 0, allrank_totalPairs = 0, allrank_kmersprocessed = 0, allrank_nsmers = 0;
			CHECK_MPI( MPI_Reduce(&HTsize_GPU, &allrank_hashsize,  1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD) );
			CHECK_MPI( MPI_Reduce(&kmersFromHT_GPU, &allrank_totalPairs, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD) );
			CHECK_MPI( MPI_Reduce(&kmersprocessed, &allrank_kmersprocessed, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD) );

			CHECK_MPI( MPI_Reduce(&nSupermers_all, &allrank_nsmers, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD) );

			if(myrank == 0){

				int num_ranks;
				CHECK_MPI( MPI_Comm_size(MPI_COMM_WORLD, &num_ranks) );

				if(type == 1) { // K-mer based
					cout << "\nGPU - K-mer based: HTsize: " << allrank_hashsize << ", #kmers from HT: " << allrank_totalPairs
						<< " expected #kmers " << allrank_kmersprocessed << endl;
					cout << "Avg timings: parse&pack: " << (gtots_GPU[0] / num_ranks) << ", exchange K-mers: " <<  (gtots_GPU[1] / num_ranks)
						<< ", build kcounter: " << (gtots_GPU[2] / num_ranks) << ", alltoallv(): " << (gtots_GPU[3] / num_ranks) << endl;
				}

				else { // smer based
					cout << "\nGPU - Smer based: HTsize: " << allrank_hashsize << ", #kmers from HT: " << allrank_totalPairs
						<< " expected #kmers " << allrank_kmersprocessed << ", #supermers "<< allrank_nsmers <<  endl;
					cout << "Avg timings: Build smer: " << (gtots_GPU[4] / num_ranks) << ", exchange smers: " <<  (gtots_GPU[5] / num_ranks)
						<< ", build kcounter: " << (gtots_GPU[6] / num_ranks) << ", alltoallv(): " << (gtots_GPU[7] / num_ranks) << endl;
				}
			}
			destroy_hashtable(d_hashTable, myrank);
		}

		if(myrank == 0) {
			cout << __FUNCTION__ << " pass " << pass << ": Read/distributed/processed reads of " << (files_itr == allfiles.end() ? " ALL files " : files_itr->filename) << " in " << t02-t01 << " seconds" << endl;
		}

		LOGF("Finished Pass %d, Freeing bloom and other memory. kmercounts: %lld entries\n", pass, (lld) kmercounts->size());
		t02 = MPI_Wtime(); // redefine after prints

		if (bm) {
			LOGF("Freeing Bloom filter\n");
			bloom_free(bm);
			free(bm);
			bm = NULL;
		}

		freeBuffer(scratch1);
		freeBuffer(scratch2);

		if (exchangeAndCountPass == 2) {
			countTotalKmersAndCleanHash();
		}

		if (minimizer_ordering == 1) {
			cudaFree(uhs_frequencies_hashtable);
			cudaFree(uhs_mmers);
		}

		return nReads;

	}

			void countTotalKmersAndCleanHash() {
				MPI_Pcontrol(1,"HashClean");

				int64_t maxcount = 0;
				int64_t globalmaxcount = 0;
				int64_t hashsize = 0;
				for(auto itr = kmercounts->begin(); itr != kmercounts->end(); ++itr) {
#ifdef KHASH
					if(!itr.isfilled()) continue;   // not all entries are full in khash
					int allcount = get<2>( itr.value() );
#else
					int allcount =  get<2>( itr->second );
#endif
					if(allcount > maxcount)  maxcount = allcount;
					nonerrorkmers += allcount;
					++hashsize;
				}
				LOGF("my hashsize=%lld, nonerrorkmers=%lld, maxcount=%lld\n", (lld) hashsize, (lld) nonerrorkmers, (lld) maxcount);


				int64_t totalnonerror;
				int64_t distinctnonerror;
				CHECK_MPI( MPI_Reduce(&nonerrorkmers, &totalnonerror, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD) );
				CHECK_MPI( MPI_Reduce(&hashsize, &distinctnonerror, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD) );
				CHECK_MPI( MPI_Allreduce(&maxcount, &globalmaxcount, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD) );
				if(myrank == 0) {
					cout << "Counting finished " << endl;
					cout << __FUNCTION__ << ": Kmerscount hash includes " << distinctnonerror << " distinct elements" << endl;
					cout << __FUNCTION__ << ": Kmerscount non error kmers count is " << totalnonerror << endl;
					cout << __FUNCTION__ << ": Global max count is " << globalmaxcount << endl;
					cout << __FUNCTION__ << ": Large count histogram is of size " << HIGH_NUM_BINS << endl;
					ADD_DIAG("%lld", "distinct_non_error_kmers", (lld) distinctnonerror);
					ADD_DIAG("%lld", "total_non_error_kmers", (lld) totalnonerror);
					ADD_DIAG("%lld", "global_max_count", (lld) globalmaxcount);
				}
#ifdef HISTOGRAM
				vector<int64_t> hist(COUNT_THRESHOLD,0); 	// zero initialize
				vector<int64_t> hist_high(HIGH_NUM_BINS,0);
#endif

#ifdef HEAVYHITTERS
				if (heavyhitters) {
					double shh = MPI_Wtime();
					MPI_Op ufxreducempiop;
					if (MYSV._my_log && HIPMER_VERBOSITY > 2) {
						for(size_t i=0; i<nfreqs; ++i) {
							Kmer kmer = heavyhitters->Get(i);
							//DBG2("myHeavyHitter: %s %d [%d,%d,%d,%d] [%d,%d,%d,%d]\n", kmer.toString().c_str(), Frequents[i].count, Frequents[i].ACGTleft[0], Frequents[i].ACGTleft[1], Frequents[i].ACGTleft[2], Frequents[i].ACGTleft[3], Frequents[i].ACGTrigh[0], Frequents[i].ACGTrigh[1], Frequents[i].ACGTrigh[2], Frequents[i].ACGTrigh[3]);
							DBG2("myHeavyHitter: %s %d \n", kmer.toString().c_str(), Frequents[i].count);
						}
					}
					CHECK_MPI( MPI_Op_create(MPI_UFXReduce, true, &ufxreducempiop) );    // create commutative mpi-reducer
					CHECK_MPI( MPI_Allreduce(MPI_IN_PLACE, Frequents, nfreqs, MPIType<UFX2ReduceObj>(), ufxreducempiop, MPI_COMM_WORLD) );
					int heavyitems = 0;
					int millioncaught = 0;
					int64_t heavycounts = 0;
#ifdef USE_UPC_ALLOCATOR_IN_MPI
					KmerAllocator::startUPC();
#endif
					for(size_t i=0; i<nfreqs; ++i) {
						Kmer kmer = heavyhitters->Get(i);
						assert( heavyhitters->IsMember(kmer) );
						assert( heavyhitters->FindIndex( kmer ) == i );
						uint64_t myhash = kmer.hash();
						double range = static_cast<double>(myhash) * static_cast<double>(nprocs);
						size_t owner = range / static_cast<double>(numeric_limits<uint64_t>::max());
						if(owner == myrank) {
							bool inserted = false, merged = false;
							auto iter = kmercounts->find( kmer.getArray() );
							if (iter != kmercounts->end()) {
#ifdef KHASH
								KmerCountType &kct = *iter;
#else
								KmerCountType &kct = iter->second;
#endif
								DBG("kmer count for %s exists (HH: %d)\n", kmer.toString().c_str(), Frequents[i].count);
							}
							if(UFX2ReduceObjHasCount(Frequents[i])) {
								if (iter == kmercounts->end()) {
									if (reliable_max && Frequents[i].count <= reliable_max) { // if Frequents[i].count > reliable_max, don't store it at all
										inserted = true;
#ifdef KHASH
										kmercounts->insert(kmer.getArray(), make_(newReadIdList(), newPositionsList(), Frequents[i].count));
#else
										kmercounts->insert(make_pair(kmer.getArray(), make_tuple(newReadIdList(), newPositionsList(), Frequents[i].count)));
#endif
									}
								} else {
									ASSERT( false,"" ); // This should not happen
									merged = true;
#ifdef KHASH
									KmerCountType &kct = *iter;
#else
									KmerCountType &kct = iter->second;
#endif
									//DBG("before %s %d [%d,%d,%d,%d] [%d,%d,%d,%d]\n", kmer.toString().c_str(), get<2>(kct), get<0>(kct)[0], get<0>(kct)[1], get<0>(kct)[2], get<0>(kct)[3], get<1>(kct)[0], get<1>(kct)[1], get<1>(kct)[2], get<1>(kct)[3]);
									DBG("before %s %d \n", kmer.toString().c_str(), get<1>(kct));

									get<2>(kct) += Frequents[i].count;
									/*
									   for(int b = 0; b< 4; b++) {
									   get<0>(kct)[b] += Frequents[i].ACGTleft[b];
									   get<1>(kct)[b] += Frequents[i].ACGTrigh[b];
									   }
									 */
								}

								nonerrorkmers += Frequents[i].count;
								heavycounts += Frequents[i].count;
								if(Frequents[i].count > maxcount)  maxcount = Frequents[i].count;
								if(Frequents[i].count > MILLION)  ++millioncaught;
							}
							//DBG2("HeavyHitter (%s): %s %d [%d,%d,%d,%d] [%d,%d,%d,%d]\n", (inserted|merged) ? (inserted?"insert":"merged") : "notused", kmer.toString().c_str(), Frequents[i].count, Frequents[i].ACGTleft[0], Frequents[i].ACGTleft[1], Frequents[i].ACGTleft[2], Frequents[i].ACGTleft[3], Frequents[i].ACGTrigh[0], Frequents[i].ACGTrigh[1], Frequents[i].ACGTrigh[2], Frequents[i].ACGTrigh[3]);
							DBG2("HeavyHitter (%s): %s %d \n", (inserted|merged) ? (inserted?"insert":"merged") : "notused", kmer.toString().c_str(), Frequents[i].count);
							++hashsize;
							++heavyitems;
						} else { // not the owner
							ASSERT( kmercounts->find( kmer.getArray() ) == kmercounts->end(),"" );
						}
					}
					delete [] Frequents;    // free memory
#ifdef USE_UPC_ALLOCATOR_IN_MPI
					KmerAllocator::endUPC();
#endif
					if (heavyhitters) {
						heavyhitters->Clear();
						delete heavyhitters;
					}
					heavyhitters = NULL;
					int totalheavyitems, totalmillioncaught;
					int64_t totalheavycounts;
					CHECK_MPI( MPI_Reduce(&millioncaught, &totalmillioncaught, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD) );
					CHECK_MPI( MPI_Reduce(&heavyitems, &totalheavyitems, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD) );
					CHECK_MPI( MPI_Reduce(&heavycounts, &totalheavycounts, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD) );
					CHECK_MPI( MPI_Reduce(&nonerrorkmers, &totalnonerror, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD) );
					CHECK_MPI( MPI_Reduce(&hashsize, &distinctnonerror, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD) );
					CHECK_MPI( MPI_Allreduce(&maxcount, &globalmaxcount, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD) );
					double ehh = MPI_Wtime();
					if(myrank == 0) {
						cout << "****** After high freq k-mers are communicated (" << (ehh-shh) << " s)*******" << endl;
						cout << __FUNCTION__ << ": Total heavy hitters " << totalheavyitems << " for a total of " << totalheavycounts << " counts" << endl;
						cout << __FUNCTION__ << ": Averaging " << static_cast<double>(totalheavycounts) / static_cast<double>(totalheavyitems) << " count per item" << endl;
						cout << __FUNCTION__ << ": which caught " << totalmillioncaught << " entries of over million counts" << endl;
						cout << __FUNCTION__ << ": Kmerscount hash includes " << distinctnonerror << " distinct elements" << endl;
						cout << __FUNCTION__ << ": Kmerscount non error kmers count is " << totalnonerror << endl;
						cout << __FUNCTION__ << ": Global max count is " << globalmaxcount << endl;
						ADD_DIAG("%d", "total_heavy_hitters", totalheavyitems);
						ADD_DIAG("%lld", "total_heavy_counts", (lld) totalheavycounts);
						ADD_DIAG("%lld", "distinct_non_error_kmers", (lld) distinctnonerror);
						ADD_DIAG("%lld", "total_non_error_kmers", (lld) totalnonerror);
						ADD_DIAG("%lld", "global_max_count", (lld) globalmaxcount);
					}
				} else {
					SLOG("Skipping heavy hitter counts\n");
				}
#endif // HEAVYHITTERS

				if (globalmaxcount == 0) {
					SDIE("There were no kmers found, perhaps your KMER_LENGTH (%d) is longer than your reads?", KMER_LENGTH);
				}

				nonerrorkmers = 0;	// reset
				distinctnonerror = 0;
				int64_t overonecount = 0;
				auto itr = kmercounts->begin();
				while(itr != kmercounts->end()) {
#ifdef KHASH
					if(!itr.isfilled()) { ++itr; continue; }
					int allcount = get<2>(itr.value());
#else
					int allcount =  get<2>(itr->second);
#endif
#ifdef HISTOGRAM
					if(allcount <= 0)
						++(hist[0]);
					else if(allcount <= COUNT_THRESHOLD)
						++(hist[allcount-1]);
					else if(allcount <= COUNT_THRESHOLD_HIGH)
						++(hist_high[(allcount - COUNT_THRESHOLD-1)/HIGH_BIN]);
					else
						++(hist_high[HIGH_NUM_BINS-1]);
#endif // HISTOGRAM
					if(allcount < ERR_THRESHOLD || (reliable_max > 0 && allcount > reliable_max) ) {
						--hashsize;
#ifdef KHASH
						auto newitr = itr;
						++newitr;
						kmercounts->erase(itr);	// amortized constant
						// Iterators, pointers and references referring to elements removed by the function are invalidated.
						// All other iterators, pointers and references keep their validity.
#else
						// C++11 style erase returns next iterator after erased entry
						itr = kmercounts->erase(itr);	// amortized constant
#endif
					} else {
						nonerrorkmers += allcount;
						distinctnonerror++;
						++itr;
					}
					if(allcount > 1) {
						overonecount += allcount;
					}
				}
#ifdef HISTOGRAM
				hist[0] = kmersprocessed - overonecount;	// over-ride the value for 1, which wasn't correct anyway
#endif

				CHECK_MPI( MPI_Reduce(&nonerrorkmers, &totalnonerror, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD) );
				CHECK_MPI( MPI_Reduce(&hashsize, &distinctnonerror, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD) );
				if(myrank == 0) {
					cout << __FUNCTION__ << ": Erroneous count < " << ERR_THRESHOLD  << " and high frequency > "<< reliable_max <<" cases removed " << endl;
					cout << __FUNCTION__ << ": Kmerscount hash includes " << distinctnonerror << " distinct elements" << endl;
					cout << __FUNCTION__ << ": Kmerscount non error kmers count is " << totalnonerror << endl;
					ADD_DIAG("%lld", "distinct_non_error_kmers", (lld) distinctnonerror);
					ADD_DIAG("%lld", "total_non_error_kmers", (lld) totalnonerror);
					ADD_DIAG("%lld", "global_max_count", (lld) globalmaxcount);
				}
#ifdef HISTOGRAM
				assert( hist.size() == COUNT_THRESHOLD );
				assert( ((int64_t*) hist.data()) + COUNT_THRESHOLD - 1 == &(hist.back()) );
				assert( hist_high.size() == HIGH_NUM_BINS );
				assert( ((int64_t*) hist_high.data()) + HIGH_NUM_BINS - 1 == &(hist_high.back()) );
				if(myrank == 0) {
					CHECK_MPI( MPI_Reduce(MPI_IN_PLACE, hist.data(), COUNT_THRESHOLD, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD) );
					CHECK_MPI( MPI_Reduce(MPI_IN_PLACE, hist_high.data(), HIGH_NUM_BINS, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD) );

					stringstream ss, ss1,ss2;
					ss << KMER_LENGTH;
					string hname = "histogram_k";
					hname += ss.str();
					string hnamehigh = hname + "_beyond";
					ss1 << COUNT_THRESHOLD;
					hnamehigh += ss1.str();
					hnamehigh += "_binsize";
					ss2 << HIGH_BIN;
					hnamehigh += ss2.str();
					hname += ".txt";
					hnamehigh += ".txt";
					hname = getRankPath(hname.c_str(), -1);
					hnamehigh = getRankPath(hnamehigh.c_str(), -1);
					ofstream hout(hname.c_str());
					int bin = 0;
					for(auto it = hist.begin(); it != hist.end(); it++) {
						hout << ++bin << ' ' << *it << "\n";
					}

					ofstream hhigh(hnamehigh.c_str());
					bin = COUNT_THRESHOLD;
					for(auto it = hist_high.begin(); it != hist_high.end(); it++) {
						hout << bin << "' '" << *it << "\n";
						bin += HIGH_BIN;
					}
				} else {
					CHECK_MPI( MPI_Reduce(hist.data(), NULL, COUNT_THRESHOLD, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD) );	// receive buffer not significant at non-root
					CHECK_MPI( MPI_Reduce(hist_high.data(), NULL, HIGH_NUM_BINS, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD) );
				}
				if (myrank == 0) cout << "Generated histograms" << endl;
#endif
				MPI_Pcontrol(-1,"HashClean");
			}

			int kmermatch_main(int argc, char ** argv)
			{
				CHECK_MPI( MPI_Comm_size(MPI_COMM_WORLD,&nprocs) );
				CHECK_MPI( MPI_Comm_rank(MPI_COMM_WORLD,&myrank) );

				double time_start = MPI_Wtime(); // for total elapsed time in kmer counting and overlap
				double time_temp,
				       time_cardinality_est,
				       time_first_data_pass,
				       time_second_data_pass,
				       time_computing_loadimbalance,
				       time_total_readoverlap,
				       time_kmercounting_output,
				       time_sanity_output,
				       time_total_elapsed,
				       time_global_total;

				KMER_LENGTH = MAX_KMER_SIZE-1;
				KMER_PACKED_LENGTH = (MAX_KMER_SIZE-1+3)/4;

				nonerrorkmers = 0;
				totaltime = 0;
				kmersprocessed = 0;
				readsprocessed = 0;

				bool opt_err = false;
				char *input_fofn = NULL;

				/*
				 * fastq input and kmer output caching flag
				 */
				bool cached_io = false;
				/*
				 * cache output overlaps in virtual memory
				 */
				bool per_thread_overlaps = false;
				bool cached_overlaps = false;
				string prefix = "";
				const char* base_dir = ".";
				bool use_hll = false;
				bool save_ufx = false;
				char* overlap_filename = NULL;
				int min_seed_distance = KMER_LENGTH;
				int max_num_seeds = -1;

				option_t *this_opt;
				option_t *opt_list = GetOptList(argc, argv, "i:e:u:p:BHESaP:k:x:y:m:q:d:t:w:n:h:f:M:");
				print_args(opt_list, __func__);

				while (opt_list) {
					this_opt = opt_list;
					opt_list = opt_list->next;
					switch (this_opt->option) {
						case 'i':
							input_fofn = strdup(this_opt->argument);
							if (input_fofn == NULL) { SDIE("Please enter a valid input file name after -i. (none entered)"); }
							break;
						case 'm':
							overlap_filename = strdup(this_opt->argument);
							if (overlap_filename == NULL) { SDIE("Please enter a valid output file name after -m. (none entered)"); }
							break;
						case 'e':
							ERR_THRESHOLD = strtol(this_opt->argument, NULL, 10);
							if (ERR_THRESHOLD < 0) ERR_THRESHOLD = 0;
							break;
						case 'k':
							KMER_LENGTH = strtol(this_opt->argument, NULL, 10);
							if (KMER_LENGTH >= MAX_KMER_SIZE) { SDIE("Kmer length (%d) must be < %d for this binary.  Please recompile with a larger MAX_KMER_SIZE", KMER_LENGTH, MAX_KMER_SIZE); }
							break;
						case 'p':
							prefix = this_opt->argument;
							break;
						case 'B':
							base_dir = "/dev/shm";
							cached_io = 1;
							break;
						case 'a':
							cached_overlaps = true;
							break;
						case 'P':
							per_thread_overlaps = true;
							break;
						case 'H':
							use_hll = true;
							break;
						case 'S':
							save_ufx = true;
							break;
						case 't':
							type = strtol(this_opt->argument, NULL, 10);
							break;
						case 'w':
							window = strtol(this_opt->argument, NULL, 10);
							break;
						case 'n':
							MINIMIZER_LENGTH = strtol(this_opt->argument, NULL, 10);
							break;
						case 'x':
							reliable_min = strtol(this_opt->argument, NULL, 10);
							break;
						case 'u':
							reliable_max = strtol(this_opt->argument, NULL, 10);
							if (reliable_max <= ERR_THRESHOLD) { SDIE("Upper limit set by -h must be greater than the error threshold set by -e.", reliable_max, ERR_THRESHOLD); }
							if (reliable_max < 2) { SDIE("Upper limit set by -h must be greater than 1, -h= %d", reliable_max); }
							if (reliable_max > MAX_NUM_READS) { SDIE("Upper limit set by -h is larger than MAX_NUM_READS, -h= %d. Use compilation with MAX_NUM_READS > %d", reliable_max, MAX_NUM_READS); }
							break;
						case 'd': {
								  min_seed_distance = strtol(this_opt->argument, NULL, 10);
								  if (min_seed_distance < 0) {
									  SDIE("The minimum seed distance must be greater than or equal to 0 (-d %s).", this_opt->argument);
								  }
								  break;
							  }
						case 'q': {
								  max_num_seeds = strtol(this_opt->argument, NULL, 10);
								  // q=-1 will use all seeds
								  // q=0 won't use any
								  if (max_num_seeds < 1) {
									  SWARN("The maximum number of seeds to extend has been set to %d. If this was not intentional, rerun with -q set to a positive value.\n", max_num_seeds);
								  }
								  break;
							  }
                        case 'h':
                            minimizer_ordering = 1;
                            uhs_file_path = this_opt->argument;
                            break;
                        case 'f':
                        	uhs_sample_fraction = strtof(this_opt->argument, NULL);
                        	break;
                        case 'M':
                        	uhs_mpi_mode = strtol(this_opt->argument, NULL, 10);
                        	break;
						default:
							opt_err = true;
					}
				}

                printf("~~~~ Hello from MPI rank %d, line 1988 ~~~~\n", myrank);

				if (opt_err || !input_fofn)
				{
					if(myrank  == 0)
					{
						cout << "Usage: ./kmermatch -i filename -e error_threshold -u reliable_upper_limit <-p prefix> <-m overlap_file_name> <-s> <-H> <-a>" << endl;
						cout << "'filename' is a text file containing the paths to each file to be counted - one on each line -" << endl;
						cout << "'error_threshold' is the lowest number of occurrences of a k-mer for it to be considered a real (non-error) k-mer" << endl;
						cout << "reliable_upper_limit sets the maximum number of reads that will be recorded per k-mer" << endl;
						cout << "'prefix' is optional and if set, the code prints all k-mers with the prefix" << endl;
						cout << "'overlap_file_name' (optional) the name of the file to which output overlaps will be written." << endl;
						cout << "-s is optional and if set, shared memory will be used to cache all files" << endl;
						cout << "-H is optional and if set, then HyperLogLog will be used to estimate the required bloomfilter" << endl;
						cout << "-P (optional) if set, overlaps will be written to per thread output files" << endl;
						cout << "-a (optional) if set, overlaps will be written to virtual memory for the next stage" << endl;
					}
					return 0;
				}

				if (overlap_filename == NULL) {
					size_t str_length = strlen(OVERLAP_OUT_FNAME);
					overlap_filename = (char*) malloc(str_length);
					memset(overlap_filename, '\0', str_length);
					sprintf(overlap_filename, OVERLAP_OUT_FNAME, str_length);
				}
				if(myrank  == 0)
				{
					cout << type << "WTH " << endl;
					cout << endl << "************ k-mer analysis (supports CPU and NVIDIA GPU platforms) ************" << endl << endl;
					switch(type){
						case 0: cout << "\tRunning kmer based kmer-counter on CPU" << endl; break;
						case 1: cout << "\tRunning kmer based kmer-counter on NVIDIA GPU" << endl; break;
						case 2: cout << "\tRunning supermer based kmer-counter on CPU" << endl; break;
						case 3: cout << "\tRunning supermer based kmer-counter on NVIDIA-GPU" << endl; break;
					}
					cout << "\tInput fastq file list: " << input_fofn <<endl;
					cout << "\tK-mer length = " << KMER_LENGTH << endl;
					cout << "\tMax k-mer length (internal) = " << MAX_KMER_SIZE << endl;
					if(type == 2 || type ==3){
						cout << "\tMINIMIZER length = " << MINIMIZER_LENGTH << endl << endl;
					}
					// cout << "\tMax read-records per k-mer = " << reliable_max << endl;
					// cout << "\tMax representable read length = " << sizeof(PosInRead) << " bytes" << endl;
					// cout << "\tMax number of seeds to retain per candidate = " << max_num_seeds << endl;
					// cout << "\tMinimum distance between retained seeds = " << min_seed_distance << endl;
				}

				time_temp = MPI_Wtime(); // start cardinality estimate timer
				vector<filedata> allfiles = GetFiles(input_fofn);
				Kmer::set_k(KMER_LENGTH);
				double cardinality;
				unsigned int estReadsPerProc = 0;
				if (use_hll) {
					heavyhitters = new HeavyHitters(MAXHITTERS);
					ProudlyParallelCardinalityEstimate(allfiles, cardinality, cached_io, base_dir); // doesn't update kmersprocessed yet (but updates readsprocessed)
					estReadsPerProc = readsprocessed / nprocs;
				} else {
					// just estimate the cardinality using fastq size and sampling.
					int64_t mySums[3] = {0,0,0};
					int64_t &myCardinality = mySums[0];
					int64_t &myTotalReads = mySums[1];
					int64_t &myTotalBases = mySums[2];
					for(int i = 0; i < allfiles.size(); i++) {
						if (MurmurHash3_x64_64((char*) &i, sizeof(int)) % nprocs == myrank) {
							int64_t totalReads, totalBases;
							int64_t fileSize = estimate_fq(allfiles[i].filename, 5000, &totalReads, &totalBases);
							myTotalReads += totalReads;
							myTotalBases += totalBases;
							ASSERT(fileSize == allfiles[i].filesize,"");
							if (totalReads > 0) {
								int64_t kmersPerRead = ((totalBases+totalReads-1) / totalReads) - KMER_LENGTH + 1;
								myCardinality += kmersPerRead * totalReads;
								cout << "Cardinality for " << allfiles[i].filename << ": " << myCardinality << endl;
							}
						}
					}
					MPI_Allreduce(MPI_IN_PLACE, mySums, 3, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
					if (myrank == 0) {
						cout << __FUNCTION__ << ": Estimated our cardinality: " << myCardinality << " totalReads: " << myTotalReads << " totalBases: " << myTotalBases << endl;
					}
					// baseline of 10M kmers
					if (myCardinality < 10000000) myCardinality = 10000000;
					// assume maximum of 90% of the kmers are unique, because at least some have to repeat...
					cardinality = 0.9 * myCardinality / (double) nprocs;
					estReadsPerProc = myTotalReads / nprocs;
				}
#ifndef HIPMER_BLOOM64
				if (cardinality > 1L<<31) {
					cout << __FUNCTION__ << ": Reduced cardinality to fit within int32_t" << endl;
					cardinality = 1L<<31 - 1L;
				}
#endif
				time_cardinality_est = MPI_Wtime() - time_temp;
				// TODO HLL time should be separated from HLL+HH time in this report
				serial_printf("%s: Pre 1st pass cardinality estimate, elapsed time: %0.3f s\n", __FUNCTION__, time_cardinality_est);
				time_temp = MPI_Wtime();

				//
				// Prepare for first pass over input data with k-mer extraction
				//
				ASSERT(kmercounts == NULL,"");
				kmercounts = new KmerCountsType();
				uint64_t myReserve = cardinality * 0.1 + 64000; // assume we have at least 10x depth of kmers to store and a minimum 64k
				if (myReserve >= 4294967291ul) myReserve = 4294967290ul; // maximum supported by khash
#if UPC_ALLOCATOR
				SLOG("Using upc-shared memory for kmercounts and bloom\n");
#endif
#ifdef KHASH
				LOGF("Reserving %lld entries in KHASH for cardinality %lld\n", (lld) myReserve, (lld) cardinality);
				kmercounts->reserve(myReserve);
#else
				LOGF("Reserving %lld entries in VectorMap for cardinality %lld\n", (lld) myReserve, (lld) cardinality);
				kmercounts->reserve(myReserve);
				rsrv = myReserve;
#endif
				DBG("Reserved kmercounts\n");

				// initialize readNameMap for storing ReadID -> names/tags of reads
				std::unordered_map<ReadId, std::string>* readNameMap = new std::unordered_map<ReadId, std::string>();

				// pass 1
				ReadId myReadStartIndex = 0;
				assert(base_dir != NULL);
				int nReads = ProcessFiles(allfiles, 1, cardinality, cached_io, base_dir, myReadStartIndex, *readNameMap);	// determine final hash-table entries using bloom filter
				DBG("my nreads=%lld\n", nReads);

				time_first_data_pass = MPI_Wtime() - time_temp;
				serial_printf("%s: 1st input data pass, elapsed time: %0.3f s\n", __FUNCTION__, time_first_data_pass);
				time_temp = MPI_Wtime();

				//
				// Prepare for second pass over the input data, extracting k-mers with read ID's, names, etc.
				//

				// exchange number of reads-per-processor to calculate read indices
				uint64_t sndReadCounts[nprocs];
				for (int i = 0; i < nprocs; i++) { sndReadCounts[i] = nReads; }
				uint64_t recvReadCounts[nprocs];
				CHECK_MPI( MPI_Alltoall(sndReadCounts, 1, MPI_UINT64_T, recvReadCounts, 1, MPI_UINT64_T, MPI_COMM_WORLD) );
				myReadStartIndex = 1;
				for (int i = 0; i < myrank; i++) {
					myReadStartIndex += recvReadCounts[i];
				}
#ifdef DEBUG
				if(myrank == nprocs-1) {
					for (int i = 0; i < nprocs; i++) {
						LOGF("recvReadCounts[%lld]=%lld\n", recvReadCounts[i]);
					}
				}
#endif
				DBG("my startReadIndex=%lld\n", myReadStartIndex);

				CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );

				// initialize end read ranges
				ReadId readRanges[nprocs];
				ReadId last = 0;
				for (int i = 0; i < nprocs; i++) {
					readRanges[i] = last + recvReadCounts[i];
					last += recvReadCounts[i];
				}
				DBG("My read range is [%lld - %lld]\n", (myrank==0? 1 : readRanges[myrank-1]+1), readRanges[myrank]);
				//////////////////////////

				exit(0);//
				// return 0;

				// perform pass 2
				ProcessFiles(allfiles, 2, cardinality, cached_io, base_dir, myReadStartIndex, *readNameMap);
				time_second_data_pass = MPI_Wtime() - time_temp;
				serial_printf("%s: 2nd input data pass, elapsed time: %0.3f s\n", __FUNCTION__, time_second_data_pass);
				time_temp = MPI_Wtime();

				int64_t sendbuf = kmercounts->size();
				int64_t recvbuf, totcount, maxcount;
				CHECK_MPI( MPI_Exscan(&sendbuf, &recvbuf, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD) );
				CHECK_MPI( MPI_Allreduce(&sendbuf, &totcount, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD) );
				CHECK_MPI( MPI_Allreduce(&sendbuf, &maxcount, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD) );
				int64_t totkmersprocessed;
				CHECK_MPI( MPI_Reduce(&kmersprocessed, &totkmersprocessed, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD) );
				time_computing_loadimbalance = MPI_Wtime() - time_temp;
				if(myrank  == 0) {
					cout << __FUNCTION__ << ": Total number of stored k-mers: " << totcount << endl;
					double imbalance = static_cast<double>(nprocs * maxcount) / static_cast<double>(totcount);
					cout << __FUNCTION__ << ": Load imbalance for final k-mer counts: " << imbalance << endl;
					cout << __FUNCTION__ << ": CardinalityEstimate " << static_cast<double>(totkmersprocessed) / (MEGA * max((time_first_data_pass),0.001) * nprocs) << " MEGA k-mers per sec/proc in " << (time_first_data_pass) << " seconds"<< endl;
					cout << __FUNCTION__ << ": Bloom filter + hash pre-allocation  " << static_cast<double>(totkmersprocessed) / (MEGA * max((time_second_data_pass),0.001) * nprocs) << " MEGA k-mers per sec/proc in " << (time_second_data_pass) << " seconds" << endl;
					cout << __FUNCTION__ << ": Actual hash-based counting  " << static_cast<double>(totkmersprocessed) / (MEGA * max((time_computing_loadimbalance),0.001) * nprocs) << " MEGA k-mers per sec/proc in " << (time_computing_loadimbalance) << " seconds" << endl;
				}
				serial_printf("%s: Total time computing load imbalance: %0.3f s\n", __FUNCTION__, time_computing_loadimbalance);
				CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
				time_temp =  MPI_Wtime();

				//
				// Begin output of other data
				//
				vector<filedata>().swap(allfiles);
				string countfile = string(input_fofn)+".ufx.bin";
				string intermediate_countfile = "interm_"+countfile; // string("intermediates/") + countfile; // intermediates/ was a directory created by previous stages of hipmer and not created in stand-alone ufx

#ifndef BENCHMARKONLY
				if(myrank  == 0) cout << "Writing k-mer data to binary via MPI-IO" << endl;
				MPI_File thefile;
				int64_t lengthuntil;

				if (save_ufx || !cached_io) {
					if(myrank == 0) MPI_File_delete((char*) intermediate_countfile.c_str(), MPI_INFO_NULL); // Unlink first (if it exists do not check error status)
					CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
					CHECK_MPI( MPI_File_open(MPI_COMM_WORLD, (char*) intermediate_countfile.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &thefile) );
					if(myrank == 0)
						lengthuntil = 0;
					else
						lengthuntil = recvbuf;
				}

#ifdef DEBUG
				char mergraphfilepath[MAX_FILE_PATH];
				snprintf(mergraphfilepath, MAX_FILE_PATH, "%s.mergraph.%d", countfile.c_str(), myrank);
				char *mergraphfile = get_rank_path(mergraphfilepath, myrank);
				ofstream mergraph(mergraphfile);
#endif // DEBUG

				uint64_t ufxcount = kmercounts->size();
				ufxpack * packed = new ufxpack[ufxcount];
				memset(packed, 0, ufxcount * sizeof(ufxpack));

				uint64_t packedIdx = 0;
				for(auto itr = kmercounts->begin(); itr != kmercounts->end(); ++itr)
				{
#ifdef KHASH
					if(!itr.isfilled()) { continue; }
					auto key = itr.key();
					auto val = itr.value();
#else
					auto key = itr->first;
					auto val = itr->second;;
#endif

					Kmer kmer(key);
					ufxpack upack;
					upack.arr = kmer.getArray();
					PackIntoUFX(get<0>(val), get<1>(val), get<2>(val), upack);
					CHECK_BOUNDS(packedIdx, ufxcount);
					packed[packedIdx++] = upack;
				}
				ASSERT(packedIdx == ufxcount,"");

				// explicitly free / destruct kmercounts memory now
				//    SLOG("Deallocating kmercount memory: %lld\n", (lld) kmercounts->size());
				//    long totMem = sizeof(kmercounts) + kmercounts->size() * (sizeof(MERARR) + sizeof(KmerCountType));
				//    WARN("Deallocating kmercount memory: %lld (%ld %.2f M)\n", (lld) kmercounts->size(),
				//         totMem, (double)totMem / 1024 / 1024);
				DBG("Deallocating kmercount memory (%lld entries)\n", (lld) kmercounts->size());
#ifdef KHASH
				kmercounts->free();
#else
#ifdef USE_UPC_ALLOCATOR_IN_MPI
				KmerAllocator::startUPC();
#endif
				kmercounts->clear();
#ifdef USE_UPC_ALLOCATOR_IN_MPI
				KmerAllocator::endUPC();
#endif
#endif
				delete kmercounts;
				kmercounts = NULL;
				DBG2("Deallocated kmercount memory\n");

				if (save_ufx || !cached_io) {
					// write per thread files of ids -> names // TODO optimize
					char* outfilename = "idtags.txt";
					char fname[MAX_FILE_PATH];
					memset(fname, 0, MAX_FILE_PATH*sizeof(char));
					sprintf(fname, VMFILE_PATH);
					strcat(fname, outfilename);
					get_rank_path(fname, myrank);
					ofstream myoutputfile;
					myoutputfile.open (fname, std::ofstream::out | std::ofstream::trunc);
					if (!myoutputfile.is_open()) {
						DIE("Could not open %s: %s\n", fname, strerror(errno));
						return -1;
					}
					DBG("Successfully opened file %s\n", fname);
					for(auto map_itr = readNameMap->begin(); map_itr != readNameMap->end(); map_itr++) {
						myoutputfile << map_itr->first << " " << map_itr->second << "\n";
					}
					myoutputfile << flush;
					myoutputfile.close();

					// write a single global file
					MPI_Datatype datatype;
					CHECK_MPI( MPI_Type_contiguous(sizeof(ufxpack), MPI_UNSIGNED_CHAR, &datatype ) );
					CHECK_MPI( MPI_Type_commit(&datatype) );
					int dsize;
					CHECK_MPI( MPI_Type_size(datatype, &dsize) );
					ASSERT( dsize == sizeof(ufxpack),"" );

					int mpi_err = MPI_File_set_view(thefile, lengthuntil * dsize, datatype, datatype, (char*)"external32", MPI_INFO_NULL);
					if (mpi_err == 51) {
						// external32 datarep is not supported, use native instead
						CHECK_MPI( MPI_File_set_view(thefile, lengthuntil * dsize, datatype, datatype, (char*)"native", MPI_INFO_NULL) );
					} else {
						CHECK_MPI(mpi_err);
					}

					// write in batches smaller than 2GB per node, assuming maximum of 256 cores in a node
					uint64_t coresPerNode = MYSV.cores_per_node;
					assert(dsize * coresPerNode > 0);
					uint64_t maxBatch = (1ULL<<31) / (dsize * coresPerNode);
					assert(maxBatch > 0); //debugging
					uint64_t idx = 0;
					int num_ranks;
					CHECK_MPI( MPI_Comm_size(MPI_COMM_WORLD, &num_ranks) );
					assert( (num_ranks + maxBatch - 1) > 0); //debugging
					int numBatches = ( (totcount+num_ranks-1) / num_ranks  + maxBatch - 1) / maxBatch;
					assert (numBatches >= 0);
					SLOG("Writing output in %d batches of about %lld per rank\n", numBatches, (numBatches==0? 0 : (lld) totcount / numBatches / num_ranks));
					MPI_Status status;
					for (int batch = 0; batch < numBatches; batch++ ) {
						uint64_t size = (ufxcount + numBatches - 1) / numBatches;
						if (idx+size > ufxcount) size = ufxcount - idx;
						LOGF("Writing %lld of %lld from idx %lld\n", (lld) size, (lld) ufxcount, (lld) idx);
						CHECK_MPI( MPI_File_write_all(thefile, packed+idx, size, datatype, &status) );
						int count;
						CHECK_MPI( MPI_Get_count(&status, datatype, &count) );
						assert( count == size );
						idx += size;
					}
					assert(idx == ufxcount);
					CHECK_MPI( MPI_File_close(&thefile) );
					CHECK_MPI( MPI_Type_free(&datatype) );
					if (myrank == 0)
						link_chk(intermediate_countfile.c_str(), countfile.c_str());
				}
				if (cached_io) {
					stringstream ss;
					string rank;
					ss << myrank;
					ss >> rank;
					unsigned found = countfile.find_last_of("/");
					countfile = countfile.substr(found+1);
					countfile += rank;
					int64_t ufxcount64bit = static_cast<int64_t>(ufxcount);
					int64_t bytes2write =  ufxcount64bit * static_cast<int64_t>(sizeof(ufxpack));

					char fname[MAX_FILE_PATH];
					memset(fname, 0, MAX_FILE_PATH*sizeof(char));
					strcat(fname, VMFILE_PATH);
					strcat(fname, countfile.c_str());
					strcat(fname, GZIP_EXT); // cached file will be compressed
					get_rank_path(fname, myrank);

					GZIP_FILE UFX_f = GZIP_OPEN(fname, "wb");
					int64_t bytesWritten = 0;
					int64_t maxWrite = 16777216;
					// write in batches smaller than 16MB to prevent zlib overflows and too much memory consumed
					while (bytes2write > bytesWritten) {
						int64_t remaining = bytes2write - bytesWritten;
						assert(remaining > 0);
						if (remaining > maxWrite) remaining = maxWrite;
						int64_t wroteBytes = GZIP_FWRITE(((char*)packed) + bytesWritten, 1, remaining, UFX_f);
						if (wroteBytes != remaining) DIE("Could not write %lld bytes (out of %lld) to UFX file %s... Write %lld\n", (lld) remaining, (lld) bytes2write, fname, (lld) wroteBytes);
						bytesWritten += wroteBytes;
					}
					if (bytesWritten != bytes2write)
						DIE("Could not write the full number of bytes to file %s: expected %lld, got %lld\n", fname, (lld) bytes2write, (lld) bytesWritten);
					GZIP_CLOSE(UFX_f);

					// now write the total size to file - because we cannot use the mpi functions in
					// readufx.cpp which is used by numerous upc stages to read in the ufx binary data
					int64_t totalufxcount;
					CHECK_MPI(MPI_Allreduce(&ufxcount64bit, &totalufxcount, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD));
					strcat(fname, ".entries");
					FILE *f = fopen_chk(fname, "w");
					fprintf(f, "%lld\t%lld\n", (lld) totalufxcount, (lld) ufxcount64bit);
					fclose(f);
				}

				// write each processor's starting ReadIds
				/*
				   if(myrank==0) {
				   char fname[MAX_FILE_PATH] = INIT_READIDS_FNAME;
				   FILE* readIdFile = fopen_chk(fname, "w");
				   for (int i = 0; i < nprocs; i++ ) {
				   fprintf(readIdFile, "%lld ", (lld) readRanges[i]);
				   }
				   fprintf(readIdFile, "\n");
				   fclose(readIdFile);
				   }*/

				time_kmercounting_output = MPI_Wtime() - time_temp;
				serial_printf("%s: elapsed time over kmercounting output: %0.3f s\n", __FUNCTION__, time_kmercounting_output);
				if(myrank == 0)	cout << "File write completed\n";
				time_temp = MPI_Wtime();

				if (!cached_io)
				{
					if(prefix != "")
					{
						string sanity = "ufx_";
						stringstream ss;
						ss << myrank;
						sanity += ss.str();
						sanity += string(".");
						sanity += prefix;
						ofstream outsanity(sanity.c_str());

						int prelen = prefix.length();
						for(int i=0; i < ufxcount; ++i)
						{
							Kmer kmer(packed[i].arr);
							string kstr = kmer.toString();
							if(kstr.substr(0, prelen) == prefix)
							{
								outsanity << kstr << "\t" << packed[i].count << "\t";
								for (int r = 0; r < reliable_max; r++) {
									outsanity << packed[i].reads[r] << " ";
								}
								outsanity << endl;
							}
						}
						outsanity.close();
					}

					if(myrank == 0)
					{
						cout << "Finished writing, here is the top of processor 0's data" << endl;
						for(int i=0; i< 10 && i < ufxcount; ++i)
						{
							Kmer kmer(packed[i].arr);
							cout << kmer.toString() << "\t" << packed[i].count << "\t";
							for (int r = 0; r < reliable_max; r++) {
								cout << packed[i].reads[r] << " ";
							}
							cout << "\t";
							for (int r = 0; r < reliable_max; r++) {
								cout << packed[i].positions[r] << " ";
							}
							cout << endl;
						}

#ifdef DEBUG__DISABLED
						cout << "Here is a random sample from the global data" << endl;
#if GCC_VERSION < 40300
						auto generator = tr1::mt19937();
						tr1::uniform_int<int64_t> distribution(0,count-1);
						generator.seed((unsigned int)time(NULL));
#else
						auto generator = mt19937();
						uniform_int_distribution<int64_t> distribution(0,count-1);
						generator.seed((unsigned int)time(NULL));
#endif

						FILE * f = fopen_chk(countfile.c_str(), "r");
						if(!f)
						{
							cerr << "Problem reading binary input file\n";
							return 1;
						}
						for(int i=0; i< 10; ++i)
						{
							int64_t randindex  = distribution(generator);
							ufxpack upack;
							fseek (f, randindex * static_cast<int64_t>(dsize), SEEK_SET );
							fread(&upack, dsize, 1, f);
							Kmer kmer(upack.arr);
							cout << "LOC " << randindex << ":\t" << kmer.toString() << "\t\t";
						}
#endif // #ifdef DEBUG__DISABLED
					}
				}// !cached_io
				delete [] packed;
#endif // #ifndef BENCHMARKONLY

				delete readNameMap;

				time_sanity_output = MPI_Wtime() - time_temp;
				serial_printf("%s: Time elapsed outputting sanity-checking data: %0.3f s\n", __FUNCTION__, time_sanity_output);

				time_total_elapsed = MPI_Wtime() - time_start;
				CHECK_MPI( MPI_Allreduce(&time_total_elapsed, &time_global_total, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD) );
				serial_printf("%s: Overall time for %s is %.2f s\n", basename(argv[0]), __FUNCTION__, time_global_total);

				return 0;
			}


#ifndef SINGLE_EXEC

			int main(int argc, char **argv)
			{

				CHECK_MPI( MPI_Init(&argc, &argv) );
				OPEN_MY_LOG("kmermatch");
				int ret = kmermatch_main(argc, argv);
				MPI_Finalize();
				return ret;
			}

#endif
