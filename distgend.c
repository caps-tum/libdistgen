/**
 * Copyright 2016 by LRR-TUM
 * Jens Breitbart     <j.breitbart@tum.de>
 * Josef Weidendorfer <weidendo@in.tum.de>
 *
 * Licensed under GNU General Public License 2.0 or later.
 * Some rights reserved. See LICENSE
 */

#include "distgend.h"
#include "distgen_internal.h"

#define _GNU_SOURCE
//#define __USE_GNU

#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sched.h>
#include <assert.h>

#include <omp.h>

// TODO We currently allocate the buffers once, should we change this?

// GByte/s measured for i cores is stored in [i-1]
static double distgen_mem_bw_results[DISTGEN_MAXTHREADS];

// the configuration of the system
static distgend_initT system_config;

// Prototypes
static void set_affinity(distgend_initT init);
static void check_affinity(distgend_initT init);
static double bench(distgend_configT config);

void distgend_init(distgend_initT init) {
	assert(init.number_of_threads < DISTGEN_MAXTHREADS);
	assert(init.NUMA_domains < init.number_of_threads);
	assert((init.number_of_threads % init.NUMA_domains) == 0);
	assert(init.number_of_threads % (init.NUMA_domains * init.SMT_factor) == 0);

	system_config = init;

	// we currently measure maximum read bandwidth
	pseudoRandom = 0;
	depChain = 0;
	doWrite = 0;

	// number of iterations. currently a magic number
	iter = 1000;

	// set a size of 50 MB
	// TODO we should compute this based on L3 size
	addDist(50000000);

	// set the number of threads to the maximum available in the system
	tcount = init.number_of_threads;
	omp_set_num_threads(tcount);

	set_affinity(init);
	check_affinity(init);

	initBufs();

	// fill distgen_mem_bw_results
	distgend_configT config;
	for (unsigned char i = 0; i < init.number_of_threads / init.NUMA_domains; ++i) {
		config.number_of_threads = i + 1;
		config.threads_to_use[i] = i;

		distgen_mem_bw_results[i] = bench(config);

		check_affinity(init);
	}
}

double distgend_get_max_bandwidth(distgend_configT config) {
	assert(config.number_of_threads > 0);

	const size_t phys_cores_per_numa =
		system_config.number_of_threads / (system_config.NUMA_domains * system_config.SMT_factor);

	double res = 0.0;

	size_t cores_per_numa_domain[DISTGEN_MAXTHREADS];
	for (size_t i = 0; i < system_config.NUMA_domains; ++i) cores_per_numa_domain[i] = 0;

	// for every NUMA domain we use
	// -> count the cores used
	// TODO how to handle multiple HTs for one core?
	for (size_t i = 0; i < config.number_of_threads; ++i) {
		size_t t = config.threads_to_use[i];
		size_t n = (t / phys_cores_per_numa) % system_config.NUMA_domains;
		++cores_per_numa_domain[n];
	}

	for (size_t i = 0; i < system_config.NUMA_domains; ++i) {
		size_t temp = cores_per_numa_domain[i];
		if (temp > 0) res += distgen_mem_bw_results[temp - 1];
	}

	return res;
}

double distgend_is_membound(distgend_configT config) {
	check_affinity(system_config);

	// run benchmark on given cores
	// compare the result with distgend_get_max_bandwidth();
	const double m = bench(config);
	const double c = distgend_get_max_bandwidth(config);
	const double res = m / c;

	return (res > 1.0) ? 1.0 : res;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// INTERNAL FUNCTIONS
//////////////////////////////////////////////////////////////////////////////////////////////////

static double bench(distgend_configT config) {
	u64 aCount = 0;
	const double t1 = wtime();

#pragma omp parallel reduction(+ : aCount)
	{
		size_t tid = (size_t)omp_get_thread_num();
		for (size_t i = 0; i < config.number_of_threads; ++i) {
			if (tid == config.threads_to_use[i]) {
				// pid_t tid = (pid_t)syscall(SYS_gettid);
				// #pragma omp critical
				//{ printf("I'm thread %d, %2d, on core %d.\n", tid, omp_get_thread_num(), sched_getcpu()); }
				double tsum = 0.0;
				u64 taCount = 0;

				runBench(buffer[omp_get_thread_num()], iter, depChain, doWrite, &tsum, &taCount);

				aCount += taCount;
			}
		}
	}

	const double t2 = wtime();

	const double gData = aCount * 64.0 / 1024.0 / 1024.0 / 1024.0;
	return gData / (t2 - t1);
}

static void set_affinity(distgend_initT init) {
	const size_t phys_cores_per_numa = init.number_of_threads / (init.NUMA_domains * init.SMT_factor);

	size_t i = 0;

	for (size_t n = 0; n < init.NUMA_domains; ++n) {
		size_t next_core = n * phys_cores_per_numa;
		for (size_t s = 0; s < init.SMT_factor; ++s) {
			for (size_t c = 0; c < phys_cores_per_numa; ++c) {
#pragma omp parallel
				{
					if ((size_t)omp_get_thread_num() == i) {
						// set thread affinity
						cpu_set_t set;
						CPU_ZERO(&set);
						CPU_SET(next_core, &set);

						assert(sched_setaffinity(0, sizeof(set), &set) == 0);
						// pid_t tid = (pid_t)syscall(SYS_gettid);
						// printf("I'm thread %d, %2d, on core %d.\n", tid, omp_get_thread_num(), sched_getcpu());
					}
				}

				++next_core;
				++i;
			}
			next_core += phys_cores_per_numa * (init.NUMA_domains - 1);
		}
	}
	assert(i == init.number_of_threads);
}

static void check_affinity(distgend_initT init) {
	const size_t phys_cores_per_numa = init.number_of_threads / (init.NUMA_domains * init.SMT_factor);

// we expect there to a thread pool. Check if affinity is still ok.
#pragma omp parallel
	{
		size_t s = 0;
		size_t t = (size_t)omp_get_thread_num();
		size_t n = (t / phys_cores_per_numa);
		while (n >= init.NUMA_domains) {
			assert(init.SMT_factor > 1);
			++s;
			--n;
		}
		size_t c = t % phys_cores_per_numa;

		size_t i = 0;
		i += n * init.SMT_factor * phys_cores_per_numa;
		i += s * phys_cores_per_numa;
		i += c;

		cpu_set_t set;
		CPU_ZERO(&set);

		pid_t tid = (pid_t)syscall(SYS_gettid);

		assert(sched_getaffinity(tid, sizeof(set), &set) == 0);

		cpu_set_t expected_res;
		CPU_ZERO(&expected_res);
		CPU_SET(i, &expected_res);

		assert(CPU_EQUAL(&set, &expected_res) != 0);
	}
}
