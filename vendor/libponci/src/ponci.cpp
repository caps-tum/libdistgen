/**
 * po     n  c       i
 * poor mans cgroups interface
 *
 * Copyright 2016 by LRR-TUM
 * Jens Breitbart     <j.breitbart@tum.de>
 *
 * Licensed under GNU Lesser General Public License 2.1 or later.
 * Some rights reserved. See LICENSE
 */

#include "ponci/ponci.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <cassert>
#include <cstring>

#include <errno.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

// TODO function to change prefix

// default mount path
static std::string path_prefix("/sys/fs/cgroup/");

/////////////////////////////////////////////////////////////////
// PROTOTYPES
/////////////////////////////////////////////////////////////////
static inline std::string cgroup_path(const char *name);
template <typename T> static inline void write_array_to_file(std::string filename, T *arr, size_t size);
template <typename T> static inline void write_value_to_file(std::string filename, T val);
template <typename T> static inline void append_value_to_file(std::string filename, T val);
template <typename T> static inline void check_value_in_file(std::string filename, T val);

/////////////////////////////////////////////////////////////////
// EXPORTED FUNCTIONS
/////////////////////////////////////////////////////////////////
void cgroup_create(const char *name) {
	const int err = mkdir(cgroup_path(name).c_str(), S_IRWXU | S_IRWXG);

	if (err != 0 && errno != EEXIST) throw std::runtime_error(strerror(errno));
	errno = 0;
}

void cgroup_delete(const char *name) {
	const int err = rmdir(cgroup_path(name).c_str());

	if (err != 0) throw std::runtime_error(strerror(errno));
}
void cgroup_add_me(const char *name) {
	pid_t me = static_cast<pid_t>(syscall(SYS_gettid));
	cgroup_add_task(name, me);
}

void cgroup_add_task(const char *name, const pid_t tid) {
	std::string filename = cgroup_path(name) + std::string("tasks");

	append_value_to_file(filename, tid);

	check_value_in_file(filename, tid);
}

void cgroup_set_cpus(const char *name, const size_t *cpus, size_t size) {
	std::string filename = cgroup_path(name) + std::string("cpuset.cpus");

	write_array_to_file(filename, cpus, size);
	// TODO add check. The current check function does not understand the output
}

void cgroup_set_mems(const char *name, const size_t *mems, size_t size) {
	std::string filename = cgroup_path(name) + std::string("cpuset.mems");

	write_array_to_file(filename, mems, size);
	// TODO add check. The current check function does not understand the output
}

void cgroup_set_memory_migrate(const char *name, size_t flag) {
	assert(flag == 0 || flag == 1);
	std::string filename = cgroup_path(name) + std::string("cpuset.memory_migrate");

	write_value_to_file(filename, flag);
	check_value_in_file(filename, flag);
}

void cgroup_set_cpus_exclusive(const char *name, size_t flag) {
	assert(flag == 0 || flag == 1);
	std::string filename = cgroup_path(name) + std::string("cpuset.cpu_exclusive");

	write_value_to_file(filename, flag);
	check_value_in_file(filename, flag);
}

void cgroup_set_mem_hardwall(const char *name, size_t flag) {
	assert(flag == 0 || flag == 1);
	std::string filename = cgroup_path(name) + std::string("cpuset.mem_hardwall");

	write_value_to_file(filename, flag);
	check_value_in_file(filename, flag);
}

void cgroup_set_scheduling_domain(const char *name, int flag) {
	assert(flag >= -1 && flag <= 5);
	std::string filename = cgroup_path(name) + std::string("cpuset.sched_relax_domain_level");

	write_value_to_file(filename, flag);
	check_value_in_file(filename, flag);
}

void cgroup_freeze(const char *name) {
	assert(strcmp(name, "") != 0);
	std::string filename = cgroup_path(name) + std::string("freezer.state");

	write_value_to_file(filename, "FROZEN");
}

void cgroup_thaw(const char *name) {
	assert(strcmp(name, "") != 0);
	std::string filename = cgroup_path(name) + std::string("freezer.state");

	write_value_to_file(filename, "THAWED");
}

/////////////////////////////////////////////////////////////////
// INTERNAL FUNCTIONS
/////////////////////////////////////////////////////////////////

static inline std::string cgroup_path(const char *name) {
	std::string res(path_prefix);
	if (strcmp(name, "") != 0) {
		res.append(name);
		res.append("/");
	}
	return res;
}

template <typename T> static inline void write_array_to_file(std::string filename, T *arr, size_t size) {
	assert(size > 0);
	assert(filename.compare("") != 0);

	std::ofstream file;
	file.open(filename, std::ofstream::out);
	if (!file.good()) throw std::runtime_error(strerror(errno));

	std::string str;
	for (size_t i = 0; i < size; ++i) {
		str.append(std::to_string(arr[i]));
		str.append(",");
	}

	file << str;
	if (!file.good()) throw std::runtime_error(strerror(errno));
}

template <typename T> static inline void write_value_to_file(std::string filename, T val) {
	assert(filename.compare("") != 0);

	std::ofstream file;
	file.open(filename, std::ofstream::out);
	if (!file.good()) throw std::runtime_error(strerror(errno));

	file << val;
	if (!file.good()) throw std::runtime_error(strerror(errno));
}

template <typename T> static inline void append_value_to_file(std::string filename, T val) {
	assert(filename.compare("") != 0);

	std::ofstream file;
	file.open(filename, std::ofstream::app);
	if (!file.good()) throw std::runtime_error(strerror(errno));

	file << val;
	if (!file.good()) throw std::runtime_error(strerror(errno));
}

template <typename T> static inline void check_value_in_file(std::string filename, T val) {
	// check if the value was actually written
	// ofstream errors are not sufficient (?)
	std::ifstream infile(filename);
	std::string line;

	while (std::getline(infile, line)) {
		std::istringstream iss(line);
		T n;
		iss >> n;
		if (val == n) return;
	}
	throw std::runtime_error("Value was not found in file.");
}
