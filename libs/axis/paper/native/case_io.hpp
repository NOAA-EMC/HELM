// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <netcdf.h>

#include "records.hpp"
namespace paper {
void nc_check(int rc);
struct File {
    int id = -1;
    File(const std::filesystem::path &p, bool write = false);
    ~File();
    File(const File &) = delete;
};
int group(int id, const char *name);
std::string attr(int id, const char *name, const std::string &fallback = "");
std::vector<double> doubles(int id, const char *name);
std::vector<int> integers(int id, const char *name);
void put(int id, const char *name, const std::vector<double> &v);
void put(int id, const char *name, const std::string &s);
Case read_case(const std::filesystem::path &p);
Reference read_reference(const std::filesystem::path &p);
Output read_output(const std::filesystem::path &p);
void write_reference(const std::filesystem::path &p, const Reference &r);
void write_output(const std::filesystem::path &p, const Output &r);
}  // namespace paper
