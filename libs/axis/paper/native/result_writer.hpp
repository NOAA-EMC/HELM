// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <functional>

#include "records.hpp"
namespace paper {
std::string quote(const std::string &s);
std::string number(double x);
void atomic_text(const std::filesystem::path &p, const std::string &s);
void observations(const std::filesystem::path &p, const std::vector<Observation> &v);
int guarded(const std::function<void()> &fn);
}  // namespace paper
