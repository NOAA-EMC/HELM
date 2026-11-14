#include <gtest/gtest.h>

#include <conf/config.hpp>
#include <conf/error.hpp>
#include <cstdlib>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include "../src/detail/yaml_tree.hpp"

namespace {
thread_local bool count_allocations = false;
thread_local std::size_t allocations = 0;
}  // namespace

// Deterministic resource regression: avoid timing thresholds on busy CI hosts.
void *operator new(std::size_t bytes) {
    if (count_allocations) ++allocations;
    if (void *memory = std::malloc(bytes ? bytes : 1)) return memory;
    throw std::bad_alloc();
}
void *operator new[](std::size_t bytes) {
    return ::operator new(bytes);
}
void operator delete(void *memory) noexcept {
    std::free(memory);
}
void operator delete[](void *memory) noexcept {
    std::free(memory);
}
void operator delete(void *memory, std::size_t) noexcept {
    std::free(memory);
}
void operator delete[](void *memory, std::size_t) noexcept {
    std::free(memory);
}

namespace {
std::size_t LookupAllocations(const conf::detail::Yaml_Tree &tree) {
    allocations = 0;
    count_allocations = true;
    const auto selected = tree.resolve("selected.value");
    count_allocations = false;
    const auto result = allocations;
    EXPECT_EQ(selected.as<int>(), 20);
    return result;
}
}  // namespace

TEST(ResolverSnapshots, ScalarLookupDoesNotCopyUnrelatedDocumentEntries) {
    const std::string prefix = "selected: {value: 20}\nunrelated:\n";
    auto small = conf::detail::Yaml_Tree::from_string(prefix + "  - 0\n");
    std::string document = prefix;
    for (int i = 0; i < 1024; ++i) document += "  - " + std::to_string(i) + "\n";
    auto large = conf::detail::Yaml_Tree::from_string(document);
    const auto small_allocations = LookupAllocations(small);
    const auto large_allocations = LookupAllocations(large);
    RecordProperty("small_lookup_allocations", static_cast<int>(small_allocations));
    RecordProperty("large_lookup_allocations", static_cast<int>(large_allocations));
    EXPECT_LE(large_allocations, small_allocations + 32);
}

TEST(ResolverSnapshots, ResolvedNodesRemainIndependentSnapshots) {
    auto tree = conf::detail::Yaml_Tree::from_string("root:\n  items:\n    - value: 10\n    - value: 20\nother: 30\n");
    const auto &root = tree.root();
    const YAML::Node expected = root["root"]["items"][1]["value"];
    YAML::Node resolved = tree.resolve("root.items.1.value");
    EXPECT_FALSE(resolved.is(expected));
    resolved = 99;
    EXPECT_EQ(tree.resolve("root.items.1.value").as<int>(), 20);
}

TEST(ResolverSnapshots, SuccessfulAndMissingLookupsLeaveRootAndAliasesUnchanged) {
    auto tree = conf::detail::Yaml_Tree::from_string("left: &shared\n  items: [10, 20]\nright: *shared\nother: 30\n");
    const auto before = YAML::Dump(tree.root());
    for (int repeat = 0; repeat < 100; ++repeat) {
        EXPECT_EQ(tree.resolve("left.items.0").as<int>(), 10);
        EXPECT_EQ(tree.resolve("right.items.1").as<int>(), 20);
        EXPECT_EQ(tree.resolve("other").as<int>(), 30);
        EXPECT_THROW((void)tree.resolve("left.missing"), conf::Conf_Error);
        EXPECT_THROW((void)tree.resolve("right.items.99"), conf::Conf_Error);
        EXPECT_THROW((void)tree.resolve("left.items.0.child"), conf::Conf_Error);
        EXPECT_EQ(YAML::Dump(tree.root()), before);
    }
}

TEST(ResolverSnapshots, CyclicAliasesKeepIndependentSnapshotSemantics) {
    auto tree = conf::detail::Yaml_Tree::from_string("loop: &loop {self: *loop, value: 7}\n");
    auto snapshot = tree.resolve("loop");
    snapshot["value"] = 99;
    EXPECT_EQ(tree.resolve("loop.self.self.value").as<int>(), 7);
}

TEST(ResolverSnapshots, ConcurrentFirstReadsKeepSharedConfigUnchanged) {
    const auto config = conf::Config::from_string("items: [10, 20, 30]\nroot: {value: 7}\n");
    std::vector<std::thread> readers;
    for (int thread = 0; thread < 8; ++thread) {
        readers.emplace_back([&] {
            for (int repeat = 0; repeat < 100; ++repeat) {
                EXPECT_EQ(config.get_int("items.1"), 20);
                EXPECT_EQ(config.get_int("root.value"), 7);
                EXPECT_EQ(config.get_int_list("items"), (std::vector<int>{10, 20, 30}));
                EXPECT_EQ(config.size("items"), 3u);
                EXPECT_FALSE(config.has("root.missing"));
            }
        });
    }
    for (auto &reader : readers) reader.join();
    EXPECT_EQ(config.get_int("items.0"), 10);
}
