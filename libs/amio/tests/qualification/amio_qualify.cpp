#include "reference_fields.hpp"
#include "backpressure.hpp"
#ifdef AMIO_QUALIFY_HAS_MPI
#include "aggregation.hpp"
#include "stream.hpp"
#endif

#include <charconv>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string_view>

namespace {

bool parse_count(std::string_view text, std::uint64_t &value) {
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size() && value > 0;
}

bool parse_workload(int argc, char **argv, int first, amio::qualification::ReferenceWorkload &workload) {
    if (argc != first + 6 || std::string_view(argv[first]) != "--steps" || std::string_view(argv[first + 2]) != "--fields" ||
        std::string_view(argv[first + 4]) != "--cells") {
        return false;
    }
    return parse_count(argv[first + 1], workload.steps) && parse_count(argv[first + 3], workload.fields) &&
           parse_count(argv[first + 5], workload.cells) && amio::qualification::valid(workload);
}

int self_test() {
    const amio::qualification::ReferenceWorkload workload{2, 3, 5};
    std::stringstream fixture;
    amio::qualification::write_reference(fixture, workload);
    std::string error;
    if (!amio::qualification::verify_reference(fixture, workload, error)) {
        std::cerr << "reference self-test failed: " << error << '\n';
        return 1;
    }
    return 0;
}

void usage(std::ostream &out) {
    out << "Usage:\n"
           "  amio_qualify --self-test\n"
           "  amio_qualify generate --output FILE --steps N --fields N --cells N\n"
           "  amio_qualify verify --input FILE --steps N --fields N --cells N\n"
           "  amio_qualify backpressure --delay-seconds N --budget-mib 64 --output DIR\n"
           "  amio_qualify aggregate --writers N --backend netcdf4 --output DIR (MPI builds)\n"
           "  amio_qualify stream --policy reliable|latest [--slow-consumer-ms N] (MPI builds)\n";
}

} // namespace

int main(int argc, char **argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--self-test") {
        return self_test();
    }
    if (argc < 2) {
        usage(std::cerr);
        return 2;
    }

    const std::string_view command(argv[1]);
    if (command == "backpressure") return run_backpressure_fixture(argc, argv);
#ifdef AMIO_QUALIFY_HAS_MPI
    if (command == "aggregate") return run_aggregation_fixture(argc, argv);
    if (command == "stream") return run_stream_qualification(argc, argv);
#endif
    const bool generate = command == "generate";
    const bool verify = command == "verify";
    if (!generate && !verify) {
        usage(std::cerr);
        return 2;
    }

    const std::string_view path_flag(generate ? "--output" : "--input");
    if (argc < 4 || std::string_view(argv[2]) != path_flag || argv[3][0] == '\0') {
        usage(std::cerr);
        return 2;
    }
    amio::qualification::ReferenceWorkload workload{};
    if (!parse_workload(argc, argv, 4, workload)) {
        usage(std::cerr);
        return 2;
    }

    const std::filesystem::path path(argv[3]);
    if (generate) {
        std::error_code ec;
        if (std::filesystem::exists(path, ec)) {
            std::cerr << "refusing to overwrite existing output: " << path << '\n';
            return 2;
        }
        if (ec) {
            std::cerr << "cannot inspect output path: " << ec.message() << '\n';
            return 5;
        }
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path(), ec);
            if (ec) {
                std::cerr << "cannot create output directory: " << ec.message() << '\n';
                return 5;
            }
        }
        std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!out) {
            std::cerr << "cannot open output: " << path << '\n';
            return 5;
        }
        amio::qualification::write_reference(out, workload);
        if (!out) {
            std::cerr << "failed while writing reference data: " << path << '\n';
            return 5;
        }
        return 0;
    }

    std::ifstream in(path, std::ios::in | std::ios::binary);
    if (!in) {
        std::cerr << "cannot open input: " << path << '\n';
        return 5;
    }
    std::string error;
    if (!amio::qualification::verify_reference(in, workload, error)) {
        std::cerr << "reference verification failed: " << error << '\n';
        return 4;
    }
    return 0;
}
