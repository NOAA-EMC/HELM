/** @file
 * @brief Shared-file GRIB2 MPI tests using public AMIO writes and independent g2c decoding.
 */
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <halo/collectives.hpp>
#include <iterator>
#include <logs/logger.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "amio/amio.h"
extern "C" {
#include <grib2.h>
}

namespace {
/** @brief Fail the test with an exception handled by the MPI-aware main routine. */
void require(bool ok, const std::string &message) {
    if (!ok) throw std::runtime_error(message);
}
/** @brief Check and describe a public AMIO status without hiding its numeric value. */
void status_is(amio_status_t actual, amio_status_t expected, const std::string &operation) {
    require(actual == expected, operation + ": expected " + std::to_string(expected) + ", got " + std::to_string(actual));
}
/** @brief Synchronize test phases using HELM HALO. */
void sync(const halo::Communicator &comm) {
    (void)halo::allreduce(comm, std::vector<int>{1}, MPI_SUM);
}
/** @brief Construct a rank-specific product manifest for one common output path. */
std::string manifest(const std::string &path, int rank, const std::string &drt) {
    return "backend: grib2\npath: " + path + "\nstaging_pool: {buffer_count: 32, buffer_capacity_bytes: 65536}\nworker_pool: {threads: 4}\n" +
           "data_representation_template: " + drt + "\ngrib2: {discipline: 0, parameter_category: 0, parameter_number: " + std::to_string(rank) +
           ", type_of_first_fixed_surface: 100, scaled_value_first_surface: 50000, decimal_scale_factor: 0}\n";
}
/** @brief Unequal complete-field grids produce unequal message sizes across ranks. */
amio_shape_t shape_for(int rank) {
    amio_shape_t shape{};
    shape.rank = 2;
    shape.extents[0] = 8 + rank;
    shape.extents[1] = 11 + 2 * rank;
    return shape;
}
/** @brief Exactly representable values distinguish file, round, rank, and grid point. */
std::vector<float> field_for(int rank, int round, int file) {
    const auto shape = shape_for(rank);
    std::vector<float> field(shape.extents[0] * shape.extents[1]);
    for (std::size_t i = 0; i < field.size(); ++i) field[i] = file * 1000000 + round * 10000 + rank * 1000 + i;
    return field;
}
/** @brief Verify message framing, exact ordering, product identity, and decoded values. */
void verify_file(const std::string &path, int ranks, int rounds, int file) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "open independent GRIB reader");
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    std::size_t position = 0;
    std::vector<std::uint64_t> lengths;
    for (int round = 0; round < rounds; ++round) {
        for (int rank = 0; rank < ranks; ++rank) {
            require(bytes.size() - position >= 20, "missing or truncated GRIB message");
            require(std::memcmp(bytes.data() + position, "GRIB", 4) == 0, "GRIB indicator at assigned offset");
            std::uint64_t length = 0;
            for (int i = 8; i < 16; ++i) length = (length << 8) | bytes[position + i];
            require(length >= 20 && length <= bytes.size() - position, "valid encoded length");
            require(std::memcmp(bytes.data() + position + length - 4, "7777", 4) == 0, "complete message terminator");
            gribfield *decoded = nullptr;
            require(g2_getfld(bytes.data() + position, 1, 1, 1, &decoded) == 0 && decoded, "independent g2c decode");
            const auto expected = field_for(rank, round, file);
            require(decoded->ngrdpts == static_cast<g2int>(expected.size()), "complete field grid size");
            require(decoded->ipdtmpl[1] == rank, "message order is round then communicator rank");
            for (std::size_t i = 0; i < expected.size(); ++i) require(decoded->fld[i] == expected[i], "snapshot preserved and bytes do not overlap");
            g2_free(decoded);
            lengths.push_back(length);
            position += static_cast<std::size_t>(length);
        }
    }
    require(position == bytes.size(), "no stale trailing data or failed-round bytes");
    if (ranks > 1)
        require(*std::min_element(lengths.begin(), lengths.end()) != *std::max_element(lengths.begin(), lengths.end()),
                "exercise unequal encoded lengths");
}
/** @brief Enqueue a field, then immediately reuse its host buffer. */
amio_io_handle submit(amio_dataset_handle dataset, int rank, int round, int file) {
    auto field = field_for(rank, round, file);
    const auto shape = shape_for(rank);
    amio_io_handle io = nullptr;
    status_is(amio_write(dataset, "field", field.data(), AMIO_DTYPE_F32, &shape, &io), AMIO_OK, "enqueue GRIB field");
    std::fill(field.begin(), field.end(), -999);
    return io;
}
/** @brief Exercise multiple active datasets, worker ordering, truncation, and rank-local messages. */
void round_trip(const halo::Communicator &comm, amio_core_handle core, const std::string &prefix, const std::string &drt) {
    std::array<amio_dataset_handle, 2> datasets{};
    std::array<std::string, 2> paths{prefix + "-a.grb2", prefix + "-b.grb2"};
    for (int file = 0; file < 2; ++file) {
        if (comm.rank() == 0) {
            std::ofstream stale(paths[file], std::ios::binary);
            stale << std::string(65536, 'x');
        }
        sync(comm);
        const auto cfg = manifest(paths[file], comm.rank(), drt);
        status_is(amio_open_dataset_from_string(core, cfg.c_str(), "yaml", AMIO_MODE_WRITE, &datasets[file]), AMIO_OK, "collective GRIB open");
    }
    std::vector<amio_io_handle> pending;
    for (int round = 0; round < 6; ++round)
        for (int file = 0; file < 2; ++file) pending.push_back(submit(datasets[file], comm.rank(), round, file));
    for (auto io : pending) status_is(amio_wait(io, 10000), AMIO_OK, "collective message completion");
    for (auto dataset : datasets) {
        status_is(amio_flush(dataset, 10000), AMIO_OK, "MPI file sync");
        status_is(amio_close_dataset(dataset), AMIO_OK, "MPI file close");
    }
    sync(comm);
    if (comm.rank() == 0)
        for (int file = 0; file < 2; ++file) {
            verify_file(paths[file], comm.size(), 6, file);
            std::filesystem::remove(paths[file]);
        }
    sync(comm);
}
/** @brief One rank's encode failure must complete every handle without writing a partial round. */
void asymmetric_failure(const halo::Communicator &comm, amio_core_handle core, const std::string &path) {
    const auto cfg = manifest(path, comm.rank(), "libaec");
    amio_dataset_handle dataset = nullptr;
    status_is(amio_open_dataset_from_string(core, cfg.c_str(), "yaml", AMIO_MODE_WRITE, &dataset), AMIO_OK, "open failure-test output");
    status_is(amio_wait(submit(dataset, comm.rank(), 0, 3), 10000), AMIO_OK, "commit initial good round");
    status_is(amio_flush(dataset, 10000), AMIO_OK, "sync good prefix");
    const auto prefix_size = std::filesystem::file_size(path);
    const auto shape = shape_for(comm.rank());
    auto valid = field_for(comm.rank(), 1, 3);
    std::vector<double> invalid(valid.begin(), valid.end());
    amio_io_handle bad = nullptr;
    const bool inject = comm.rank() == 1;
    status_is(amio_write(dataset, "field", inject ? static_cast<void *>(invalid.data()) : static_cast<void *>(valid.data()),
                         inject ? AMIO_DTYPE_F64 : AMIO_DTYPE_F32, &shape, &bad),
              AMIO_OK, "admit rank-local encode failure");
    const auto after_failure = submit(dataset, comm.rank(), 2, 3);
    status_is(amio_wait(bad, 10000), AMIO_ERR_BACKEND_FAILURE, "all ranks observe encode failure");
    status_is(amio_wait(after_failure, 10000), AMIO_ERR_BACKEND_FAILURE, "poisoned file rejects later appends");
    status_is(amio_flush(dataset, 10000), AMIO_ERR_BACKEND_FAILURE, "flush reports failure");
    status_is(amio_close_dataset(dataset), AMIO_ERR_BACKEND_FAILURE, "close releases failed dataset");
    sync(comm);
    require(std::filesystem::file_size(path) == prefix_size, "failed round emitted zero bytes");
    if (comm.rank() == 0) {
        verify_file(path, comm.size(), 1, 3);
        std::filesystem::remove(path);
    }
    sync(comm);
}
/** @brief Configuration/path disagreement and failed file opens must reject collectively. */
void rejected_opens(const halo::Communicator &comm, amio_core_handle core, const std::string &prefix) {
    for (int mode = 0; mode < 3; ++mode) {
        const auto path = mode == 2 ? prefix + "-missing/output.grb2"
                                    : prefix + "-reject" + std::to_string(mode) + (comm.rank() == 1 && mode == 1 ? "b" : "a") + ".grb2";
        const auto cfg = manifest(path, comm.rank(), comm.rank() == 1 && mode == 0 ? "invalid" : "libaec");
        amio_dataset_handle dataset = nullptr;
        status_is(amio_open_dataset_from_string(core, cfg.c_str(), "yaml", AMIO_MODE_WRITE, &dataset), AMIO_ERR_BACKEND_FAILURE,
                  "collectively reject invalid open");
        require(!dataset && !std::filesystem::exists(path), "invalid open creates no output");
    }
}
}  // namespace

/** @brief Initialize MPI, execute real codec tests, and abort all ranks on assertion failure. */
int main(int argc, char **argv) {
    const bool single = argc > 1 && std::string(argv[1]) == "single";
    int provided = 0;
    MPI_Init_thread(&argc, &argv, single ? MPI_THREAD_SINGLE : MPI_THREAD_MULTIPLE, &provided);
    halo::Environment::initialize();
    {
        halo::Communicator comm(MPI_COMM_WORLD);
        logs::Logger logger;
        logger.configure_communicator(comm.handle());
        try {
            require(single || provided == MPI_THREAD_MULTIPLE, "MPI_THREAD_MULTIPLE required");
            const auto prefix = (std::filesystem::current_path() / "amio-mpi-grib2").string();
            const auto cfg = manifest(prefix + "-gate.grb2", comm.rank(), "libaec");
            amio_core_handle core = nullptr;
            status_is(amio_init_from_string(cfg.c_str(), "yaml", &core), AMIO_OK, "init");
            if (single) {
                amio_dataset_handle dataset = nullptr;
                status_is(amio_open_dataset_from_string(core, cfg.c_str(), "yaml", AMIO_MODE_WRITE, &dataset), AMIO_ERR_THREADING_UNSUPPORTED,
                          "GRIB2 background MPI thread gate");
                require(!dataset && !std::filesystem::exists(prefix + "-gate.grb2"), "thread rejection precedes file creation");
            } else {
                round_trip(comm, core, prefix + "-aec", "libaec");
                round_trip(comm, core, prefix + "-jpeg", "lossless_jpeg2000");
                if (comm.size() > 1) {
                    asymmetric_failure(comm, core, prefix + "-failure.grb2");
                    rejected_opens(comm, core, prefix);
                }
            }
            status_is(amio_finalize(core), AMIO_OK, "finalize");
        } catch (const std::exception &error) {
            logger.log(logs::Severity_Level::ERROR, error.what());
            MPI_Abort(comm.handle(), 1);
        }
    }
    MPI_Finalize();
}
