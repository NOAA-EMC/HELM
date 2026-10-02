// generators.hpp -- RapidCheck Arbitrary<T> specializations for AMIO types.
//
// Provides reusable generators for property-based tests:
//   * Arbitrary<amio_shape_t>  -- valid shapes (rank [1,7], extents [1,1024])
//   * Arbitrary<amio_dtype_t>  -- uniform over 10 valid enum values
//   * Arbitrary<amio::detail::Config> (aliased as Manifest)
//   * Arbitrary<amio::detail::CodecConfig>
//   * Arbitrary<amio::pbt::Payload> -- random bytes sized for shape+dtype
//
// Also provides "invalid" generators for negative testing:
//   * genInvalidShape()   -- out-of-range rank, zero/negative extents
//   * genInvalidManifest() -- out-of-range fields, missing fields
//
// Validates: R11.2 (testing infrastructure)

#ifndef AMIO_TESTS_PBT_GENERATORS_HPP
#define AMIO_TESTS_PBT_GENERATORS_HPP

#include <rapidcheck.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "pbt_common.hpp"

namespace rc {

// ===================================================================
// Arbitrary<amio_dtype_t> -- uniform over the 10 valid enum values.
// ===================================================================

template <>
struct Arbitrary<amio_dtype_t> {
    static Gen<amio_dtype_t> arbitrary() {
        return gen::elementOf(std::vector<amio_dtype_t>{AMIO_DTYPE_F32, AMIO_DTYPE_F64, AMIO_DTYPE_I8, AMIO_DTYPE_I16, AMIO_DTYPE_I32, AMIO_DTYPE_I64,
                                                        AMIO_DTYPE_U8, AMIO_DTYPE_U16, AMIO_DTYPE_U32, AMIO_DTYPE_U64});
    }
};

// ===================================================================
// Arbitrary<amio_shape_t> -- valid shapes.
//
// Generates:
//   rank    in [1, 7]
//   extents in [1, 1024] for each dimension up to rank
//   strides = 0 (contiguous, AMIO derives strides from extents)
//   Entries beyond rank are zero.
//
// The total element count is bounded to avoid generating payloads
// that exceed reasonable memory for testing (cap at ~16M elements).
// ===================================================================

template <>
struct Arbitrary<amio_shape_t> {
    static Gen<amio_shape_t> arbitrary() {
        return gen::exec([]() {
            amio_shape_t shape = {};

            // Generate rank in [1, 7]
            shape.rank = *gen::inRange(1, 8);

            // Generate extents such that total element count stays
            // reasonable for testing.  We cap per-dimension extent
            // based on rank to keep total elements bounded.
            //
            // For rank 1: up to 1024
            // For rank 2: up to 1024 per dim (~1M total)
            // For rank 3-4: up to 64 per dim (~16M total)
            // For rank 5-7: up to 16 per dim (~16M total)
            int max_extent = 1024;
            if (shape.rank >= 3 && shape.rank <= 4) {
                max_extent = 64;
            } else if (shape.rank >= 5) {
                max_extent = 16;
            }

            for (int32_t d = 0; d < shape.rank; ++d) {
                shape.extents[d] = *gen::inRange(static_cast<int64_t>(1), static_cast<int64_t>(max_extent + 1));
            }

            // Entries beyond rank must be zero (already initialized).
            // Strides = 0 means contiguous (AMIO derives row-major).
            for (int32_t d = 0; d < AMIO_MAX_RANK; ++d) {
                shape.strides[d] = 0;
            }

            return shape;
        });
    }
};

// ===================================================================
// Arbitrary<amio::detail::CodecConfig> -- valid codec configurations.
// ===================================================================

template <>
struct Arbitrary<amio::detail::CodecConfig> {
    static Gen<amio::detail::CodecConfig> arbitrary() {
        return gen::exec([]() {
            amio::detail::CodecConfig cfg;

            // Valid codec names
            static const std::vector<std::string> valid_codecs = {"blosc", "zstandard", "libaec", "lossless_jpeg2000"};

            // Generate a non-empty subset of valid codecs for the allow-list
            std::size_t list_size = *gen::inRange<std::size_t>(1, valid_codecs.size() + 1);
            std::vector<std::size_t> indices;
            for (std::size_t i = 0; i < valid_codecs.size(); ++i) {
                indices.push_back(i);
            }
            // Shuffle and take first list_size
            for (std::size_t i = indices.size() - 1; i > 0; --i) {
                std::size_t j = *gen::inRange<std::size_t>(0, i + 1);
                std::swap(indices[i], indices[j]);
            }
            for (std::size_t i = 0; i < list_size; ++i) {
                cfg.lossless_allow_list.push_back(valid_codecs[indices[i]]);
            }

            // Active codec must be on the allow-list
            std::size_t active_idx = *gen::inRange<std::size_t>(0, cfg.lossless_allow_list.size());
            cfg.active_codec = cfg.lossless_allow_list[active_idx];

            return cfg;
        });
    }
};

// ===================================================================
// Arbitrary<amio::detail::Config> -- valid manifests.
//
// All numeric fields are generated within their declared inclusive
// ranges.  All required fields are present.
// ===================================================================

template <>
struct Arbitrary<amio::detail::Config> {
    static Gen<amio::detail::Config> arbitrary() {
        return gen::exec([]() {
            amio::detail::Config cfg;

            // Staging pool: buffer_count [1, 4096], capacity [1, 1 GiB]
            // For testing, cap capacity at 1 MiB to avoid OOM.
            cfg.staging_pool.buffer_count = *gen::inRange<std::size_t>(1, 4097);
            cfg.staging_pool.buffer_capacity_bytes = *gen::inRange<std::size_t>(1, 1048577);  // [1, 1 MiB] for tests
            // Auto-grow ceiling: [buffer_count, 4096] (validate() requires
            // max >= count so the pool can always hold its provisioned slots).
            cfg.staging_pool.max_buffer_count = *gen::inRange<std::size_t>(cfg.staging_pool.buffer_count, 4097);

            // Worker pool: threads [1, 256]
            cfg.worker_pool.threads = *gen::inRange<std::size_t>(1, 257);

            // Optional CPU cores (empty or small list)
            bool has_cores = *gen::arbitrary<bool>();
            if (has_cores) {
                std::size_t n_cores = *gen::inRange<std::size_t>(1, 5);
                for (std::size_t i = 0; i < n_cores; ++i) {
                    cfg.worker_pool.cpu_cores.push_back(*gen::inRange(0, 64));
                }
            }

            // Optional NUMA domain
            bool has_numa = *gen::arbitrary<bool>();
            if (has_numa) {
                cfg.worker_pool.numa_domain = *gen::inRange(0, 8);
            }

            // Prefetch: depth [1, 1024], read_timeout_s [1, 3600]
            cfg.prefetch.depth = *gen::inRange<std::size_t>(1, 1025);
            cfg.prefetch.read_timeout_s = *gen::inRange<std::size_t>(1, 3601);

            // Staging timeout [1, 60000] ms
            cfg.staging_timeout_ms = *gen::inRange<std::size_t>(1, 60001);

            // Backpressure: 0 <= L < H <= capacity
            cfg.backpressure.queue_capacity = *gen::inRange<std::size_t>(2, 4097);
            cfg.backpressure.high_watermark = *gen::inRange<std::size_t>(1, cfg.backpressure.queue_capacity + 1);
            cfg.backpressure.low_watermark = *gen::inRange<std::size_t>(0, cfg.backpressure.high_watermark);

            // Backend key
            cfg.backend = *gen::elementOf(std::vector<std::string>{"netcdf4", "zarr3", "grib2"});

            // Codec configuration
            cfg.codec = *gen::arbitrary<amio::detail::CodecConfig>();

            // I/O ranks (empty or small list)
            bool has_io_ranks = *gen::arbitrary<bool>();
            if (has_io_ranks) {
                std::size_t n_ranks = *gen::inRange<std::size_t>(1, 5);
                for (std::size_t i = 0; i < n_ranks; ++i) {
                    cfg.io_ranks.push_back(*gen::inRange(0, 64));
                }
            }

            return cfg;
        });
    }
};

// ===================================================================
// Arbitrary<amio::pbt::Payload> -- random payload bytes sized for
// a generated shape + dtype.
// ===================================================================

template <>
struct Arbitrary<amio::pbt::Payload> {
    static Gen<amio::pbt::Payload> arbitrary() {
        return gen::exec([]() {
            amio::pbt::Payload payload;
            payload.dtype = *gen::arbitrary<amio_dtype_t>();
            payload.shape = *gen::arbitrary<amio_shape_t>();

            std::size_t byte_count = amio::pbt::payload_byte_count(payload.shape, payload.dtype);

            // Cap payload size for testing speed (4 MiB max).
            if (byte_count > 4 * 1024 * 1024) {
                // Shrink shape to fit.
                payload.shape.rank = 1;
                payload.shape.extents[0] = 1024;
                for (int32_t d = 1; d < AMIO_MAX_RANK; ++d) {
                    payload.shape.extents[d] = 0;
                }
                byte_count = amio::pbt::payload_byte_count(payload.shape, payload.dtype);
            }

            payload.bytes.resize(byte_count);
            for (std::size_t i = 0; i < byte_count; ++i) {
                payload.bytes[i] = static_cast<uint8_t>(*gen::inRange(0, 256));
            }

            return payload;
        });
    }
};

}  // namespace rc

// ===================================================================
// Custom generators in amio::pbt namespace
// ===================================================================

namespace amio::pbt {

// Generate an invalid shape (for negative testing).
// Produces shapes with:
//   - rank outside [1, 7], OR
//   - zero/negative extents, OR
//   - non-zero entries beyond rank
inline rc::Gen<amio_shape_t> genInvalidShape() {
    return rc::gen::exec([]() {
        amio_shape_t shape = {};

        // Choose which kind of invalidity to produce
        int kind = *rc::gen::inRange(0, 4);

        switch (kind) {
            case 0: {
                // rank = 0
                shape.rank = 0;
                break;
            }
            case 1: {
                // rank > AMIO_MAX_RANK
                shape.rank = *rc::gen::inRange(8, 32);
                for (int d = 0; d < AMIO_MAX_RANK; ++d) {
                    shape.extents[d] = *rc::gen::inRange<int64_t>(1, 100);
                }
                break;
            }
            case 2: {
                // Valid rank but zero extent in one dimension
                shape.rank = *rc::gen::inRange(1, 8);
                for (int32_t d = 0; d < shape.rank; ++d) {
                    shape.extents[d] = *rc::gen::inRange<int64_t>(1, 100);
                }
                // Zero out one random dimension
                int bad_dim = *rc::gen::inRange(0, static_cast<int>(shape.rank));
                shape.extents[bad_dim] = 0;
                break;
            }
            case 3: {
                // Valid rank but negative extent in one dimension
                shape.rank = *rc::gen::inRange(1, 8);
                for (int32_t d = 0; d < shape.rank; ++d) {
                    shape.extents[d] = *rc::gen::inRange<int64_t>(1, 100);
                }
                int bad_dim = *rc::gen::inRange(0, static_cast<int>(shape.rank));
                shape.extents[bad_dim] = *rc::gen::inRange<int64_t>(-100, 0);
                break;
            }
        }

        return shape;
    });
}

// Generate an invalid manifest (for negative testing).
// Produces configs with at least one field out of range.
inline rc::Gen<amio::detail::Config> genInvalidManifest() {
    return rc::gen::exec([]() {
        // Start with a valid config and corrupt one field
        auto cfg = *rc::gen::arbitrary<amio::detail::Config>();

        int field = *rc::gen::inRange(0, 7);
        switch (field) {
            case 0: {
                // buffer_count out of range
                static const std::vector<std::size_t> bad_counts = {0, 4097, 10000};
                cfg.staging_pool.buffer_count = *rc::gen::elementOf(bad_counts);
                break;
            }
            case 1:
                // buffer_capacity out of range (> 1 GiB)
                cfg.staging_pool.buffer_capacity_bytes = static_cast<std::size_t>(1'073'741'825);
                break;
            case 2: {
                // threads out of range
                static const std::vector<std::size_t> bad_threads = {0, 257, 1000};
                cfg.worker_pool.threads = *rc::gen::elementOf(bad_threads);
                break;
            }
            case 3: {
                // prefetch depth out of range
                static const std::vector<std::size_t> bad_depths = {0, 1025, 5000};
                cfg.prefetch.depth = *rc::gen::elementOf(bad_depths);
                break;
            }
            case 4: {
                // read_timeout_s out of range
                static const std::vector<std::size_t> bad_timeouts = {0, 3601, 10000};
                cfg.prefetch.read_timeout_s = *rc::gen::elementOf(bad_timeouts);
                break;
            }
            case 5: {
                // staging_timeout_ms out of range
                static const std::vector<std::size_t> bad_staging = {0, 60001, 100000};
                cfg.staging_timeout_ms = *rc::gen::elementOf(bad_staging);
                break;
            }
            case 6:
                // Invalid codec (not on allow-list)
                cfg.codec.active_codec = "lossy_jpeg";
                break;
        }

        return cfg;
    });
}

// Generate a payload (random bytes) sized for a given shape + dtype.
inline rc::Gen<std::vector<std::byte>> genPayload(const amio_shape_t &shape, amio_dtype_t dtype) {
    std::size_t byte_count = payload_byte_size(shape, dtype);
    return rc::gen::exec([byte_count]() {
        std::vector<std::byte> payload(byte_count);
        for (std::size_t i = 0; i < byte_count; ++i) {
            payload[i] = static_cast<std::byte>(*rc::gen::inRange(0, 256));
        }
        return payload;
    });
}

// Generate a payload as a vector of bytes for a given size.
inline rc::Gen<std::vector<std::byte>> genPayloadOfSize(std::size_t byte_count) {
    return rc::gen::exec([byte_count]() {
        std::vector<std::byte> payload(byte_count);
        for (std::size_t i = 0; i < byte_count; ++i) {
            payload[i] = static_cast<std::byte>(*rc::gen::inRange(0, 256));
        }
        return payload;
    });
}

}  // namespace amio::pbt

#endif  // AMIO_TESTS_PBT_GENERATORS_HPP
