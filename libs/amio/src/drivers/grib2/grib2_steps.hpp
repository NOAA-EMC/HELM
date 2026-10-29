#ifndef AMIO_SRC_DRIVERS_GRIB2_GRIB2_STEPS_HPP
#define AMIO_SRC_DRIVERS_GRIB2_GRIB2_STEPS_HPP

#include "catalog/catalog_format.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace amio::detail {

struct Grib2Settings;

/** Byte range occupied by one complete GRIB2 message in a step payload. */
struct Grib2MessageRange {
    std::uint64_t offset = 0;
    std::uint64_t length = 0;
};

/** One field's ordered message locations (producer then submission round). */
struct Grib2FieldRanges {
    std::string field;
    std::uint32_t producer = 0;
    std::vector<Grib2MessageRange> messages;
};

/** Immutable-after-publication byte payload plus ordered field/message index. */
struct Grib2StepPayload {
    std::vector<unsigned char> bytes;
    std::vector<Grib2FieldRanges> fields;
};

/** Append a complete message and return its location. Reject overflow and empty messages. */
[[nodiscard]] Grib2MessageRange append_grib2_message_range(std::uint64_t &payload_bytes, std::uint64_t message_bytes);

/** Check the producer-local encoded message and codec scratch ceilings before allocating either. */
void preflight_grib2_message(std::size_t field_bytes, bool requires_repack,
                             std::size_t max_encoded_buffer_bytes, std::size_t max_codec_scratch_bytes);

/** Reject partial-message transfer; GRIB2 fragments may not be composed into a message. */
void require_complete_grib2_message(bool complete_message);
/** Validate the GRIB2 indicator length, edition, and end marker of encoded bytes. */
void validate_complete_grib2_message(std::span<const unsigned char> message);

/**
 * Append a producer's complete encoded messages and record their byte ranges.
 * The whole field is validated and budgeted before the payload is mutated.
 * Repeated (field, producer) entries and regional fragments are rejected.
 */
void append_grib2_field_messages(Grib2StepPayload &payload, const std::string &field,
                                 std::uint32_t producer,
                                 std::span<const std::vector<unsigned char>> messages,
                                 std::uint64_t max_payload_bytes,
                                 bool complete_field = true);

/** Write a complete immutable message stream using create-only publication. */
[[nodiscard]] bool write_grib2_step_payload(const std::string &path, const Grib2StepPayload &payload,
                                            std::string &error);
/** Read and validate one field's indexed complete messages under a byte ceiling. */
[[nodiscard]] bool read_grib2_field_messages(const std::string &path, const Grib2FieldRanges &field,
                                             std::size_t max_encoded_bytes,
                                             std::vector<std::vector<unsigned char>> &messages,
                                             std::string &error);
/** Add each complete field's message ranges to a completion record. */
[[nodiscard]] amio_status_t append_grib2_catalog_payloads(const Grib2StepPayload &payload,
                                                           const std::string &relative_path,
                                                           CompletionRecord &record);

/** Validate configured grid/product integers and dimensions before g2c casts them. */
void validate_grib2_representability(const Grib2Settings &settings, std::int64_t ni, std::int64_t nj);

} // namespace amio::detail

#endif
