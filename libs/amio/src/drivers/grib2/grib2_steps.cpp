#include "drivers/grib2/grib2_steps.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

#include "drivers/grib2/grib2_driver.hpp"

namespace amio::detail {

Grib2StreamWriter::Grib2StreamWriter(const std::string &path, std::size_t max_chunk_bytes, std::size_t max_index_entries,
                                     std::uint64_t max_message_bytes)
    : path_(path), max_chunk_bytes_(max_chunk_bytes), max_index_entries_(max_index_entries), max_message_bytes_(max_message_bytes) {
    if (path.empty() || max_chunk_bytes == 0 || max_index_entries == 0 || max_message_bytes < 20 ||
        max_chunk_bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) || max_index_entries > index_.max_size())
        throw std::invalid_argument("invalid GRIB2 stream writer limits");
    index_.reserve(max_index_entries);
    static std::atomic<std::uint64_t> sequence{0};
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count() ^ sequence.fetch_add(1);
    temporary_ = path + ".amio-stream-" + std::to_string(nonce);
    stream_ = std::fopen(temporary_.c_str(), "wbx");
    if (stream_ == nullptr) throw std::runtime_error("unable to create exclusive temporary GRIB2 stream payload");
    if (std::setvbuf(stream_, nullptr, _IONBF, 0) != 0) {
        std::fclose(stream_);
        stream_ = nullptr;
        std::error_code ignored;
        std::filesystem::remove(temporary_, ignored);
        throw std::runtime_error("unable to configure bounded GRIB2 stream buffering");
    }
}

Grib2StreamWriter::~Grib2StreamWriter() {
    if (stream_ != nullptr) std::fclose(stream_);
    std::error_code ignored;
    if (!temporary_.empty()) std::filesystem::remove(temporary_, ignored);
}

bool Grib2StreamWriter::begin_message(std::uint32_t field_id, std::uint32_t producer, std::uint64_t message_bytes, std::string &error) {
    if (failed_ || finished_ || active_ || index_.size() >= max_index_entries_ || message_bytes < 20 || message_bytes > max_message_bytes_ ||
        message_bytes > std::numeric_limits<std::uint64_t>::max() - payload_bytes_) {
        error = "GRIB2 stream message is invalid or exceeds a configured limit";
        return false;
    }
    field_id_ = field_id;
    producer_ = producer;
    expected_ = message_bytes;
    received_ = 0;
    header_.fill(0);
    trailer_.fill(0);
    active_ = true;
    return true;
}

bool Grib2StreamWriter::append_chunk(std::uint64_t offset, std::span<const std::byte> bytes, std::string &error) {
    if (failed_ || finished_ || !active_ || bytes.empty() || bytes.size() > max_chunk_bytes_ || offset != received_ ||
        bytes.size() > expected_ - received_) {
        error = "GRIB2 stream chunk is out of order or exceeds a configured limit";
        return false;
    }
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        const auto position = received_ + i;
        const auto value = std::to_integer<unsigned char>(bytes[i]);
        if (position < header_.size()) header_[static_cast<std::size_t>(position)] = value;
        if (position >= expected_ - trailer_.size()) trailer_[static_cast<std::size_t>(position - (expected_ - trailer_.size()))] = value;
    }
    if (std::fwrite(bytes.data(), 1, bytes.size(), stream_) != bytes.size()) {
        failed_ = true;
        error = "unable to write GRIB2 stream chunk";
        return false;
    }
    received_ += bytes.size();
    if (received_ == expected_) {
        std::uint64_t declared = 0;
        for (std::size_t i = 8; i < header_.size(); ++i) declared = (declared << 8) | header_[i];
        if (header_[0] != 'G' || header_[1] != 'R' || header_[2] != 'I' || header_[3] != 'B' || header_[7] != 2 || declared != expected_ ||
            trailer_ != std::array<unsigned char, 4>{'7', '7', '7', '7'}) {
            failed_ = true;
            error = "GRIB2 streamed message has an invalid indicator, length, or end marker";
            return false;
        }
        index_.push_back({field_id_, producer_, {payload_bytes_, expected_}});
        payload_bytes_ += expected_;
        active_ = false;
    }
    return true;
}

bool Grib2StreamWriter::finish(std::string &error) {
    if (failed_ || finished_ || active_ || index_.empty()) {
        error = "GRIB2 stream payload is incomplete or already closed";
        return false;
    }
    if (std::fflush(stream_) != 0) {
        failed_ = true;
        error = "unable to flush GRIB2 stream payload";
        return false;
    }
    const auto close_status = std::fclose(stream_);
    stream_ = nullptr;
    if (close_status != 0) {
        failed_ = true;
        error = "unable to close GRIB2 stream payload";
        return false;
    }
    std::error_code ec;
    std::filesystem::create_hard_link(temporary_, path_, ec);
    if (ec) {
        failed_ = true;
        error = "unable to publish immutable GRIB2 stream payload: " + ec.message();
        return false;
    }
    finished_ = true;
    std::filesystem::remove(temporary_, ec);
    return true;
}

Grib2MessageRange append_grib2_message_range(std::uint64_t &payload_bytes, std::uint64_t message_bytes) {
    if (message_bytes == 0) throw std::invalid_argument("GRIB2 step payload cannot index an empty message");
    if (message_bytes > std::numeric_limits<std::uint64_t>::max() - payload_bytes)
        throw std::overflow_error("GRIB2 step payload range overflows uint64");
    const Grib2MessageRange result{payload_bytes, message_bytes};
    payload_bytes += message_bytes;
    return result;
}

void preflight_grib2_message(std::size_t field_bytes, bool requires_repack, std::size_t max_encoded_buffer_bytes,
                             std::size_t max_codec_scratch_bytes) {
    // g2c is provisioned for raw float bytes + one byte per point + 8 KiB
    // section slack. Its float field copy and optional layout repack are
    // simultaneously live with that buffer.
    constexpr std::size_t section_slack = 8192;
    if (field_bytes == 0 || field_bytes > std::numeric_limits<std::size_t>::max() - section_slack)
        throw std::overflow_error("GRIB2 producer field size cannot be represented");
    const std::size_t points = field_bytes / sizeof(float);
    if (field_bytes % sizeof(float) != 0 || points > std::numeric_limits<std::size_t>::max() - field_bytes - section_slack)
        throw std::overflow_error("GRIB2 encoded buffer size cannot be represented");
    const std::size_t encoded_capacity = field_bytes + points + section_slack;
    if (encoded_capacity > max_encoded_buffer_bytes) throw std::runtime_error("GRIB2 producer encoded-message buffer exceeds its declared budget");
    if (field_bytes > (std::numeric_limits<std::size_t>::max() - encoded_capacity) / (requires_repack ? 2 : 1))
        throw std::overflow_error("GRIB2 codec scratch size cannot be represented");
    const std::size_t scratch = encoded_capacity + field_bytes * (requires_repack ? 2 : 1);
    if (scratch > max_codec_scratch_bytes) throw std::runtime_error("GRIB2 producer codec scratch exceeds its declared budget");
}

void require_complete_grib2_message(bool complete_message) {
    if (!complete_message) throw std::runtime_error("GRIB2 step aggregation does not support regional message fragments");
}

void validate_complete_grib2_message(std::span<const unsigned char> message) {
    constexpr std::size_t minimum_size = 20;
    if (message.size() < minimum_size || message[0] != 'G' || message[1] != 'R' || message[2] != 'I' || message[3] != 'B' || message[7] != 2 ||
        message[message.size() - 4] != '7' || message[message.size() - 3] != '7' || message[message.size() - 2] != '7' ||
        message[message.size() - 1] != '7') {
        throw std::runtime_error("GRIB2 step encoder returned an incomplete or invalid message");
    }
    std::uint64_t declared_size = 0;
    for (std::size_t i = 8; i < 16; ++i) declared_size = (declared_size << 8) | message[i];
    if (declared_size != message.size()) {
        throw std::runtime_error("GRIB2 indicator-section length does not match encoded message bytes");
    }
}

void append_grib2_field_messages(Grib2StepPayload &payload, const std::string &field, std::uint32_t producer,
                                 std::span<const std::vector<unsigned char>> messages, std::uint64_t max_payload_bytes, bool complete_field) {
    require_complete_grib2_message(complete_field);
    if (field.empty() || messages.empty()) throw std::invalid_argument("GRIB2 step fields require a name and at least one complete message");
    for (const auto &existing : payload.fields)
        if (existing.field == field && existing.producer == producer)
            throw std::invalid_argument("GRIB2 step field already has messages from this producer");

    std::uint64_t addition = 0;
    for (const auto &message : messages) {
        validate_complete_grib2_message(message);
        if (message.size() > std::numeric_limits<std::uint64_t>::max() - addition)
            throw std::overflow_error("GRIB2 producer messages exceed uint64 byte range");
        addition += static_cast<std::uint64_t>(message.size());
    }
    const auto current = static_cast<std::uint64_t>(payload.bytes.size());
    if (addition > std::numeric_limits<std::uint64_t>::max() - current || current + addition > max_payload_bytes ||
        current + addition > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("GRIB2 step payload exceeds its encoded-buffer budget");

    Grib2FieldRanges index;
    index.field = field;
    index.producer = producer;
    index.messages.reserve(messages.size());
    std::uint64_t offset = current;
    for (const auto &message : messages) index.messages.push_back(append_grib2_message_range(offset, message.size()));

    payload.bytes.reserve(static_cast<std::size_t>(current + addition));
    payload.fields.reserve(payload.fields.size() + 1);
    for (const auto &message : messages) payload.bytes.insert(payload.bytes.end(), message.begin(), message.end());
    payload.fields.push_back(std::move(index));
}

bool write_grib2_step_payload(const std::string &path, const Grib2StepPayload &payload, std::string &error) {
    if (path.empty() || payload.bytes.empty() || payload.fields.empty()) {
        error = "GRIB2 step payload requires a path, encoded messages, and a field index";
        return false;
    }
    if (payload.bytes.size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        error = "GRIB2 step payload exceeds the supported stream write size";
        return false;
    }
    std::vector<Grib2MessageRange> ranges;
    std::set<std::pair<std::string, std::uint32_t>> identities;
    for (const auto &field : payload.fields) {
        if (field.field.empty() || field.messages.empty() || !identities.emplace(field.field, field.producer).second) {
            error = "GRIB2 step payload has an empty or duplicate field/producer index";
            return false;
        }
        ranges.insert(ranges.end(), field.messages.begin(), field.messages.end());
    }
    std::sort(ranges.begin(), ranges.end(), [](const auto &a, const auto &b) { return a.offset < b.offset; });
    std::uint64_t next = 0;
    for (const auto &range : ranges) {
        if (range.offset != next || range.length == 0 || range.offset > payload.bytes.size() || range.length > payload.bytes.size() - range.offset) {
            error = "GRIB2 step message ranges do not cover the payload contiguously";
            return false;
        }
        try {
            validate_complete_grib2_message(
                std::span<const unsigned char>(payload.bytes.data() + range.offset, static_cast<std::size_t>(range.length)));
        } catch (const std::exception &e) {
            error = e.what();
            return false;
        }
        next += range.length;
    }
    if (next != payload.bytes.size()) {
        error = "GRIB2 step payload contains bytes outside its message index";
        return false;
    }

    static std::atomic<std::uint64_t> sequence{0};
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count() ^ sequence.fetch_add(1);
    const std::filesystem::path destination(path);
    const auto temporary = destination.string() + ".amio-tmp-" + std::to_string(nonce);
    std::error_code ec;
    try {
        {
            std::ofstream stream(temporary, std::ios::binary | std::ios::out | std::ios::trunc);
            if (!stream) throw std::runtime_error("Unable to create temporary GRIB2 step payload");
            stream.write(reinterpret_cast<const char *>(payload.bytes.data()), static_cast<std::streamsize>(payload.bytes.size()));
            stream.flush();
            if (!stream) throw std::runtime_error("Unable to write complete GRIB2 step payload");
        }
        // A hard link creates the final name atomically and fails if it
        // already exists, preserving immutable step IDs on retry/collision.
        std::filesystem::create_hard_link(temporary, destination, ec);
        if (ec) throw std::runtime_error("Unable to publish immutable GRIB2 step payload: " + ec.message());
        std::filesystem::remove(temporary, ec);
    } catch (const std::exception &e) {
        std::error_code cleanup_error;
        std::filesystem::remove(temporary, cleanup_error);
        error = e.what();
        return false;
    }
    return true;
}

bool read_grib2_field_messages(const std::string &path, const Grib2FieldRanges &field, std::size_t max_encoded_bytes,
                               std::vector<std::vector<unsigned char>> &messages, std::string &error) {
    messages.clear();
    if (path.empty() || field.field.empty() || field.messages.empty()) {
        error = "GRIB2 field read requires a payload path and indexed messages";
        return false;
    }
    std::error_code ec;
    const auto file_bytes = std::filesystem::file_size(path, ec);
    if (ec) {
        error = "Unable to stat GRIB2 step payload: " + ec.message();
        return false;
    }
    std::uint64_t total = 0;
    std::uint64_t previous_end = 0;
    for (const auto &range : field.messages) {
        if (range.length == 0 || range.offset < previous_end || range.offset > file_bytes || range.length > file_bytes - range.offset ||
            range.length > max_encoded_bytes - std::min<std::uint64_t>(total, max_encoded_bytes) ||
            range.length > std::numeric_limits<std::size_t>::max() ||
            range.length > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()) ||
            range.offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
            error = "GRIB2 indexed message range is invalid or exceeds the read budget";
            return false;
        }
        total += range.length;
        previous_end = range.offset + range.length;
    }
    try {
        std::ifstream stream(path, std::ios::binary);
        if (!stream) throw std::runtime_error("Unable to open GRIB2 step payload");
        std::vector<std::vector<unsigned char>> result;
        result.reserve(field.messages.size());
        for (const auto &range : field.messages) {
            std::vector<unsigned char> message(static_cast<std::size_t>(range.length));
            stream.seekg(static_cast<std::streamoff>(range.offset));
            stream.read(reinterpret_cast<char *>(message.data()), static_cast<std::streamsize>(message.size()));
            if (!stream) throw std::runtime_error("Unable to read complete indexed GRIB2 message");
            validate_complete_grib2_message(message);
            result.push_back(std::move(message));
        }
        messages = std::move(result);
    } catch (const std::exception &e) {
        messages.clear();
        error = e.what();
        return false;
    }
    return true;
}

amio_status_t append_grib2_catalog_payloads(const Grib2StepPayload &payload, const std::string &relative_path, CompletionRecord &record) {
    const std::filesystem::path relative(relative_path);
    if (payload.bytes.empty() || payload.fields.empty() || relative.is_absolute() || relative.empty() ||
        relative_path.find('\\') != std::string::npos || relative_path.find('\0') != std::string::npos || relative.begin()->string() != "payloads")
        return AMIO_ERR_INVALID_INPUT;
    for (const auto &component : relative)
        if (component == "." || component == ".." || component == "\\") return AMIO_ERR_INVALID_INPUT;
    std::set<std::string, std::less<>> seen;
    std::vector<Grib2MessageRange> all_ranges;
    for (const auto &field : payload.fields) {
        if (field.field.empty() || field.messages.empty() || !seen.insert(field.field).second ||
            std::find(record.present_fields.begin(), record.present_fields.end(), field.field) != record.present_fields.end())
            return AMIO_ERR_INVALID_INPUT;
        std::uint64_t previous_end = 0;
        for (const auto &range : field.messages) {
            if (range.length == 0 || range.offset < previous_end || range.offset > payload.bytes.size() ||
                range.length > payload.bytes.size() - range.offset)
                return AMIO_ERR_INVALID_INPUT;
            try {
                validate_complete_grib2_message(
                    std::span<const unsigned char>(payload.bytes.data() + range.offset, static_cast<std::size_t>(range.length)));
            } catch (...) {
                return AMIO_ERR_INVALID_INPUT;
            }
            previous_end = range.offset + range.length;
            all_ranges.push_back(range);
        }
    }
    std::sort(all_ranges.begin(), all_ranges.end(), [](const auto &a, const auto &b) { return a.offset < b.offset; });
    std::uint64_t next = 0;
    for (const auto &range : all_ranges) {
        if (range.offset != next) return AMIO_ERR_INVALID_INPUT;
        next += range.length;
    }
    if (next != payload.bytes.size()) return AMIO_ERR_INVALID_INPUT;
    for (const auto &field : payload.fields) {
        record.present_fields.push_back(field.field);
        PayloadDescriptor descriptor;
        descriptor.field = field.field;
        descriptor.relative_path = relative_path;
        descriptor.bytes = payload.bytes.size();
        descriptor.producer = field.producer;
        descriptor.message_ranges.reserve(field.messages.size());
        for (const auto &range : field.messages) descriptor.message_ranges.push_back({range.offset, range.length});
        record.payloads.push_back(std::move(descriptor));
    }
    return AMIO_OK;
}

void validate_grib2_representability(const Grib2Settings &s, std::int64_t ni, std::int64_t nj) {
    constexpr auto low = static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::min());
    constexpr auto high = static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max());
    if (ni <= 0 || nj <= 0 || ni > high / nj) throw std::runtime_error("GRIB2 grid dimensions are not representable in the encoder");
    const std::int64_t values[] = {s.discipline,
                                   s.center,
                                   s.subcenter,
                                   s.master_table_version,
                                   s.local_table_version,
                                   s.significance_of_ref_time,
                                   s.production_status,
                                   s.type_of_data,
                                   s.gdt_number,
                                   s.lat_first,
                                   s.lon_first,
                                   s.lat_last,
                                   s.lon_last,
                                   s.pdt_number,
                                   s.parameter_category,
                                   s.parameter_number,
                                   s.type_of_first_fixed_surface,
                                   s.scale_factor_first_surface,
                                   s.scaled_value_first_surface,
                                   s.forecast_time,
                                   s.indicator_of_unit_of_time,
                                   s.decimal_scale_factor,
                                   s.chemical_constituent_type,
                                   s.aerosol_type,
                                   s.size_dist_param_first,
                                   s.size_dist_param_second,
                                   s.optical_property_type,
                                   s.wavelength_first_nm,
                                   s.wavelength_last_nm,
                                   s.ensemble_perturbation_number,
                                   s.statistical_process,
                                   s.time_range_unit,
                                   s.time_range_length,
                                   s.number_of_time_range_specs,
                                   s.total_missing_from_statistical_process,
                                   s.n_parallel};
    for (auto value : values)
        if (value < low || value > high) throw std::runtime_error("GRIB2 grid/product metadata is not representable in the encoder");
    if (s.gdt_number != 0 && s.gdt_number != 40) throw std::runtime_error("GRIB2 step uses an unsupported grid definition template");
    switch (s.pdt_number) {
        case 0:
        case 8:
        case 40:
        case 44:
        case 45:
        case 46:
        case 48:
        case 49:
            break;
        default:
            throw std::runtime_error("GRIB2 step uses an unsupported product definition template");
    }
}

}  // namespace amio::detail
