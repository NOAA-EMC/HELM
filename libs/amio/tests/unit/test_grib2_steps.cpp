#include "drivers/grib2/grib2_driver.hpp"
#include "drivers/grib2/grib2_steps.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace amio::detail;

namespace {

template <class F> bool throws(F &&function) {
    try { function(); } catch (const std::exception &) { return true; }
    return false;
}

void test_complete_message_ranges_are_ordered_and_checked() {
    std::uint64_t size = 0;
    const auto first = append_grib2_message_range(size, 13);
    const auto second = append_grib2_message_range(size, 7);
    assert(first.offset == 0 && first.length == 13);
    assert(second.offset == 13 && second.length == 7);
    assert(size == 20);
    assert(throws([&] { (void)append_grib2_message_range(size, 0); }));
    size = std::numeric_limits<std::uint64_t>::max() - 2;
    assert(throws([&] { (void)append_grib2_message_range(size, 3); }));
}

void test_encoded_and_scratch_budgets_preflight_before_encoding() {
    preflight_grib2_message(400, false, 9000, 10000);
    assert(throws([&] { preflight_grib2_message(400, false, 400, 10000); }));
    assert(throws([&] { preflight_grib2_message(400, true, 9000, 1000); }));
    assert(throws([&] { preflight_grib2_message(0, false, 9000, 10000); }));
    assert(throws([&] { preflight_grib2_message(401, false, 9000, 10000); }));
    assert(throws([&] { require_complete_grib2_message(false); }));
}

void test_grib2_complete_message_validation() {
    std::vector<unsigned char> message(20, 0);
    message[0] = 'G'; message[1] = 'R'; message[2] = 'I'; message[3] = 'B';
    message[7] = 2;
    message[15] = static_cast<unsigned char>(message.size());
    message[16] = '7'; message[17] = '7'; message[18] = '7'; message[19] = '7';
    validate_complete_grib2_message(message);
    message[15] = 19;
    assert(throws([&] { validate_complete_grib2_message(message); }));
    message[15] = 20;
    message[19] = 'X';
    assert(throws([&] { validate_complete_grib2_message(message); }));
    assert(throws([&] { validate_complete_grib2_message(std::span<const unsigned char>(message.data(), 19)); }));
}

std::vector<unsigned char> valid_message(unsigned char marker) {
    std::vector<unsigned char> message(21, 0);
    message[0] = 'G'; message[1] = 'R'; message[2] = 'I'; message[3] = 'B';
    message[7] = 2;
    message[15] = static_cast<unsigned char>(message.size());
    message[16] = marker;
    message[17] = '7'; message[18] = '7'; message[19] = '7'; message[20] = '7';
    return message;
}

void test_step_payload_appends_complete_messages_and_indexes_ranges() {
    Grib2StepPayload payload;
    const std::array<std::vector<unsigned char>, 2> temperature = {valid_message(1), valid_message(2)};
    append_grib2_field_messages(payload, "temperature", 4, temperature, 1000);
    assert(payload.bytes.size() == 42);
    assert(payload.fields.size() == 1);
    assert(payload.fields[0].field == "temperature" && payload.fields[0].producer == 4);
    assert(payload.fields[0].messages.size() == 2);
    assert(payload.fields[0].messages[0].offset == 0 && payload.fields[0].messages[0].length == 21);
    assert(payload.fields[0].messages[1].offset == 21 && payload.fields[0].messages[1].length == 21);
    assert(payload.bytes[16] == 1 && payload.bytes[37] == 2); // producer order is retained

    const std::array<std::vector<unsigned char>, 1> pressure = {valid_message(3)};
    append_grib2_field_messages(payload, "pressure", 1, pressure, 1000);
    assert(payload.fields.size() == 2 && payload.fields[1].messages[0].offset == 42);
    const auto original_bytes = payload.bytes;
    const auto original_fields = payload.fields.size();
    assert(throws([&] { append_grib2_field_messages(payload, "temperature", 4, pressure, 1000); }));
    assert(throws([&] { append_grib2_field_messages(payload, "humidity", 2, pressure, 59); }));
    assert(throws([&] { append_grib2_field_messages(payload, "humidity", 2, pressure, 1000, false); }));
    auto invalid = valid_message(9);
    invalid[15] = 19;
    const std::array<std::vector<unsigned char>, 1> malformed = {invalid};
    assert(throws([&] { append_grib2_field_messages(payload, "humidity", 2, malformed, 1000); }));
    assert(payload.bytes == original_bytes && payload.fields.size() == original_fields);
}

void test_immutable_step_payload_file_and_indexed_field_reads() {
    Grib2StepPayload payload;
    const std::array<std::vector<unsigned char>, 2> temperature = {valid_message(4), valid_message(5)};
    const std::array<std::vector<unsigned char>, 1> pressure = {valid_message(6)};
    append_grib2_field_messages(payload, "temperature", 0, temperature, 1000);
    append_grib2_field_messages(payload, "pressure", 1, pressure, 1000);
    const auto path = std::filesystem::temp_directory_path() / "amio-grib2-step-index-test.bin";
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::string error;
    assert(write_grib2_step_payload(path.string(), payload, error));
    assert(!write_grib2_step_payload(path.string(), payload, error)); // immutable step IDs are create-only

    std::vector<std::vector<unsigned char>> messages;
    assert(read_grib2_field_messages(path.string(), payload.fields[0], 100, messages, error));
    assert(messages == std::vector<std::vector<unsigned char>>(temperature.begin(), temperature.end()));
    assert(!read_grib2_field_messages(path.string(), payload.fields[0], 20, messages, error));
    assert(messages.empty());

    auto corrupt_index = payload.fields[0];
    corrupt_index.messages[1].offset = 0;
    assert(!read_grib2_field_messages(path.string(), corrupt_index, 100, messages, error));
    std::filesystem::remove(path, ec);
}

void test_grib2_message_index_is_added_to_the_completion_record() {
    Grib2StepPayload payload;
    const std::array<std::vector<unsigned char>, 2> temperature = {valid_message(7), valid_message(8)};
    append_grib2_field_messages(payload, "temperature", 2, temperature, 1000);
    CompletionRecord record;
    record.dataset_uuid = "dataset-uuid-0001";
    record.schema_digest = "schema-digest-1";
    record.required_fields = {"temperature"};
    assert(append_grib2_catalog_payloads(payload, "payloads/steps/0001.grib2", record) == AMIO_OK);
    assert(record.present_fields == std::vector<std::string>({"temperature"}));
    assert(record.payloads.size() == 1 && record.payloads[0].producer == 2);

    std::string encoded;
    assert(encode_commit_document(record, encoded) == AMIO_OK);
    CompletionRecord decoded;
    assert(parse_commit_document(encoded, decoded) == AMIO_OK);
    assert(decoded.payloads[0].message_ranges.size() == 2);
    assert(decoded.payloads[0].message_ranges[0].offset == 0);
    assert(decoded.payloads[0].message_ranges[1].offset == 21);
    assert(append_grib2_catalog_payloads(payload, "payloads/steps/0001.grib2", record) == AMIO_ERR_INVALID_INPUT);
    assert(record.payloads.size() == 1);
    CompletionRecord invalid_path_record;
    assert(append_grib2_catalog_payloads(payload, "payloads/../outside.grib2", invalid_path_record) == AMIO_ERR_INVALID_INPUT);
}

void test_grid_and_product_settings_are_representable() {
    Grib2Settings settings;
    validate_grib2_representability(settings, 360, 181);
    settings.gdt_number = 1;
    assert(throws([&] { validate_grib2_representability(settings, 360, 181); }));
    settings.gdt_number = 0;
    settings.pdt_number = 999;
    assert(throws([&] { validate_grib2_representability(settings, 360, 181); }));
    settings.pdt_number = 0;
    assert(throws([&] { validate_grib2_representability(settings, 0, 181); }));
    settings.center = std::numeric_limits<std::int64_t>::max();
    assert(throws([&] { validate_grib2_representability(settings, 360, 181); }));
}

} // namespace

int main() {
    test_complete_message_ranges_are_ordered_and_checked();
    test_encoded_and_scratch_budgets_preflight_before_encoding();
    test_grib2_complete_message_validation();
    test_step_payload_appends_complete_messages_and_indexes_ranges();
    test_immutable_step_payload_file_and_indexed_field_reads();
    test_grib2_message_index_is_added_to_the_completion_record();
    test_grid_and_product_settings_are_representable();
}
