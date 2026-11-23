#include "utility_fixtures.hpp"

#include <fstream>

bool write_amio_utility_fixtures(const std::filesystem::path &directory) {
    std::error_code ec;
    std::filesystem::create_directories(directory, ec);
    if (ec) return false;
    {
        std::ofstream out(directory / "zarr-copy.yaml", std::ios::binary | std::ios::trunc);
        if (!out) return false;
        out << "source: source-zarr\n"
               "destination: copy-zarr\n"
               "format: zarr3\n"
               "steps: 0:3\n"
               "fields: temperature,latitude,longitude\n"
               "expected: copy only when the configured Zarr backend is available; otherwise exit 3 before destination creation\n";
    }
    {
        std::ofstream out(directory / "stalled-run.json", std::ios::binary | std::ios::trunc);
        if (!out) return false;
        out << "{\"api_version\":2,\"topology\":{\"execution_mode\":\"mpi_stream\",\"progress_mode\":\"caller\",\"rank\":0,\"rank_count\":2},"
               "\"scope_count\":1,\"evicted_scope_count\":0,\"dropped_sample_count\":2,"
               "\"cumulative_counters\":{\"accepted\":4,\"rejected\":0,\"completed\":1,\"failed\":0,\"bytes_accepted\":8192,\"bytes_written\":4096,"
               "\"bytes_published\":4096,\"source_released_bytes\":8192,\"transport_received_bytes\":4096,\"transport_published_bytes\":4096,\"view_"
               "bytes\":0,\"stream_skips\":0,\"stream_delivered_bytes\":4096,\"stream_released_bytes\":0,\"stream_disconnects\":0,\"operator_input_"
               "bytes\":0,\"operator_output_bytes\":0},"
               "\"scopes\":[{\"context_epoch\":1,\"dataset_id\":7,\"step_id\":3,\"field_id\":1,\"role\":1,\"participant\":0,\"counters\":[1,0,0,0,"
               "8192,0,0,8192,0,4096,0,0,0,0,0,0,0],\"resource_current\":[0,0,0,0,0,0,0,0,0,0,256,1],\"resource_high_water\":[0,0,0,0,0,0,0,0,0,0,"
               "256,1],\"resource_ceilings\":[0,0,0,0,0,0,0,0,0,0,4096,4],\"first_error\":{\"status\":0,\"message\":\"\"}}],"
               "\"first_error\":{\"status\":0,\"message\":\"\"},\"events\":[{\"kind\":5,\"context_epoch\":1,\"dataset_id\":7,\"step_id\":3,\"field_"
               "id\":1,\"role\":1,\"participant\":0,\"message\":\"credential=must-not-be-echoed\"}]}";
    }
    return true;
}
