#include "amio/amio_evolution.h"

#include <mpi.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#define check(value) do { if (!(value)) { std::fprintf(stderr, "GRIB2 operator check failed at line %d: %s\n", __LINE__, #value); std::abort(); } } while (false)

int main(int argc, char **argv) {
    int provided = MPI_THREAD_SINGLE;
    check(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) == MPI_SUCCESS);
    check(provided == MPI_THREAD_MULTIPLE);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    check(size == 1 && rank == 0);

    constexpr std::array<std::int32_t,1> producers{0};
    constexpr char config[] = "drt: libaec\ngrib2:\n  decimal_scale_factor: 0\n";
    amio_context_descriptor_t context_desc{};
    context_desc.struct_size=sizeof(context_desc); context_desc.version=AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    context_desc.execution_mode=AMIO_EXECUTION_PERSISTENT_DIRECT; context_desc.progress_mode=AMIO_PROGRESS_CALLER;
    context_desc.rank_count=1; context_desc.producer_count=1; context_desc.producers=producers.data();
    context_desc.fortran_communicator=static_cast<std::int64_t>(MPI_Comm_c2f(MPI_COMM_WORLD));
    context_desc.canonical_configuration=config; context_desc.configuration_bytes=sizeof(config)-1;
    amio_context_handle context=nullptr;
    check(amio_init_ex(&context_desc,&context)==AMIO_OK);

    const auto root=std::filesystem::temp_directory_path()/"amio-operator-grib2";
    std::error_code ec; std::filesystem::remove_all(root,ec);
    const auto root_text=root.string();
    amio_dataset_descriptor_t dataset_desc{};
    dataset_desc.struct_size=sizeof(dataset_desc); dataset_desc.version=AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    dataset_desc.location=root_text.data(); dataset_desc.location_bytes=root_text.size(); dataset_desc.format=AMIO_FORMAT_GRIB2;
    dataset_desc.budget_bytes=1024*1024; dataset_desc.budget_descriptors=64;
    dataset_desc.control_budget_bytes=64*1024; dataset_desc.control_budget_descriptors=16;
    amio_dataset_handle dataset=nullptr;
    check(amio_open_dataset_ex(context,&dataset_desc,&dataset)==AMIO_OK);

    constexpr char dim_name[]="x", aec_name[]="field_aec", jpeg_name[]="field_jpeg";
    amio_dimension_descriptor_t dimension{};
    dimension.struct_size=sizeof(dimension); dimension.version=AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    dimension.name=dim_name; dimension.name_bytes=sizeof(dim_name)-1; dimension.extent=4;
    constexpr char codec_key[]="amio.codec", aec_codec[]="libaec", jpeg_codec[]="lossless_jpeg2000";
    auto make_variable = [&](const char *name, const char *codec, amio_variable_descriptor_t &variable,
                             amio_attribute_descriptor_t &attribute) {
        attribute.struct_size=sizeof(attribute); attribute.version=AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        attribute.name=codec_key; attribute.name_bytes=sizeof(codec_key)-1; attribute.type=AMIO_ATTRIBUTE_UTF8;
        attribute.element_count=1; attribute.value=codec; attribute.value_bytes=std::char_traits<char>::length(codec);
        variable.struct_size=sizeof(variable); variable.version=AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        variable.name=name; variable.name_bytes=std::char_traits<char>::length(name);
        variable.source_dtype=AMIO_DTYPE_F32; variable.stored_dtype=AMIO_DTYPE_F32; variable.required=1;
        variable.dimension_count=1; variable.dimensions=&dimension; variable.attribute_count=1; variable.attributes=&attribute;
    };
    amio_variable_descriptor_t variables[2]{};
    amio_attribute_descriptor_t attributes[2]{};
    make_variable(aec_name,aec_codec,variables[0],attributes[0]);
    make_variable(jpeg_name,jpeg_codec,variables[1],attributes[1]);
    amio_schema_descriptor_t schema_desc{};
    schema_desc.struct_size=sizeof(schema_desc); schema_desc.version=AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    schema_desc.variable_count=2; schema_desc.variables=variables;
    amio_schema_handle schema=nullptr;
    check(amio_define_schema(dataset,&schema_desc,&schema)==AMIO_OK);
    amio_step_descriptor_t step_desc{};
    step_desc.struct_size=sizeof(step_desc); step_desc.version=AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    step_desc.step_id=0; step_desc.model_time=1.0; step_desc.schema=schema;
    amio_step_handle step=nullptr;
    check(amio_begin_step(dataset,&step_desc,&step)==AMIO_OK);
    constexpr std::array<float,4> values{1.0f,2.0f,3.0f,4.0f};
    amio_operation_handle operations[2]{};
    check(amio_put_field(step,aec_name,sizeof(aec_name)-1,values.data(),sizeof(values),&operations[0])==AMIO_OK);
    check(amio_put_field(step,jpeg_name,sizeof(jpeg_name)-1,values.data(),sizeof(values),&operations[1])==AMIO_OK);
    check(amio_end_step(step)==AMIO_OK);
    std::array<float,4> aec_read{}, jpeg_read{};
    std::size_t bytes=0;
    check(amio_read_step_field(dataset,0,aec_name,sizeof(aec_name)-1,nullptr,aec_read.data(),sizeof(aec_read),&bytes)==AMIO_OK);
    check(bytes==sizeof(aec_read) && aec_read==values);
    check(amio_read_step_field(dataset,0,jpeg_name,sizeof(jpeg_name)-1,nullptr,jpeg_read.data(),sizeof(jpeg_read),&bytes)==AMIO_OK);
    check(bytes==sizeof(jpeg_read) && jpeg_read==values);

    check(amio_release_operation(operations[0])==AMIO_OK);
    check(amio_release_operation(operations[1])==AMIO_OK);
    check(amio_release_step_handle(step)==AMIO_OK);
    check(amio_release_schema(schema)==AMIO_OK);
    check(amio_close_dataset_ex(dataset)==AMIO_OK);
    check(amio_release_context(context)==AMIO_OK);
    std::filesystem::remove_all(root,ec);
    MPI_Finalize();
}
