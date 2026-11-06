#include <ESMC.h>

#include <iostream>
int main() {
    int rc;
    rc = ESMC_Initialize(nullptr, ESMC_InitArgLogKindFlag(ESMC_LOGKIND_NONE), ESMC_ArgLast);
    if (rc != ESMF_SUCCESS)
        return rc;
    // Ensure these exact entry points resolve at link time.
    volatile auto create = &ESMC_MeshCreate;
    volatile auto nodes = &ESMC_MeshAddNodes;
    volatile auto elements = &ESMC_MeshAddElements;
    volatile auto field = &ESMC_FieldCreateMeshTypeKind;
    volatile auto store = &ESMC_FieldRegridStore;
    volatile auto apply = &ESMC_FieldRegrid;
    volatile auto area = &ESMC_FieldRegridGetArea;
    volatile auto release = &ESMC_FieldRegridRelease;
    (void)create;
    (void)nodes;
    (void)elements;
    (void)field;
    (void)store;
    (void)apply;
    (void)area;
    (void)release;
    std::cout << "{\"direct_element_arities\":[3,4],\"variable_arity\":false,\"reason\":\"Adapter "
                 "restricts to documented TRI/QUAD constants\"}\n";
    return ESMC_Finalize();
}
