#include <stdint.h>
#include <stdio.h>

int main(void) {
    const uint64_t rows = 656182601ULL;
    const uint64_t cols = 656182189ULL;
    const uint64_t nnz  = 98431741898ULL;
    const uint64_t final_iter = 2564096ULL;
    const uint64_t checkpoint_stride = 8192ULL;
    const uint64_t retained_stride = 32768ULL;
    const uint64_t sequences = 2ULL;
    const uint64_t width = 256ULL;
    const uint64_t m = 512ULL, n = 512ULL;
    const uint64_t generator_length = 1281607ULL;
    const uint64_t mksol_ranges = 40ULL;
    const uint64_t kernel_vectors = 64ULL;
    const uint64_t dependencies = 26ULL;
    const uint64_t first_factor_dependency = 12ULL;

    if (final_iter % checkpoint_stride != 0) return 2;
    const uint64_t checkpoint_intervals = final_iter / checkpoint_stride;
    const uint64_t regular_retained = final_iter / retained_stride;
    const uint64_t retained_tail = final_iter % retained_stride;

    printf("rsa260_public_la_execution_envelope\n");
    printf("matrix=%llux%llu nnz=%llu density=150.0\n",
           (unsigned long long)rows,
           (unsigned long long)cols,
           (unsigned long long)nnz);
    printf("krylov m=%llu n=%llu sequences=%llu width=%llu final_iter=%llu\n",
           (unsigned long long)m,
           (unsigned long long)n,
           (unsigned long long)sequences,
           (unsigned long long)width,
           (unsigned long long)final_iter);
    printf("checkpoint_stride=%llu checkpoint_intervals=%llu retained_stride=%llu regular_retained=%llu retained_tail=%llu\n",
           (unsigned long long)checkpoint_stride,
           (unsigned long long)checkpoint_intervals,
           (unsigned long long)retained_stride,
           (unsigned long long)regular_retained,
           (unsigned long long)retained_tail);
    printf("lingen_generator_length=%llu mksol_ranges=%llu kernel_vectors=%llu dependencies=%llu first_factor_dependency=%llu\n",
           (unsigned long long)generator_length,
           (unsigned long long)mksol_ranges,
           (unsigned long long)kernel_vectors,
           (unsigned long long)dependencies,
           (unsigned long long)first_factor_dependency);

    /* Snowball boundary: these are public execution-envelope coordinates only. */
    printf("production_matrix_bytes_acquired=false\n");
    printf("production_checkpoint_bytes_acquired=false\n");
    printf("production_generator_bytes_acquired=false\n");
    printf("production_kernel_vector_bytes_acquired=false\n");
    printf("final_endpoint_retained_under_every_fourth_rule=unresolved\n");
    printf("allocation_hover_metadata_reconstructed=false\n");
    return 0;
}
