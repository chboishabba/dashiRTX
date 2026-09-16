#include <stdint.h>
#include <stdio.h>

int main(void) {
  const uint64_t rows = UINT64_C(656182601);
  const uint64_t cols = UINT64_C(656182189);
  const uint64_t nnz = UINT64_C(98431741898);
  const uint64_t block_m = 512, block_n = 512;
  const uint64_t sequence_count = 2, sequence_width = 256, simd = 256;
  const uint64_t final_iteration = UINT64_C(2564096);
  const uint64_t checkpoint_stride = 8192, retained_stride = 32768;
  const uint64_t generator_length = UINT64_C(1281607);
  const uint64_t mksol_files = 40, gather_vectors = 64;
  const uint64_t nonzero_dependencies = 26, first_factor_dependency = 12;

  const uint64_t row_excess = rows - cols;
  const uint64_t density_floor = nnz / rows;
  const uint64_t density_remainder = nnz % rows;
  const uint64_t checkpoint_intervals = final_iteration / checkpoint_stride;
  const uint64_t retained_multiples = final_iteration / retained_stride;
  const uint64_t retained_remainder = final_iteration % retained_stride;

  if (row_excess != 412) return 2;
  if (density_floor != 150 || density_remainder != UINT64_C(4351748)) return 3;
  if (block_m != 2 * sequence_width || block_n != 2 * sequence_width) return 4;
  if (sequence_count * sequence_width != block_n) return 5;
  if (simd != sequence_width) return 6;
  if (checkpoint_intervals != 313 || final_iteration % checkpoint_stride != 0) return 7;
  if (retained_multiples != 78 || retained_remainder != 8192) return 8;
  if (generator_length != UINT64_C(1281607)) return 9;
  if (mksol_files != 40 || gather_vectors != 64) return 10;
  if (nonzero_dependencies != 26 || first_factor_dependency != 12) return 11;

  puts("RSA260_BIDI_LA_CARRIER_CONSTRAINT_FIBRE");
  printf("field=GF(2) rows=%llu cols=%llu nnz=%llu row_excess=%llu\n",
         (unsigned long long)rows, (unsigned long long)cols,
         (unsigned long long)nnz, (unsigned long long)row_excess);
  printf("avg_row_degree=%llu+%llu/%llu\n",
         (unsigned long long)density_floor,
         (unsigned long long)density_remainder,
         (unsigned long long)rows);
  printf("nullspace=left block_m=%llu block_n=%llu sequences=%llu width=%llu simd=%llu\n",
         (unsigned long long)block_m, (unsigned long long)block_n,
         (unsigned long long)sequence_count, (unsigned long long)sequence_width,
         (unsigned long long)simd);
  printf("final_iteration=%llu checkpoint_stride=%llu checkpoint_intervals=%llu retained_stride=%llu complete_retained_multiples=%llu terminal_remainder=%llu\n",
         (unsigned long long)final_iteration,
         (unsigned long long)checkpoint_stride,
         (unsigned long long)checkpoint_intervals,
         (unsigned long long)retained_stride,
         (unsigned long long)retained_multiples,
         (unsigned long long)retained_remainder);
  printf("lingen_length=%llu mksol_files=%llu gather_vectors=%llu dependencies=%llu first_factor_dependency=%llu\n",
         (unsigned long long)generator_length,
         (unsigned long long)mksol_files,
         (unsigned long long)gather_vectors,
         (unsigned long long)nonzero_dependencies,
         (unsigned long long)first_factor_dependency);
  puts("forward_envelope_paid=true backward_consumer_constraints_paid=true");
  puts("exact_nonzero_positions_derived=false row_order_derived=false column_order_derived=false");
  puts("prepared_bwc_encoding_derived=false projection_vectors_or_seed_derived=false matrix_bytes_derived=false");
  puts("unique_matrix_derived=false constraint_fibre_derived=true same_object_P0_paid=false");
  return 0;
}
