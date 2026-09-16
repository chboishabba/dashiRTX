#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum ArtifactKind {
    ART_METADATA_ONLY = 0,
    ART_MATRIX_OR_PREP = 1,
    ART_KRYLOV_CHECKPOINT = 2,
    ART_LINGEN_GENERATOR = 3,
    ART_MKSOL_COLLECTION = 4,
    ART_GATHER_KERNEL_COLLECTION = 5
};

enum AdmissionDepth {
    DEPTH_ENVELOPE_ONLY = 0,
    DEPTH_MATRIX_INPUT = 1,
    DEPTH_KRYLOV_STATE = 2,
    DEPTH_LINGEN_OUTPUT = 3,
    DEPTH_MKSOL_OUTPUT = 4,
    DEPTH_GATHER_OUTPUT = 5
};

typedef struct {
    enum ArtifactKind kind;
    int bytes_acquired;
    int digest_bound;
    int source_identity_bound;
    int same_run_bound;
    uint64_t rows;
    uint64_t cols;
    uint64_t nnz;
    uint64_t sequence_width;
    uint64_t sequence_index_lo;
    uint64_t sequence_index_hi;
    uint64_t iteration;
    uint64_t generator_length;
    uint64_t file_count;
    uint64_t vector_count;
} Candidate;

typedef struct {
    int admitted;
    enum AdmissionDepth depth;
    int retained_for_mksol_paid;
    int reconstructs_matrix;
    const char *reason;
} Decision;

static int identity_paid(const Candidate *c) {
    return c->bytes_acquired && c->digest_bound && c->source_identity_bound && c->same_run_bound;
}

static Decision reject(const char *reason) {
    Decision d = {0, DEPTH_ENVELOPE_ONLY, 0, 0, reason};
    return d;
}

static Decision admit(const Candidate *c) {
    const uint64_t ROWS = 656182601ULL;
    const uint64_t COLS = 656182189ULL;
    const uint64_t NNZ  = 98431741898ULL;
    const uint64_t FINAL = 2564096ULL;
    const uint64_t CKPT = 8192ULL;
    const uint64_t RETAIN = 32768ULL;
    const uint64_t GEN = 1281607ULL;

    if (c->kind == ART_METADATA_ONLY)
        return reject("metadata is execution-envelope evidence, not same-object bytes");
    if (!identity_paid(c))
        return reject("bytes/digest/source/run identity not all bound");

    if (c->kind == ART_MATRIX_OR_PREP) {
        if (c->rows != ROWS || c->cols != COLS || c->nnz != NNZ)
            return reject("matrix shape/nonzero count mismatch");
        Decision d = {1, DEPTH_MATRIX_INPUT, 0, 1, "same-object matrix/prep candidate admitted"};
        return d;
    }

    if (c->kind == ART_KRYLOV_CHECKPOINT) {
        if (c->sequence_width != 256ULL)
            return reject("checkpoint width mismatch");
        if (!((c->sequence_index_lo == 0ULL && c->sequence_index_hi == 256ULL) ||
              (c->sequence_index_lo == 256ULL && c->sequence_index_hi == 512ULL)))
            return reject("checkpoint sequence identity mismatch");
        if (c->iteration == 0ULL || c->iteration > FINAL || c->iteration % CKPT != 0ULL)
            return reject("checkpoint iteration is off the public 8192-step lattice");
        Decision d = {1, DEPTH_KRYLOV_STATE,
                      (c->iteration % RETAIN == 0ULL),
                      0,
                      "same-object Krylov checkpoint candidate admitted"};
        return d;
    }

    if (c->kind == ART_LINGEN_GENERATOR) {
        if (c->generator_length != GEN)
            return reject("lingen generator length mismatch");
        Decision d = {1, DEPTH_LINGEN_OUTPUT, 0, 0, "same-object lingen generator candidate admitted"};
        return d;
    }

    if (c->kind == ART_MKSOL_COLLECTION) {
        if (c->file_count != 40ULL)
            return reject("mksol collection must bind all 40 reported partial-solution files");
        Decision d = {1, DEPTH_MKSOL_OUTPUT, 0, 0, "same-object complete mksol collection admitted"};
        return d;
    }

    if (c->kind == ART_GATHER_KERNEL_COLLECTION) {
        if (c->vector_count != 64ULL)
            return reject("gather collection must bind all 64 reported kernel vectors");
        Decision d = {1, DEPTH_GATHER_OUTPUT, 0, 0, "same-object gather collection admitted"};
        return d;
    }

    return reject("unknown artifact kind");
}

static int expect(const char *name, Candidate c, int admitted, enum AdmissionDepth depth, int retained) {
    Decision d = admit(&c);
    printf("%s admitted=%d depth=%d retained=%d matrix=%d reason=%s\n",
           name, d.admitted, (int)d.depth, d.retained_for_mksol_paid,
           d.reconstructs_matrix, d.reason);
    return d.admitted == admitted && d.depth == depth && d.retained_for_mksol_paid == retained;
}

int main(void) {
    int ok = 1;
    Candidate base = {0};
    base.bytes_acquired = 1;
    base.digest_bound = 1;
    base.source_identity_bound = 1;
    base.same_run_bound = 1;

    Candidate metadata = {0};
    metadata.kind = ART_METADATA_ONLY;
    ok &= expect("metadata", metadata, 0, DEPTH_ENVELOPE_ONLY, 0);

    Candidate matrix = base;
    matrix.kind = ART_MATRIX_OR_PREP;
    matrix.rows = 656182601ULL; matrix.cols = 656182189ULL; matrix.nnz = 98431741898ULL;
    ok &= expect("matrix", matrix, 1, DEPTH_MATRIX_INPUT, 0);

    Candidate checkpoint = base;
    checkpoint.kind = ART_KRYLOV_CHECKPOINT;
    checkpoint.sequence_width = 256ULL;
    checkpoint.sequence_index_lo = 0ULL; checkpoint.sequence_index_hi = 256ULL;
    checkpoint.iteration = 32768ULL;
    ok &= expect("checkpoint_retained_lattice", checkpoint, 1, DEPTH_KRYLOV_STATE, 1);
    checkpoint.iteration = 2564096ULL;
    ok &= expect("checkpoint_terminal", checkpoint, 1, DEPTH_KRYLOV_STATE, 0);
    checkpoint.iteration = 12345ULL;
    ok &= expect("checkpoint_off_lattice", checkpoint, 0, DEPTH_ENVELOPE_ONLY, 0);

    Candidate generator = base;
    generator.kind = ART_LINGEN_GENERATOR;
    generator.generator_length = 1281607ULL;
    ok &= expect("generator", generator, 1, DEPTH_LINGEN_OUTPUT, 0);

    Candidate mksol = base;
    mksol.kind = ART_MKSOL_COLLECTION;
    mksol.file_count = 40ULL;
    ok &= expect("mksol", mksol, 1, DEPTH_MKSOL_OUTPUT, 0);

    Candidate gather = base;
    gather.kind = ART_GATHER_KERNEL_COLLECTION;
    gather.vector_count = 64ULL;
    ok &= expect("gather", gather, 1, DEPTH_GATHER_OUTPUT, 0);

    Candidate wrong = gather;
    wrong.vector_count = 63ULL;
    ok &= expect("gather_wrong_count", wrong, 0, DEPTH_ENVELOPE_ONLY, 0);

    printf("ok=%d\n", ok);
    return ok ? 0 : 1;
}
