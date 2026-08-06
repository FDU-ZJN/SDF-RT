#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    APERTURE_BYTES = 0x100000,
    SDF_CTRL_WORD = 0x3ffe0,
    SDF_STATUS_WORD = 0x3ffe1,
    SDF_LOCAL_CELLS_WORD = 0x3ffe2,
    SDF_GLOBAL_ROWS_WORD = 0x3ffe3,
    SDF_LOCAL_ROWS_WORD = 0x3ffe4,
    SDF_MAPPING_WORDS_WORD = 0x3ffe5,
    SDF_DATA_WORD = 0x3ffe6,
    EXPECTED_GLOBAL_ROWS = 4096,
    EXPECTED_MAPPING_WORDS = 16384,
    IMAGE_HEADER_BYTES = 64
};

struct sdf_image {
    uint8_t *file_data;
    size_t file_size;
    const uint8_t *payload;
    uint32_t global_rows;
    uint32_t local_rows;
    uint32_t mapping_words;
    uint32_t local_cells;
    uint32_t global_words_per_row;
    uint32_t local_words_per_row;
    uint32_t payload_words;
};

static uint32_t read_le32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint32_t crc32_ieee(const uint8_t *data, size_t size) {
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (unsigned bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}

static int load_image(const char *path, struct sdf_image *image) {
    const int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "open %s: %s\n", path, strerror(errno));
        return -1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < IMAGE_HEADER_BYTES) {
        fprintf(stderr, "%s: invalid or truncated image\n", path);
        close(fd);
        return -1;
    }
    image->file_size = (size_t)st.st_size;
    image->file_data = malloc(image->file_size);
    if (!image->file_data) {
        fprintf(stderr, "allocating %zu image bytes failed\n", image->file_size);
        close(fd);
        return -1;
    }
    size_t done = 0;
    while (done < image->file_size) {
        const ssize_t count = read(fd, image->file_data + done, image->file_size - done);
        if (count <= 0) {
            fprintf(stderr, "read %s: %s\n", path, count < 0 ? strerror(errno) : "unexpected EOF");
            free(image->file_data);
            close(fd);
            return -1;
        }
        done += (size_t)count;
    }
    close(fd);

    const uint8_t expected_magic[8] = {'S', 'D', 'F', '9', 'I', 'M', 'G', 0};
    const uint32_t version = read_le32(image->file_data + 8);
    const uint32_t header_bytes = read_le32(image->file_data + 12);
    image->global_rows = read_le32(image->file_data + 16);
    image->local_rows = read_le32(image->file_data + 20);
    image->mapping_words = read_le32(image->file_data + 24);
    image->local_cells = read_le32(image->file_data + 28);
    image->global_words_per_row = read_le32(image->file_data + 32);
    image->local_words_per_row = read_le32(image->file_data + 36);
    image->payload_words = read_le32(image->file_data + 40);
    const uint32_t expected_crc = read_le32(image->file_data + 44);
    if (memcmp(image->file_data, expected_magic, sizeof(expected_magic)) != 0 ||
        version != 1 || header_bytes != IMAGE_HEADER_BYTES ||
        image->global_rows != EXPECTED_GLOBAL_ROWS ||
        image->mapping_words != EXPECTED_MAPPING_WORDS ||
        image->local_cells > 16384 || image->local_rows != image->local_cells ||
        image->global_words_per_row != 3 || image->local_words_per_row != 18) {
        fprintf(stderr, "%s: incompatible SDF image header\n", path);
        free(image->file_data);
        return -1;
    }
    const uint64_t expected_words =
        (uint64_t)image->global_rows * image->global_words_per_row +
        (uint64_t)image->local_rows * image->local_words_per_row +
        image->mapping_words;
    const uint64_t expected_size = (uint64_t)header_bytes + expected_words * 4u;
    if (expected_words != image->payload_words || expected_size != image->file_size) {
        fprintf(stderr, "%s: payload size does not match its header\n", path);
        free(image->file_data);
        return -1;
    }
    image->payload = image->file_data + header_bytes;
    const uint32_t actual_crc = crc32_ieee(image->payload, image->payload_words * 4u);
    if (actual_crc != expected_crc) {
        fprintf(stderr, "%s: CRC32 mismatch (expected %08" PRIx32 ", got %08" PRIx32 ")\n",
                path, expected_crc, actual_crc);
        free(image->file_data);
        return -1;
    }
    return 0;
}

static inline void mmio_write(volatile uint32_t *regs, uint32_t word, uint32_t value) {
    regs[word] = value;
    __sync_synchronize();
}

static inline uint32_t mmio_read(volatile uint32_t *regs, uint32_t word) {
    const uint32_t value = regs[word];
    __sync_synchronize();
    return value;
}

static void print_status(volatile uint32_t *regs) {
    const uint32_t status = mmio_read(regs, SDF_STATUS_WORD);
    fprintf(stderr,
            "status=0x%08" PRIx32 " active=%u ready=%u error=%u phase=%u "
            "global=%" PRIu32 " local=%" PRIu32 " mapping=%" PRIu32 "\n",
            status, status & 1u, (status >> 1) & 1u, (status >> 2) & 1u,
            (status >> 3) & 3u,
            mmio_read(regs, SDF_GLOBAL_ROWS_WORD),
            mmio_read(regs, SDF_LOCAL_ROWS_WORD),
            mmio_read(regs, SDF_MAPPING_WORDS_WORD));
}

static int require_progress(volatile uint32_t *regs, uint32_t phase,
                            uint32_t global_rows, uint32_t local_rows,
                            uint32_t mapping_words) {
    const uint32_t status = mmio_read(regs, SDF_STATUS_WORD);
    if ((status & 1u) == 0 || (status & 4u) != 0 || ((status >> 3) & 3u) != phase ||
        mmio_read(regs, SDF_GLOBAL_ROWS_WORD) != global_rows ||
        mmio_read(regs, SDF_LOCAL_ROWS_WORD) != local_rows ||
        mmio_read(regs, SDF_MAPPING_WORDS_WORD) != mapping_words) {
        print_status(regs);
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <fpga-control-physical-base> <sdf_uram.bin>\n", argv[0]);
        return 2;
    }
    char *end = NULL;
    errno = 0;
    const uint64_t physical_base = strtoull(argv[1], &end, 0);
    if (errno || !end || *end != '\0') {
        fprintf(stderr, "invalid physical base: %s\n", argv[1]);
        return 2;
    }

    struct sdf_image image = {0};
    if (load_image(argv[2], &image) != 0)
        return 1;

    const long page_size = sysconf(_SC_PAGESIZE);
    const uint64_t page_mask = (uint64_t)page_size - 1u;
    const uint64_t map_base = physical_base & ~page_mask;
    const size_t page_offset = (size_t)(physical_base - map_base);
    const size_t map_size = page_offset + APERTURE_BYTES;
    const int mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd < 0) {
        fprintf(stderr, "open /dev/mem: %s\n", strerror(errno));
        free(image.file_data);
        return 1;
    }
    void *mapping = mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                         mem_fd, (off_t)map_base);
    if (mapping == MAP_FAILED) {
        fprintf(stderr, "mmap /dev/mem: %s\n", strerror(errno));
        close(mem_fd);
        free(image.file_data);
        return 1;
    }
    volatile uint32_t *regs = (volatile uint32_t *)((uint8_t *)mapping + page_offset);

    mmio_write(regs, SDF_LOCAL_CELLS_WORD, image.local_cells);
    mmio_write(regs, SDF_CTRL_WORD, 1);
    if (require_progress(regs, 0, 0, 0, 0) != 0)
        goto fail;

    const uint8_t *payload = image.payload;
    for (uint32_t row = 0; row < image.global_rows; ++row) {
        for (uint32_t word = 0; word < image.global_words_per_row; ++word, payload += 4)
            mmio_write(regs, SDF_DATA_WORD, read_le32(payload));
        mmio_write(regs, SDF_DATA_WORD, 0);
    }
    if (require_progress(regs, image.local_rows ? 1 : 2,
                         image.global_rows, 0, 0) != 0)
        goto fail;

    for (uint32_t row = 0; row < image.local_rows; ++row) {
        for (uint32_t word = 0; word < image.local_words_per_row; ++word, payload += 4)
            mmio_write(regs, SDF_DATA_WORD, read_le32(payload));
        mmio_write(regs, SDF_DATA_WORD, 0);
    }
    if (require_progress(regs, 2, image.global_rows, image.local_rows, 0) != 0)
        goto fail;

    for (uint32_t word = 0; word < image.mapping_words; ++word, payload += 4)
        mmio_write(regs, SDF_DATA_WORD, read_le32(payload));
    if (require_progress(regs, 3, image.global_rows, image.local_rows,
                         image.mapping_words) != 0)
        goto fail;
    if ((size_t)(payload - image.payload) != (size_t)image.payload_words * 4u) {
        fprintf(stderr, "internal payload cursor mismatch\n");
        goto fail;
    }

    mmio_write(regs, SDF_CTRL_WORD, 2);
    for (unsigned poll = 0; poll < 1000; ++poll) {
        const uint32_t status = mmio_read(regs, SDF_STATUS_WORD);
        if ((status & 2u) != 0 && (status & 5u) == 0) {
            printf("SDF load complete: global=%" PRIu32 ", local=%" PRIu32
                   ", mapping=%" PRIu32 "\n",
                   image.global_rows, image.local_rows, image.mapping_words);
            munmap(mapping, map_size);
            close(mem_fd);
            free(image.file_data);
            return 0;
        }
    }
    fprintf(stderr, "SDF LOAD_COMMIT did not produce ready\n");

fail:
    print_status(regs);
    munmap(mapping, map_size);
    close(mem_fd);
    free(image.file_data);
    return 1;
}
