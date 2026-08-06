# KV260 SDF URAM Loader

`sdf_uram_loader` programs the packed SDF memories before rendering starts.
It consumes `sdf_uram.bin`, which the simulation memory-export step emits next
to the existing `.mem` files.

Build on the target Linux system:

```sh
make
```

Load an image as root, using the physical base assigned to `fpga_top` in the
Vivado address map:

```sh
sudo ./sdf_uram_loader 0xa0000000 /path/to/sdf_uram.bin
```

The example base is illustrative; use the address in the exported XSA/device
tree. The IP aperture is 1 MiB.

## Register Protocol

All offsets below are byte offsets from the IP base.

| Offset | Name | Access | Meaning |
| ---: | --- | --- | --- |
| `0xFFF80` | `SDF_CTRL` | W | `1`: begin/restart load, `2`: validate and commit |
| `0xFFF84` | `SDF_STATUS` | R | bit 0 active, bit 1 ready, bit 2 error, bits 4:3 phase |
| `0xFFF88` | `SDF_LOCAL_CELLS` | RW | active local-cell count, written before begin |
| `0xFFF8C` | `SDF_GLOBAL_ROWS` | R | committed 72-bit global rows |
| `0xFFF90` | `SDF_LOCAL_ROWS` | R | committed 576-bit local rows (one 4x4x4 subcell) |
| `0xFFF94` | `SDF_MAPPING_WORDS` | R | accepted packed mapping words |
| `0xFFF98` | `SDF_DATA` | W | sequential data stream |

The stream order is fixed: 4096 global rows, `local_cells` local rows, then
16384 mapping words. A global row is three data writes plus one commit write. A
local row is eighteen data writes plus one commit write. Mapping rows need one write.
Only full 32-bit writes are accepted. Rendering setup and ray input stay blocked
until `LOAD_COMMIT` succeeds.
