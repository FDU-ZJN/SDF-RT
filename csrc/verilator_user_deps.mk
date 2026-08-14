# Extra prerequisites for objects linked through Verilator's generated makefile.
ifneq ($(strip $(RT_CUDA_OBJ)),)
$(RT_VERILATOR_TARGET): $(RT_CUDA_OBJ)
endif
