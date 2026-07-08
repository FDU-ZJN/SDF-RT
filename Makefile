.PHONY: sim

sim:
	$(MAKE) -C csrc MODE=fpga generate run
