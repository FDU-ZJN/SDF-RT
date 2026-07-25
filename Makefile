.PHONY: sim host-renderer-test sdf-cmodel sdf-cmodel-test sdf-sweep

sim:
	$(MAKE) -C csrc MODE=fpga generate run

host-renderer-test:
	$(MAKE) -C csrc host-renderer-test

sdf-cmodel:
	$(MAKE) -C csrc sdf-cmodel

sdf-cmodel-test:
	$(MAKE) -C csrc sdf-cmodel-test

sdf-sweep:
	$(MAKE) -C csrc sdf-sweep
