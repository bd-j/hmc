test: src/*.f95
	gfortran src/hmc.f95 src/test_hmc.f95 -o bin/test_hmc
