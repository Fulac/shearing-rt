TARGET         = main
FILES          = *.cu *.h Makefile
VER            = `date +%y%m%d`

CUDAARCH       = compute_61
CUDACODE       = sm_61

NVCC           = nvcc
NVCCFLAGS      = -gencode arch=$(CUDAARCH),code=$(CUDACODE)
NVCCLIBS       = -lcufft -lcudadevrt -lcurand
# IGNOREWARNING  = -Xcompiler "/wd 4819 /O2"

NVCCFLAGS += -O3

ifeq ($(dble),on)
  NVCCFLAGS += -DDBLE
endif

.SUFFIXES:
.SUFFIXES: .cu .o

.cu.o:
	$(NVCC) -dc $(NVCCFLAGS) $(IGNOREWARNING) $<

# main program
$(TARGET): main.o output.o tint.o shear.o fields.o four.o fft.o
	$(NVCC) $^ -o $@ $(NVCCFLAGS) $(NVCCLIBS)

main.o: shear.o shear.h tint.o tint.h
output.o: shear.o shear.h tint.o tint.h
tint.o: four.o four.h shear.o shear.h
shear.o: fields.o fields.h
fields.o: four.o four.h
four.o: fft.o fft.h
fft.o: cmplx.h

clean:
	rm -rf *.o *.exp *.lib *~ \#*

distclean: clean
	rm -rf $(TARGET)

tar:
	@echo $(TARGET)-$(VER) > .package
	@-rm -fr `cat .package`
	@mkdir `cat .package`
	@ln $(FILES) `cat .package`
	tar cvf - `cat .package` | bzip2 -9 > `cat .package`.tar.bz2
	@-rm -fr `cat .package` .package
