bsg_tiles_X = 3
bsg_tiles_Y = 3

ROM_NAME = bsg_manycore_io_complex_rom

RISCV_GCC_EXTRA_OPTS += -O2 #-DBSG_TOKEN_QUEUE_SHORT
# -funroll-loops
OBJECT_FILES=main.o bsg_set_tile_x_y.o
include ../Makefile.include

all: main.run

main.riscv: $(LINK_SCRIPT)  $(OBJECT_FILES) $(SPMD_COMMON_OBJECTS) $(BSG_MANYCORE_LIB) ../common/crt.o
	$(RISCV_LINK) $(OBJECT_FILES) $(SPMD_COMMON_OBJECTS) -L. "-l:$(BSG_MANYCORE_LIB)" -o $@ $(RISCV_LINK_OPTS) 

main.o: Makefile

main.hex: main.riscv
	$(RISCV_ELF2HEX) $(BSG_MANYCORE_BANKS) $(BSG_MANYCORE_BANKSIZE) $< > $@

main.bin: main.hex
	python $(HEX2BIN) $< 32 > $@

$(ROM_NAME).v: main.bin
	python $(BSG_ROM_GEN) main.bin $(ROM_NAME) > $(ROM_NAME).v

rom: $(ROM_NAME).v


verify: main.c
	gcc -DVERIFY main.c -std=c99 -o verify
	./verify > verify.vals

# typing make foo.check will strip out the sum codes
%.check: 
	grep 'data ' trace | awk '{print $$9}' | cut -b4-9 > $@

include ../../mk/Makefile.tail_rules








