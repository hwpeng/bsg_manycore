.DEFAULT_GOAL = all

BSG_MANYCORE_DIR := $(shell git rev-parse --show-toplevel)
# By convention, basejump_stl is in the same directory as $(BSG_MANYCORE_DIR)
BASEJUMP_STL_DIR := $(BSG_MANYCORE_DIR)/../basejump_stl

# Sets "VCS" variable
include $(BSG_MANYCORE_DIR)/../bsg_cadenv/cadenv.mk

# Include source lists
include arch_filelist.mk
include sim_filelist.mk

VCS_INCLUDES += $(foreach inc,$(VINCLUDES),+incdir+"$(inc)")
VCS_DEFINES  += $(foreach def,$(VDEFINES),+define+"$(def)")
VCS_SOURCES  += $(VHEADERS) $(VSOURCES) 
VCS_FLAGS    += +v2k -sverilog -full64 -timescale=1ps/1ps \
	 	+lint=all,noSVA-UA,noSVA-NSVU,noVCDE,noNS -top spmd_testbench

VCS_CFLAGS   =  -cpp g++
VCS_CFLAGS   += -CFLAGS "-std=c++11 -g -Wall"
VCS_CFLAGS   += -CFLAGS "-fPIC"
VCS_CFLAGS 	 += -CFLAGS "-I$(BASEJUMP_STL_DIR)/imports/DRAMSim3/src"
VCS_CFLAGS   += -CFLAGS "-I$(BASEJUMP_STL_DIR)/imports/DRAMSim3/ext/headers"
VCS_CFLAGS   += -CFLAGS "-I$(BASEJUMP_STL_DIR)/imports/DRAMSim3/ext/fmt/include"
VCS_CFLAGS   += -CFLAGS "-I$(BASEJUMP_STL_DIR)/bsg_test"
VCS_CFLAGS   += -CFLAGS "-DFMT_HEADER_ONLY=1"
VCS_CFLAGS   += -CFLAGS "-DBASEJUMP_STL_DIR=$(BASEJUMP_STL_DIR)"

DEFAULT_MACHINES = 4x4 4x4_dmc 16x8 4x4_fast_fake 4x4_gs

.PHONY: all clean

DEFAULT_TARGETS = $(foreach machine, $(DEFAULT_MACHINES),$(machine)/simv)
DEFAULT_DEBUG_TARGETS = $(foreach machine, $(DEFAULT_MACHINES),$(machine)/simv-debug)
all: $(DEFAULT_TARGETS) $(DEFAULT_DEBUG_TARGETS)

# set_machine_variables includes the Makefile.machine.include file and sets the
# VCS_DEFINES that define the architecture for a machine. This should be called
# from inside of the rule that builds the machine-specific executable for a
# machine so that the *correct* Makfile.machine.include can be included
define set_machine_variables
	$(eval include $1)
	$(eval VCS_DEFINES := +define+BSG_MACHINE_GLOBAL_X=${BSG_MACHINE_GLOBAL_X})
	$(eval VCS_DEFINES += +define+BSG_MACHINE_GLOBAL_Y=${BSG_MACHINE_GLOBAL_Y})
	$(eval VCS_DEFINES += +define+BSG_MACHINE_VCACHE_SET=${BSG_MACHINE_VCACHE_SET})
	$(eval VCS_DEFINES += +define+BSG_MACHINE_VCACHE_WAY=${BSG_MACHINE_VCACHE_WAY})
	$(eval VCS_DEFINES += +define+BSG_MACHINE_VCACHE_BLOCK_SIZE_WORDS=${BSG_MACHINE_VCACHE_BLOCK_SIZE_WORDS})
	$(eval VCS_DEFINES += +define+BSG_MACHINE_VCACHE_DMA_DATA_WIDTH=${BSG_MACHINE_VCACHE_DMA_DATA_WIDTH})
	$(eval VCS_DEFINES += +define+BSG_MACHINE_VCACHE_MISS_FIFO_ELS=${BSG_MACHINE_VCACHE_MISS_FIFO_ELS})
	$(eval VCS_DEFINES += +define+BSG_MACHINE_DRAM_SIZE_WORDS=${BSG_MACHINE_DRAM_SIZE_WORDS})
	$(eval VCS_DEFINES += +define+BSG_MACHINE_DRAM_INCLUDED=${BSG_MACHINE_DRAM_INCLUDED})
	$(eval VCS_DEFINES += +define+BSG_MACHINE_MAX_EPA_WIDTH=${BSG_MACHINE_MAX_EPA_WIDTH})
	$(eval VCS_DEFINES += +define+BSG_MACHINE_MEM_CFG=${BSG_MACHINE_MEM_CFG})
	$(eval VCS_DEFINES += +define+BSG_MACHINE_BRANCH_TRACE_EN=${BSG_MACHINE_BRANCH_TRACE_EN})
	$(eval VCS_DEFINES += +define+BSG_MACHINE_HETERO_TYPE_VEC="${BSG_MACHINE_HETERO_TYPE_VEC}")
	# These define are required by mobile_ddr.v.
	# density     	= 2048 Mbit
	# speed grade 	= 5
	# organization 	= x16
	# allocation    = FULL_MEM
	$(eval VCS_DEFINES += +define+den2048Mb+sg5+x16+FULL_MEM)
endef

%/simv : %/Makefile.machine.include $(VSOURCES) $(CSOURCES) $(VINCLUDES) $(VHEADERS) 
	$(call set_machine_variables,$<)
	$(VCS) $(VCS_FLAGS) $(VCS_CFLAGS) -o $@ $(VCS_INCLUDES) $(VCS_DEFINES) $(CSOURCES) $(VCS_SOURCES) | tee -i $(dir $@)/build.log

%/simv-debug : %/Makefile.machine.include $(VSOURCES) $(CSOURCES) $(VINCLUDES) $(VHEADERS) 
	$(call set_machine_variables,$<)
	$(eval VCS_FLAGS += -debug_pp +vcs+vcdpluson) # Debug adds these two variables to generate waveforms
	$(VCS) $(VCS_FLAGS) $(VCS_CFLAGS) -o $@ $(VCS_INCLUDES) $(VCS_DEFINES) $(CSOURCES) $(VCS_SOURCES) | tee -i $(dir $@)/build_debug.log

clean:
	rm -rf csrc
	rm -rf */*.log */*.daidir */simv */simv-debug
	rm -rf stack.info.*
	rm -f vc_hdrs.h
