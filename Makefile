MODULE_TOPDIR = ../..

PGM = i.sar.interferometry

include $(MODULE_TOPDIR)/include/Make/Script.make

# Unwrapping library loaded by the script through ctypes, installed in
# $(ETC)/$(PGM) (copied by "make install" with the rest of ETCDIR).
UNWRAPLIB = $(ETCDIR)/libsarunwrap$(SHLIB_SUFFIX)

default: script $(UNWRAPLIB)

$(OBJDIR)/sarunwrap_cl.h: sarunwrap_core.h sarunwrap_kernels.cl | $(OBJDIR)
	cat sarunwrap_core.h sarunwrap_kernels.cl | \
	sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/"/' -e 's/$$/\\n"/' > $@

$(UNWRAPLIB): sarunwrap.c sarunwrap.h sarunwrap_core.h $(OBJDIR)/sarunwrap_cl.h | $(ETCDIR)
	$(CC) -O3 -std=gnu11 $(SHLIB_CFLAGS) $(OPENMP_CFLAGS) $(OCLINCPATH) \
		-I$(OBJDIR) -I. -o $@ sarunwrap.c $(SHLIB_LD_FLAGS) -shared \
		$(OCLLIBPATH) $(OCLLIB) $(OPENMP_LIBPATH) $(OPENMP_LIB) $(MATHLIB)
