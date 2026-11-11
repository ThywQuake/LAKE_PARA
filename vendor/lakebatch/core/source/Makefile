# Compiling source files of LAKE model


 #FC = ifort
 FC = gfortran
 #FC = mpiifort
 #FC = mpif90
 #EXEC = lake.out

 INCLPATHS = 
# -I ../netcdf_ifort_9.1.043/include/ ../netcdf_ifort_9.1.043/lib/libnetcdf.a

 INCLPATH1 = 
# -I ../netcdf_ifort_9.1.043/include/
 INCLPATH2 = 
# -I ../netcdf_ifort_9.1.043/lib/libnetcdf.a

 netcdf_def = #-Dnetcdf_lib
 mpi_def =  #-Dmpi

 ifeq ($(FC),ifort)
   fcompiler = 1 #ifort
   PREPROCESS_KEY = -fpp -DFCOMPILER=$(fcompiler)
   opt_keys = -qopenmp -O3 #-fp-model source
   check_keys = -traceback #-g -check all -fpe-all=0
 endif
 ifeq ($(FC),mpiifort)
   fcompiler = 1 #ifort
   PREPROCESS_KEY = -fpp -DFCOMPILER=$(fcompiler)
   opt_keys =  -O1 #-openmp #-fp-model source
   check_keys = #-g -traceback# -check all -fpe-all=0
 endif
 ifeq ($(FC),mpif90) #mpif90-Intel
   fcompiler = 1 #ifort
   PREPROCESS_KEY = -fpp -DFCOMPILER=$(fcompiler)
   opt_keys =  -O3 #-openmp #-fp-model source
   check_keys =  -traceback #-g -check all -fpe-all=0
 endif
 #ifeq ($(FC),mpif90) #mpif90-gfortran
 #  PREPROCESS_KEY = -cpp 
 #  opt_keys =  -O3 #-openmp #-fp-model source
 #  check_keys =  -fbacktrace #-traceback#-g -check all -fpe-all=0
 #endif
 ifeq ($(FC),gfortran)
   fcompiler = 2 #gfortran
   PREPROCESS_KEY = -cpp -DFCOMPILER=$(fcompiler)
   #opt_keys = -fopenmp -O3 #-fp-model source
   #check_keys = -g -Wall -Warray-temporaries -Wconversion  -fbacktrace -ffree-line-length-0 -fcheck=all -ffpe-trap=invalid,zero,overflow,underflow -finit-real=nan
   #check_keys = -g -ffpe-trap=invalid,zero,overflow,denormal -fcheck=all -fbacktrace
   #check_keys = -g -fopenmp -ffpe-trap=invalid,zero,overflow -fcheck=all -fbacktrace
   #check_keys = -g  -fopenmp -fcheck=all -fbacktrace
   check_keys = -fopenmp -O3 -fbacktrace
 endif

 objfiles_path = ../objfiles/
 mkfile = Makefile
 
 objfiles = \
 \
 $(objfiles_path)data_parameters_mod.o \
 $(objfiles_path)lake_datatypes_mod.o \
 $(objfiles_path)flake_parameters_mod.o \
 $(objfiles_path)SfcFlx_mod.o \
 $(objfiles_path)inout_parameters_mod.o \
 $(objfiles_path)comparams_mod.o \
 $(objfiles_path)inout_mod.o \
 $(objfiles_path)driving_params_mod.o \
 $(objfiles_path)lake_modules.o \
 $(objfiles_path)dzeta_mod.o \
 $(objfiles_path)numerics_mod.o \
 $(objfiles_path)time.o \
 $(objfiles_path)phys_func.o \
 $(objfiles_path)init.o \
 $(objfiles_path)out_mod.o \
 $(objfiles_path)surf_scheme1_mod.o \
 $(objfiles_path)surf_scheme2_mod.o \
 $(objfiles_path)surf_scheme_inm_mod.o \
 $(objfiles_path)surf_scheme3.o \
 $(objfiles_path)heatbalsurf_mod.o \
 $(objfiles_path)t_solver_mod.o \
 $(objfiles_path)diffusion_mod.o \
 $(objfiles_path)oxygen_mod.o \
 $(objfiles_path)seiches_param_mod.o \
 $(objfiles_path)turb_mod.o \
 $(objfiles_path)momentum_mod.o \
 $(objfiles_path)bubble_mod.o \
 $(objfiles_path)methane_mod.o \
 $(objfiles_path)init_var_mod.o \
 $(objfiles_path)soil_mod.o \
 $(objfiles_path)vertadv_mod.o \
 $(objfiles_path)salinity_mod.o \
 $(objfiles_path)control_point_mod.o \
 $(objfiles_path)carbon_dioxide.o \
 $(objfiles_path)phosphorus_mod.o \
 $(objfiles_path)trib.o \
 $(objfiles_path)bathym_mod.o \
 $(objfiles_path)massflux_convection_v10.o \
 $(objfiles_path)convectpar.o \
 $(objfiles_path)snowcalc.o \
 $(objfiles_path)snowtemp.o \
 $(objfiles_path)t_0dim.o \
 $(objfiles_path)lake.o \
 \
 $(objfiles_path)sgetrf.o \
 $(objfiles_path)sgetrs.o \
 $(objfiles_path)xerbla.o \
 $(objfiles_path)sgesv.o \
 $(objfiles_path)ilaenv.o \
 $(objfiles_path)slaswp.o \
 $(objfiles_path)sgetrf2.o \
 $(objfiles_path)slaswp.o \
 $(objfiles_path)lsame.o \
 $(objfiles_path)strsm.o \
 $(objfiles_path)sscal.o \
 $(objfiles_path)isamax.o \
 $(objfiles_path)sgemm.o \
 $(objfiles_path)slamch.o \
 $(objfiles_path)iparam2stage.o \
 $(objfiles_path)ieeeck.o \
 $(objfiles_path)iparmq.o \
 \
 $(objfiles_path)driver_datatypes_mod.o \
 $(objfiles_path)inout_driver_parameters_mod.o \
 $(objfiles_path)driver_interfaces_mod.o \
 $(objfiles_path)driver_parameters_mod.o \
 $(objfiles_path)netcdf_inout.o \
 $(objfiles_path)lakinteract.o \
 $(objfiles_path)driver.o \
 $(objfiles_path)flake_albedo_ref.o \
 $(objfiles_path)flake_derivedtypes.o \
 $(objfiles_path)flake_paramoptic_ref.o \
 $(objfiles_path)flake_configure.o \
 $(objfiles_path)flake_methane.o \
 $(objfiles_path)flake.o \
 $(objfiles_path)src_flake_interface_1D.o \
 $(objfiles_path)postprocessing.o \
 $(objfiles_path)postprocess_surf.o 
 


 all : $(objfiles)

# Compiling module files
# $(objfiles_path)lake_modules.o : ./model/lake_modules.f90 $(mkfile)
#	$(FC) $(opt_keys) $(check_keys) $< -c -o $@ 
# $(objfiles_path)phys_func.o : ./model/phys_func.f90 $(mkfile)
#	$(FC) $(opt_keys) $(check_keys) $< -c -o $@
# $(objfiles_path)inout.o : ./model/inout.f90 $(mkfile)
#	$(FC) $(opt_keys) $(check_keys) $< -c -o $@
# $(objfiles_path)driver_interfaces_mod.o : ./driver/driver_interfaces_mod.f90 $(mkfile)
#	$(FC) $(opt_keys) $(check_keys) $< -c -o $@
# $(objfiles_path)data_parameters_mod.o : ./shared/data_parameters_mod.F90 $(mkfile)
#	$(FC) $(opt_keys) $(check_keys) $< -c -o $@
# $(objfiles_path)flake_parameters_mod.o : ./shared/flake_parameters_mod.F90 $(mkfile)
#	$(FC) $(opt_keys) $(check_keys) $< -c -o $@
# $(objfiles_path)SfcFlx_mod.o : ./shared/SfcFlx_mod.F90 $(mkfile)
#	$(FC) $(opt_keys) $(check_keys) $< -c -o $@

# Preprocessed sources
 $(objfiles_path)driver.o : ./driver/driver.f90 $(mkfile)
	$(FC) $(PREPROCESS_KEY) $(netcdf_def) $(mpi_def) $(opt_keys) $(check_keys) $< -c -o $@
 $(objfiles_path)netcdf_inout.o : ./driver/netcdf_inout.f90 $(mkfile)
	$(FC) $(PREPROCESS_KEY) $(netcdf_def) $(opt_keys) $(check_keys) $< -c -o $@
 $(objfiles_path)postprocess_surf.o : ./driver/postprocess_surf.f90 $(mkfile)
	$(FC) $(PREPROCESS_KEY) $(netcdf_def) $(opt_keys) $(check_keys) $< -c -o $@

# Compiling other source giles
 $(objfiles_path)%.o : ./model/%.f90 $(mkfile)
	$(FC) $(opt_keys) $(check_keys) $(include_paths) $(mpi_def) $(PREPROCESS_KEY) $< -c -o $@
 $(objfiles_path)%.o : ./model/%.f $(mkfile)
	$(FC) $(opt_keys) $(check_keys) $< -c -o $@
 $(objfiles_path)%.o : ./model/%.for $(mkfile)
	$(FC) $(opt_keys) $(check_keys) $< -c -o $@
 $(objfiles_path)%.o : ./driver/%.f90 $(mkfile)
	$(FC) $(opt_keys) $(check_keys) $< -c -o $@
 $(objfiles_path)%.o : ./driver/%.f $(mkfile)
	$(FC) $(opt_keys) $(check_keys) $< -c -o $@
 $(objfiles_path)%.o : ./shared/%.f90 $(mkfile)
	$(FC) $(opt_keys) $(check_keys) $< -c -o $@
 $(objfiles_path)%.o : ./shared/%.F90 $(mkfile)
	$(FC) $(opt_keys) $(check_keys) $< -c -o $@
 $(objfiles_path)%.o : ./shared/%.f $(mkfile)
	$(FC) $(opt_keys) $(check_keys) $< -c -o $@
 $(objfiles_path)%.o : ./Flake/%.f90 $(mkfile)
	$(FC) $(opt_keys) $(check_keys) $< -c -o $@
 $(objfiles_path)%.o : ./Flake/%.F90 $(mkfile)
	$(FC) $(opt_keys) $(check_keys) $< -c -o $@
 $(objfiles_path)%.o : ./lapack/%.f $(mkfile)
	$(FC) $(opt_keys) $(check_keys) $< -c -o $@
 $(objfiles_path)%.o : ./lapack/%.F $(mkfile)
	$(FC) $(opt_keys) $(check_keys) $< -c -o $@

# Cleaning *.o and *.mod files
 clean :
	rm -f $(objfiles_path)*.o
	rm -f *.mod

