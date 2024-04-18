OBJS1=	fmech_subs.o \
        pol_subs.o \
        station_subs.o \
        uncert_subs.o \
	util_subs.o

OBJS2=	fmech_subs.o \
        pol_subs.o \
        station_subs.o \
        uncert_subs.o \
	util_subs.o \
        vel_subs.o

OBJS3=	fmamp_subs.o \
        pol_subs.o \
        station_subs.o \
        uncert_subs.o \
	util_subs.o \
        vel_subs.o

OBJS4=	fmech_subs.o \
        pol_subs.o \
        station_subs_5char.o \
        uncert_subs.o \
	util_subs.o \
        vel_subs.o

hash_driver1: Makefile hash_driver1.f $(OBJS1)
	gfortran  hash_driver1.f $(OBJS1) -O -o hash_driver1

hash_driver2: Makefile hash_driver2.f $(OBJS2)
	gfortran  hash_driver2.f $(OBJS2) -O -o hash_driver2

hash_driver3: Makefile hash_driver3.f $(OBJS3)
	gfortran  hash_driver3.f $(OBJS3) -O -o hash_driver3

hash_driver4: Makefile hash_driver4.f $(OBJS4)
	gfortran  hash_driver4.f $(OBJS4) -O -o hash_driver4

hash_driver5: Makefile hash_driver5.f $(OBJS2)
	gfortran  hash_driver5.f $(OBJS2) -O -o hash_driver5

pol_subs.o : Makefile pol_subs.f
	gfortran pol_subs.f -O -c

fmech_subs.o : Makefile fmech_subs.f
	gfortran fmech_subs.f -O -c

fmamp_subs.o : Makefile fmamp_subs.f
	gfortran fmamp_subs.f -O -c

station_subs.o : Makefile station_subs.f
	gfortran station_subs.f -O -c

station_subs_5char.o : Makefile station_subs_5char.f
	gfortran station_subs_5char.f -O -c

uncert_subs.o : Makefile uncert_subs.f
	gfortran uncert_subs.f -O -c

util_subs.o : Makefile util_subs.f
	gfortran util_subs.f -O -c

vel_subs.o : Makefile vel_subs.f
	gfortran vel_subs.f -O -c



