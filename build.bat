@echo off
echo Compiling MPI programs...

echo Compiling static version...
mpicc -o static.exe static_scheduling.c

echo Compiling dynamic version...
mpicc -o dynamic.exe dynamic_scheduling.c

echo Done!
pause