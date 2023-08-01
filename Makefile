python_implementation = cpython
version = 39
architecture = x86_64
os = linux-gnu
module_name = systolic_injector

out_name = $(module_name).$(python_implementation)-$(version)-$(architecture)-$(os).so

CFLAGS = -I/usr/include/python3.9

default: build

build: $(module_name)module.o
	gcc -shared build/$(module_name)module.o -o build/$(out_name)

$(module_name)module.o:
	if [ ! -d build ]; then mkdir make_build; fi
	gcc -fPIC -c $(CFLAGS) $(module_name)module.c -o build/$(module_name)module.o

clean:
	rm -r make_build