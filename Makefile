ifeq ( $(OS), Windows_NT )
	include Makefile.win
else
	include Makefile.linux
endif
