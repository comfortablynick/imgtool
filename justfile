bin_name := 'imgtool'

alias r := run
alias b := build
alias i := install

# Run with optional args
run +args='':
	./imgtool.py {{args}}

# Define this in case vim calls it
build:
    python imgtool.py

# Define this in case vim calls it
install:
    python imgtool.py
