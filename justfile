bin_name := 'imgtool'

alias r := run
alias b := build
alias i := install

run:
    python imgtool.py

# Define this in case vim calls it
build:
    python imgtool.py

# Define this in case vim calls it
install:
    python imgtool.py

imgtool +args='':
	./imgtool.py {{args}}
