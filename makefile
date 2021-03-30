INTERPRETER := python3.8

default: run

run:
	$(INTERPRETER) main.py

test:
	$(INTERPRETER) -m unittest discover tests/

install:
	$(INTERPRETER) -m pip install -r requirements.txt

