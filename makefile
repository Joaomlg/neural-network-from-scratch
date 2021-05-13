INTERPRETER := python3
MAIN := main.py

default: run

run:
	$(INTERPRETER) $(MAIN)

train:
	$(INTERPRETER) $(MAIN) train

test:
	$(INTERPRETER) -m unittest discover tests/

install:
	$(INTERPRETER) -m pip install -r requirements.txt
