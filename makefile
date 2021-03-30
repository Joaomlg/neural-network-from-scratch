INTERPRETER := python3

test:
	$(INTERPRETER) -m unittest discover tests/

install:
	$(INTERPRETER) -m pip install -r requirements.txt
