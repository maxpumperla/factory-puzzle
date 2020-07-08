env:
	virtualenv venv && source venv/bin/activate

install:
	pip install -r requirements.txt
	python setup.py develop

offline-data:
	python examples/offline_data_generator.py

run:
	python examples/run_training.py
