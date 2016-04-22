init:
	pip install -r requirements.txt
	pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.8.0rc0-py2-none-any.whl

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	
test:
	tox
