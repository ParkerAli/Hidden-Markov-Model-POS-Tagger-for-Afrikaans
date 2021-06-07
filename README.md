# POS tagging using HMMs

HMM is a python script that allows for part-of-speech tagging using bigram and trigram [Hidden-Markov models](https://en.wikipedia.org/wiki/Hidden_Markov_model) using [laplace smoothing](https://en.wikipedia.org/wiki/Additive_smoothing). 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements for HMM.

### Linux/Mac
````bash
pip install virtualenv # if you don't already have virtualenv installed
virtualenv venv # to create your new environment (called 'venv' here)
source venv/bin/activate # to enter the virtual environment
pip install -r requirements.txt # to install the required packages for running HMM.py
````

### Windows
````powershell
pip install virtualenv # if you don't already have virtualenv installed
virtualenv venv # to create your new environment (called 'venv' here)
source .\venv\bin\Scripts\activate # to enter the virtual environment
pip install -r requirements.txt # to install the required packages for running HMM.py
````

## Usage

```bash
python HMM.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
