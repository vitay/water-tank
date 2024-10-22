# Water Tank

Water tanks are plastic reservoirs.

## Installation


Dependencies:

* `python`>= 3.10
* `numpy` >= 1.21
* `scipy`>= 1.11
* `cython` >= 3.0

Install the latest stable version:

```bash
pip install git+https://github.com/vitay/water-tank.git@main
```

The legacy branch contains the code used in March 2024:

```bash
pip install git+https://github.com/vitay/water-tank.git@legacy
```

## Documentation

<https://julien-vitay.net/water-tank/>

To build the documentation, you need `quartodoc` and `quarto` installed:

```python
cd docs
quartodoc build # to generate the API
quarto render . # to render the website
quarto publish gh-pages # to push to github pages
```


## License

`water-tank` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
