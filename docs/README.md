# Documentation for water-tank

The documentation is at <https://julien-vitay.net/water-tank/>. 

Generating the documentation requires **Quarto** (<https://quarto.org>) and `quartodoc` (<https://github.com/machow/quartodoc>, `pip install quartodoc`) for the API. 

First build the API:

```bash
cd docs
quartodoc build
```

Preview the doc:

```bash
quarto preview
```

Push it to github:

```bash
quarto publish gh-pages
```
