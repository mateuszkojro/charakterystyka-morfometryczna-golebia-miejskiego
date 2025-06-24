

## Installation on Windows platform 
1. Optionally set up and activate python environment
   
``` bash
python -m venv .venv
```

``` bash
.\.venv\Scripts\activate
```

1. install required packages
```bash
pip install -r requirements
```

1. In file .venv\Lib\site-packages\seaborn replace line 262:
```python
annotation = ("{:" + self.fmt + "}").format(val)
```
with:
```python
annotation = "{:0.2f}".format(val).replace('.', ',')
```