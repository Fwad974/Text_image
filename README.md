# text-image search engine
This text-image search engine app.

by running the application it automatically downloads [coco](http://images.cocodataset.org/zips/val2014.zip) dataset.


## Usage

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

```bash
pip install -r req.txt
```

for using fastapi application, send your requests to 0.0.0.0:8000/model/predict by using post method, and your description of the object should be named as description and your request should be sth like:
```python
{
    "description":"your description"
}
```

