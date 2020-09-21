# Indexer
The main task of the indexer is to perform different analysis steps and to store the individual features in a database. The system can be extended by several different plugins.


![](doc/indexer.png)


## Plugin Structure

```python
import numpy as np

from plugins import FeaturePlugin
from plugins import export_feature_plugin


@export_feature_plugin('TestFeature')
class TestFeaturePlugin(FeaturePlugin):

    def __init__(self):
        pass

    def call(self, image):
        print('TestPlugin')
        return np.array([1,2,3], dtype=np.float32)
```

```bash
sudo docker run -t --rm -p 8500:8500 -v "/home/springsteinm/tmp/kaggle_iart_densenet_201_export/:/models/kaggle_iart_densenet_201_export" -e MODEL_NAME=kaggle_iart_densenet_201_export tensorflow/serving
```

## Compose Up
First change the bind paths in `docker-compose.yml` to absolut paths on your system.

To test the services simply type:

```bash
docker-compose up --build
```

You can omit the `--build` flag and all images will still be created if they don't exist.
If you make changes to the projekt you will need to either include the `--build` flag or manually run the following:

```bash
docker build . -t iart/indexer
```

### Import Data
First make sure that the elasticsearch container runs and is on the correct network.
(You can do this by simply running `docker-compose up` as described above)
Then do the following (make sure to replace `/path/to/dump/`):

```bash
docker run \
    -it \
    --rm \
    --network=indexer_network \
    --volume=/path/to/dump/:/dump/ \
    elasticdump/elasticsearch-dump:latest \
        multielasticdump \
        --direction dump \
        --prefix iart \
        --input=http://elasticsearch:9200 \
        --output=/dump \
        --includeType=mapping,data
```

### Export Data
First make sure that the elasticsearch container runs and is on the correct network.
(You can do this by simply running `docker-compose up` as described above)
Then do the following (make sure to replace `/path/to/dump/`):

```bash
docker run \
    -it \
    --rm \
    --network=indexer_network \
    --volume=/path/to/dump/:/dump/ \
    elasticdump/elasticsearch-dump:latest \
        multielasticdump \
        --direction load \
        --input=/dump \
        --output=http://elasticsearch:9200 \
        --includeType=mapping,data
```
