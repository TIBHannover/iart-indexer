# iART Indexer

The main task of the indexer is to perform different analysis steps and to store the individual features in a database. The system can be extended by plugins.

![](doc/indexer.png)


## Plugin structure

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
