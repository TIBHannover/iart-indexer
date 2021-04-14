import logging
import uuid
import json
from typing import Dict, List


class Aggregator:
    def __init__(
        self,
        database,
    ):
        super().__init__()
        self.database = database

    def text_count(self, query=None, nested_field="meta", field_name=None, size=5):
        if field_name is not None:
            body = {
                "aggs": {
                    f"{nested_field}_nested": {
                        "nested": {"path": f"{nested_field}"},
                        "aggs": {
                            f"{nested_field}_name_filter": {
                                "filter": {"term": {f"{nested_field}.name": field_name}},
                                "aggs": {
                                    f"{nested_field}_name_filter_aggr": {
                                        "terms": {"size": size, "field": f"{nested_field}.value_str.keyword"}
                                    }
                                },
                            }
                        },
                    }
                }
            }

            if query is not None:
                body["query"] = query

            aggr = self.database.raw_aggregate(body=body)
            # logging.info(aggr)
            for x in aggr["aggregations"][f"{nested_field}_nested"][f"{nested_field}_name_filter"][
                f"{nested_field}_name_filter_aggr"
            ]["buckets"]:
                yield {"name": x["key"], "value": x["doc_count"]}
        else:
            body = {
                "aggs": {
                    f"{nested_field}_nested": {
                        "nested": {"path": f"{nested_field}"},
                        "aggs": {
                            f"{nested_field}_name_filter_aggr": {
                                "terms": {"size": size, "field": f"{nested_field}.value_str.keyword"}
                            }
                        },
                    }
                }
            }

            if query is not None:
                body["query"] = query

            aggr = self.database.raw_aggregate(body=body)
            # logging.info(aggr)
            for x in aggr["aggregations"][f"{nested_field}_nested"][f"{nested_field}_name_filter_aggr"]["buckets"]:
                yield {"name": x["key"], "value": x["doc_count"]}

    def __call__(self, query, field_names, size=250):
        logging.info("Aggregator")
        # logging.info(query)
        # logging.info(field_names)

        aggregations = []
        for field_name in field_names:
            # logging.info(field_name)

            field_path = field_name.split(".")
            if field_path[0] not in ["meta", "origin"]:
                continue
            if len(field_path) == 1:
                aggregation = list(self.text_count(query, nested_field=field_path[0], size=size))
                if aggregation and len(aggregation) > 0:
                    aggregations.append({"field_name": field_name, "entries": aggregation})

            if len(field_path) == 2:
                aggregation = list(
                    self.text_count(query, nested_field=field_path[0], field_name=field_path[1], size=size)
                )
                if aggregation and len(aggregation) > 0:
                    aggregations.append({"field_name": field_name, "entries": aggregation})

        return aggregations
